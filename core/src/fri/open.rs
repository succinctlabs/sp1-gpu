use std::marker::PhantomData;

use p3_challenger::FieldChallenger;
use p3_field::two_adic_coset_zerofier;
use p3_field::Field;
use tracing::trace_span;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::CanObserve;
use p3_challenger::CanSample;
use p3_challenger::CanSampleBits;
use p3_challenger::GrindingChallenger;
use p3_commit::{OpenedValues, Pcs};
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractExtensionField;
use p3_field::AbstractField;
use p3_field::TwoAdicField;
use p3_fri::fold_even_odd;
use p3_fri::CommitPhaseProofStep;
use p3_fri::FriConfig;
use p3_fri::FriProof;
use p3_fri::QueryProof;
use p3_fri::{BatchOpening, TwoAdicFriPcsProof};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Hash;
use p3_util::log2_ceil_usize;
use sp1_core::stark::Challenge;
use sp1_core::stark::Challenger;
use sp1_core::stark::{OpeningProof, PcsProverData};
use sp1_core::utils::baby_bear_poseidon2::ChallengeMmcs;
use sp1_core::utils::log2_strict_usize;
use sp1_core::utils::BabyBearPoseidon2;
use sp1_core::utils::{InnerVal, InnerValMmcs};

use crate::device::buffer::DeviceBuffer;
use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::device::CudaSync;
use crate::matrix::ColMajorMatrixDevice;
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::baby_bear_gpu::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH;
use crate::stark::BabyBearPoseidon2Config;
use crate::stark::GpuProverData;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type SC = BabyBearPoseidon2;

#[derive(Clone, Copy, Debug)]
pub struct FriCpuOpeningProver<SC>(PhantomData<SC>);

impl<SC: BabyBearPoseidon2Config> FriCpuOpeningProver<SC> {
    #[allow(clippy::type_complexity)]
    pub fn open(
        &self,
        pcs: &SC::Pcs,
        rounds: Vec<(&PcsProverData<SC>, Vec<Vec<SC::Challenge>>)>,
        challenger: &mut SC::Challenger,
    ) -> (OpenedValues<SC::Challenge>, OpeningProof<SC>) {
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::open(pcs, rounds, challenger)
    }
}

impl<SC: BabyBearPoseidon2Config> Default for FriCpuOpeningProver<SC> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FriGpuOpeningProver<SC>(PhantomData<SC>);

impl<SC: BabyBearPoseidon2Config> FriGpuOpeningProver<SC> {
    #[allow(clippy::type_complexity)]
    pub fn open(
        &self,
        pcs: &SC::Pcs,
        rounds: Vec<(&GpuProverData<SC>, Vec<Vec<SC::Challenge>>)>,
        challenger: &mut SC::Challenger,
    ) -> (OpenedValues<SC::Challenge>, OpeningProof<SC>) {
        let alpha: Challenge<SC> = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| (data.leaves.iter().collect_vec(), points))
            .collect_vec();
        let mats = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .collect_vec();

        let global_max_height = mats.iter().map(|m| m.height).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        let mut reduced_openings: [Option<Vec<SC::Challenge>>; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let compute_reduce_openings_span = trace_span!("Compute reduced openings").entered();
        // let (tx, rx) = mpsc::channel();

        // Interpolate cosets.

        let interpolate_cosets_span = tracing::debug_span!("Interpolate cosets").entered();

        // Values for coset interpolation.
        let shift = InnerVal::generator();
        let mut poly_evals = vec![];
        let mut coset_heights = vec![];
        let mut coset_log_heights = vec![];
        let mut shifts = vec![];
        let mut g_values = vec![];
        let mut opening_points = vec![];
        let mut barycentric_scalars = vec![];
        let mut total_polys = 0;

        // Values for inverse denominators.
        let mut nums_rows = vec![];
        let mut log_nums_rows = vec![];
        let mut inv_indices = vec![];
        let mut thread_generator_powers = vec![];
        let mut inv_offset = 0;
        let mut num_points = 0;
        let mut points_for_inv = vec![];
        let mut shifts_for_inv = vec![];

        // Values for computing reduced openings.
        let mut alpha_pow_offsets = vec![];
        let mut matrices_for_openings = vec![];
        let mut ys_indices = vec![];
        let mut ys_index = 0;

        let get_data_for_device_span = tracing::debug_span!("Get data for device").entered();
        mats_and_points.iter().for_each(|(mats, points)| {
            mats.iter()
                .zip(points.iter())
                .for_each(|(mat, points_for_mat)| {
                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let coset_height = mat.height() >> pcs.fri_config().log_blowup;
                    let cols = mat.width();
                    let num_polys = cols * points_for_mat.len();
                    total_polys += num_polys;
                    let coset_log_height = log2_strict_usize(coset_height);
                    let g = BabyBear::two_adic_generator(coset_log_height);
                    let denominator = F::from_canonical_usize(coset_height)
                        * shift.exp_u64(coset_height as u64 - 1);

                    g_values.extend((0..num_polys).flat_map(|_| g.powers().take(32)));
                    shifts.extend((0..num_polys).map(|_| shift));
                    coset_log_heights.extend((0..num_polys).map(|_| coset_log_height));
                    coset_heights.extend((0..num_polys).map(|_| coset_height));

                    let log_height = log2_strict_usize(mat.height());
                    let height_g = BabyBear::two_adic_generator(log_height);
                    let g_thread_gen = height_g.powers().take(1024).collect::<Vec<_>>();

                    for point in points_for_mat {
                        nums_rows.push(mat.height());
                        log_nums_rows.push(log_height);
                        inv_indices.push(inv_offset);
                        inv_offset += mat.height();
                        points_for_inv.push(*point);
                        num_points += 1;
                        thread_generator_powers.extend_from_slice(&g_thread_gen);
                        shifts_for_inv.push(shift);

                        opening_points.extend((0..cols).map(|_| *point));
                        let zerofier =
                            two_adic_coset_zerofier(coset_log_height, EF::from_base(shift), *point);
                        let barycentric_scalar = zerofier * denominator.inverse();
                        barycentric_scalars.extend((0..cols).map(|_| barycentric_scalar));

                        let num_reduced_at_height = num_reduced[log_height];
                        num_reduced[log_height] += mat.width();
                        let alpha_pow_offset = alpha.exp_u64(num_reduced_at_height as u64);
                        alpha_pow_offsets.push(alpha_pow_offset);
                        matrices_for_openings.push(mat.view());

                        ys_indices.push(ys_index);
                        ys_index += mat.width();
                    }

                    for _ in 0..points_for_mat.len() {
                        let poly_evals_iter = unsafe {
                            (0..cols).map(|col| mat.values.as_ptr().add(col * mat.height))
                        };
                        poly_evals.extend(poly_evals_iter);
                    }
                })
        });
        get_data_for_device_span.exit();

        let num_points = points_for_inv.len();
        assert_eq!(nums_rows.len(), num_points);
        assert_eq!(log_nums_rows.len(), num_points);
        assert_eq!(inv_indices.len(), num_points);
        assert_eq!(thread_generator_powers.len(), num_points * 1024);
        assert_eq!(points_for_inv.len(), num_points);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let mut inv_denominators = DeviceBuffer::<EF>::with_capacity(inv_offset);
        let inv_indices_device = inv_indices.to_device();
        unsafe {
            inv_denominators.set_max_len();
            let nums_rows = nums_rows.to_device();
            let log_nums_rows = log_nums_rows.to_device();
            let thread_generator_powers = thread_generator_powers.to_device();
            let points_for_inv = points_for_inv.to_device();
            let shifts_for_inv = shifts_for_inv.to_device();
            opening_gpu::compute_inverse_denominators(
                global_max_height,
                num_points,
                inv_indices_device.as_ptr(),
                nums_rows.as_ptr(),
                log_nums_rows.as_ptr(),
                shifts_for_inv.as_ptr(),
                thread_generator_powers.as_ptr(),
                points_for_inv.as_ptr(),
                inv_denominators.as_mut_ptr(),
            );
        }

        assert_eq!(poly_evals.len(), total_polys);
        assert_eq!(coset_heights.len(), total_polys);
        assert_eq!(coset_log_heights.len(), total_polys);
        assert_eq!(shifts.len(), total_polys);
        assert_eq!(g_values.len(), total_polys * 32);
        assert_eq!(opening_points.len(), total_polys);
        assert_eq!(barycentric_scalars.len(), total_polys);

        let ys_output_buffer = {
            let mut ys_output_buffer = DeviceBuffer::<EF>::with_capacity(total_polys);

            let poly_evals = poly_evals.to_device();
            let coset_heights = coset_heights.to_device();
            let coset_log_heights = coset_log_heights.to_device();
            let shifts = shifts.to_device();
            let g_values = g_values.to_device();
            let opening_points = opening_points.to_device();
            let barycentric_scalars = barycentric_scalars.to_device();

            unsafe {
                ys_output_buffer.set_max_len();
                opening_gpu::interpolate_cosets_raw(
                    poly_evals.as_ptr(),
                    poly_evals.len(),
                    coset_heights.as_ptr(),
                    coset_log_heights.as_ptr(),
                    shifts.as_ptr(),
                    opening_points.as_ptr(),
                    barycentric_scalars.as_ptr(),
                    g_values.as_ptr(),
                    ys_output_buffer.as_mut_ptr(),
                );
            }

            ys_output_buffer
        };

        interpolate_cosets_span.exit();

        let ys_out_host_span = tracing::debug_span!("ys_out_host").entered();
        let ys_output = ys_output_buffer.to_host();
        ys_out_host_span.exit();

        assert_eq!(ys_output.len(), total_polys);

        let compute_openings_span = tracing::debug_span!("Compute opened values").entered();
        let mut point_index = 0;
        let all_opened_values = {
            let mut reduced_openings_device = DeviceBuffer::<EF>::with_capacity(inv_offset);

            // Compute openings fused.
            let compute_reduced_openings_span =
                tracing::debug_span!("Compute reduced openings on device").entered();
            let alpha_pow_offsets_device = alpha_pow_offsets.to_device();
            let matrices_for_openings = matrices_for_openings.to_device();
            let ys_indices = ys_indices.to_device();

            unsafe {
                reduced_openings_device.set_max_len();
                opening_gpu::compute_reduced_openings(
                    matrices_for_openings.as_ptr(),
                    global_max_height,
                    points_for_inv.as_ptr(),
                    num_points,
                    inv_indices_device.as_ptr(),
                    inv_denominators.as_ptr(),
                    alpha,
                    alpha_pow_offsets_device.as_ptr(),
                    ys_output_buffer.as_ptr(),
                    ys_indices.as_ptr(),
                    reduced_openings_device.as_mut_ptr(),
                );
            }

            compute_reduced_openings_span.exit();

            let copy_reduced_openings_span =
                tracing::debug_span!("Copy reduced openings to host").entered();
            let reduced_opening_host = reduced_openings_device.to_host();
            copy_reduced_openings_span.exit();

            let copy_values_span = tracing::debug_span!("Copy opened values to host").entered();
            let mut index = 0;
            let all_opened_values = mats_and_points
                .into_iter()
                .map(|(mats, points)| {
                    mats.into_iter()
                        .zip(points)
                        .map(|(mat, points_for_mat)| {
                            let log_height = log2_strict_usize(mat.height);

                            let opened_values_for_mat = points_for_mat
                                .iter()
                                .map(|_| {
                                    let ys_host = ys_output[index..index + mat.width()].to_vec();
                                    index += mat.width();

                                    let inv_index = inv_indices[point_index];
                                    point_index += 1;

                                    let reduced_opening_for_log_height =
                                        &reduced_opening_host[inv_index..inv_index + mat.height()];

                                    let openings_at_height = reduced_openings[log_height]
                                        .get_or_insert_with(|| {
                                            vec![SC::Challenge::zero(); 1 << log_height]
                                        });
                                    for (i, x) in reduced_opening_for_log_height.iter().enumerate()
                                    {
                                        openings_at_height[i] += *x;
                                    }

                                    ys_host
                                })
                                .collect::<Vec<_>>();
                            opened_values_for_mat
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            copy_values_span.exit();
            all_opened_values
        };
        compute_openings_span.exit();

        compute_reduce_openings_span.exit();

        let (fri_proof, query_indices) = tracing::debug_span!("Fri Proof")
            .in_scope(|| prove(pcs.fri_config(), &reduced_openings, challenger));

        let query_openings_span = tracing::debug_span!("Compute query openings").entered();
        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        let max_height = data.leaves.iter().map(|m| m.height()).max().unwrap();
                        let log_max_height = log2_ceil_usize(max_height);
                        let bits_reduced = log_global_max_height - log_max_height;
                        let reduced_index = index >> bits_reduced;
                        let (opened_values, opening_proof) = open_batch(reduced_index, data);
                        BatchOpening::<SC::Val, InnerValMmcs> {
                            opened_values,
                            opening_proof,
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        query_openings_span.exit();

        (
            all_opened_values,
            TwoAdicFriPcsProof {
                fri_proof,
                query_openings,
            },
        )
    }
}

fn open_batch(
    index: usize,
    prover_data: &FieldMerkleTreeGpu<
        BabyBear,
        [BabyBear; DIGEST_WIDTH],
        CudaSync<ColMajorMatrixDevice<BabyBear>>,
    >,
) -> (Vec<Vec<F>>, Vec<[F; DIGEST_WIDTH]>) {
    let max_height = prover_data.leaves.iter().map(|m| m.height()).max().unwrap();
    let log_max_height = log2_ceil_usize(max_height);

    let openings = prover_data
        .leaves
        .iter()
        .map(|matrix| {
            let log2_height = log2_ceil_usize(matrix.height());
            let bits_reduced = log_max_height - log2_height;
            let reduced_index = index >> bits_reduced;
            let mut output_device: DeviceBuffer<F> = DeviceBuffer::with_capacity(matrix.width());
            unsafe {
                output_device.set_len(matrix.width());
                opening_gpu::fetch_row(matrix.view(), reduced_index, output_device.as_mut_ptr());
            }
            output_device.to_host()
        })
        .collect_vec();

    let proof = (0..log_max_height)
        .map(|i| {
            let start = (index >> i) ^ 1;
            let end = start + 1;
            prover_data.digest_layers[i][start..end].to_host()[0]
        })
        .collect();

    (openings, proof)
}

pub fn prove(
    config: &FriConfig<ChallengeMmcs>,
    input: &[Option<Vec<EF>>; 32],
    challenger: &mut Challenger<SC>,
) -> (FriProof<EF, ChallengeMmcs, F>, Vec<usize>) {
    let log_max_height = input.iter().rposition(Option::is_some).unwrap();

    let commit_phase_result = trace_span!("Commit phase")
        .in_scope(|| commit_phase(config, input, log_max_height, challenger));

    let pow_witness =
        trace_span!("POW witness").in_scope(|| challenger.grind(config.proof_of_work_bits));

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs_span = trace_span!("Compute query proofs").entered();
    let query_proofs = query_indices
        .iter()
        .map(|&index| answer_query(&commit_phase_result.data, index))
        .collect::<Vec<_>>();
    query_proofs_span.exit();

    (
        FriProof {
            commit_phase_commits: commit_phase_result.commits,
            query_proofs,
            final_poly: commit_phase_result.final_poly,
            pow_witness,
        },
        query_indices,
    )
}

pub fn commit_phase(
    config: &FriConfig<ChallengeMmcs>,
    input: &[Option<Vec<EF>>; 32],
    log_max_height: usize,
    challenger: &mut Challenger<SC>,
) -> CommitPhaseResult {
    let mut current = input[log_max_height].as_ref().unwrap().clone();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        let leaves = RowMajorMatrix::new(current.clone(), 2);
        let leaves_flattened = leaves.flatten_to_base();
        let tree = FieldMerkleTreeGpu::<
            BabyBear,
            [BabyBear; DIGEST_WIDTH],
            CudaSync<ColMajorMatrixDevice<BabyBear>>,
        >::new(vec![CudaSync::new(
            leaves_flattened.to_device().to_column_major(),
        )
        .unwrap()]);
        let commit: Hash<F, F, DIGEST_WIDTH> = tree.root().into();
        challenger.observe(commit);
        commits.push(commit);
        data.push(tree);

        let beta: EF = challenger.sample();
        current = fold_even_odd(current, beta);

        if let Some(v) = &input[log_folded_height] {
            current.iter_mut().zip_eq(v).for_each(|(c, v)| *c += *v);
        }
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    assert_eq!(current.len(), config.blowup());
    let final_poly = current[0];
    for x in current {
        assert_eq!(x, final_poly);
    }
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult {
        commits,
        data,
        final_poly,
    }
}

pub struct CommitPhaseResult {
    commits: Vec<Hash<F, F, DIGEST_WIDTH>>,
    data: Vec<FieldMerkleTreeGpu<F, [F; DIGEST_WIDTH], CudaSync<ColMajorMatrixDevice<BabyBear>>>>,
    final_poly: EF,
}

#[allow(clippy::type_complexity)]
pub fn answer_query(
    commit_phase_commits: &[FieldMerkleTreeGpu<
        F,
        [F; DIGEST_WIDTH],
        CudaSync<ColMajorMatrixDevice<BabyBear>>,
    >],
    index: usize,
) -> QueryProof<EF, ChallengeMmcs> {
    let commit_phase_openings = commit_phase_commits
        .iter()
        .enumerate()
        .map(|(i, commit)| {
            let index_i = index >> i;
            let index_i_sibling = index_i ^ 1;
            let index_pair = index_i >> 1;

            let (mut opened_rows, opening_proof) = open_batch(index_pair, commit);
            assert_eq!(opened_rows.len(), 1);
            let opened_row = opened_rows.pop().unwrap();
            let opened_row_ext = (0..opened_row.len() / 4)
                .map(|j| EF::from_base_slice(&opened_row[j * 4..(j + 1) * 4]))
                .collect::<Vec<_>>();
            assert_eq!(opened_row_ext.len(), 2, "Committed data should be in pairs");
            let sibling_value = opened_row_ext[index_i_sibling % 2];

            CommitPhaseProofStep {
                sibling_value,
                opening_proof,
            }
        })
        .collect();

    QueryProof {
        commit_phase_openings,
    }
}

impl<SC> Default for FriGpuOpeningProver<SC> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub mod opening_gpu {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use crate::matrix::MatrixViewDevice;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[link_name = "opening_gpu"]
    #[allow(unused_attributes)]
    extern "C" {

        #[link_name = "computeInverseDenominators"]
        pub fn compute_inverse_denominators(
            max_rows: usize,
            num_points: usize,
            inv_row_indices: *const usize,
            nums_rows: *const usize,
            logs_num_rows: *const usize,
            shifts: *const F,
            thread_generator_powers: *const F,
            points: *const EF,
            inv_denoms: *mut EF,
        );

        #[link_name = "interpolateCosets"]
        pub fn interpolate_cosets_raw(
            polys_evals: *const *const BabyBear,
            num_polys: usize,
            coset_heights: *const usize,
            coset_log_heights: *const usize,
            shift: *const F,
            point: *const EF,
            barycentric_scalars: *const EF,
            g_values: *const F,
            output: *mut EF,
        );

        #[link_name = "computeReducedOpenings"]
        pub fn compute_reduced_openings(
            mats: *const MatrixViewDevice<F>,
            max_height: usize,
            points: *const EF,
            num_points: usize,
            inv_indices: *const usize,
            inv_denoms: *const EF,
            alpha: EF,
            alpha_pow_offsets: *const EF,
            ys: *const EF,
            ys_indices: *const usize,
            reduced_openings: *mut EF,
        );

        #[link_name = "ReduceSums"]
        pub fn reduce_sums(
            heights: *const usize,
            max_height: usize,
            num_points: usize,
            inv_indices: *const usize,
            reduced_openings: *const EF,
            reduced_sums: *mut EF,
        );

        #[link_name = "numBlocksSums"]
        pub fn num_block_sums(max_height: usize) -> usize;

        #[link_name = "fetchRow"]
        pub fn fetch_row(matrix: MatrixViewDevice<F>, index: usize, output: *mut F);

        #[link_name = "batchMultiplicativeInverse"]
        pub fn batch_multiplicative_inverse(input: *const EF, output: *mut EF, num_elements: usize);
    }
}

#[cfg(test)]
mod tests {
    use crate::stark::tests::TENDERMINT_BENCHMARK_ELF;
    use crate::stark::StarkGpuProver;
    use p3_baby_bear::BabyBear;
    use p3_challenger::FieldChallenger;
    use p3_commit::Pcs;
    use p3_field::extension::BinomialExtensionField;

    use sp1_core::stark::StarkGenericConfig;
    use sp1_core::{
        runtime::Program,
        stark::{Challenge, RiscvAir},
        utils::BabyBearPoseidon2,
    };

    use crate::stark::tests::execute_core;
    type SC = BabyBearPoseidon2;
    use p3_commit::PolynomialSpace;

    use super::FriGpuOpeningProver;
    use crate::utils::init_tracer;

    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    #[ignore]
    fn test_opening_gpu() {
        let program = Program::from(TENDERMINT_BENCHMARK_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = StarkGpuProver::new(machine);

        // Execute the program.
        let record = execute_core(program);

        let open_prover = FriGpuOpeningProver::<SC>::default();

        init_tracer();

        let shards = gpu_prover.shard(record);

        for shard in shards {
            let gpu_main_data = gpu_prover.commit_main(&shard, 1);

            let mut challenger = gpu_prover.machine.config().challenger();
            let zeta: Challenge<SC> = challenger.sample_ext_element();
            let trace_opening_points = gpu_main_data
                .trace_data
                .domains
                .iter()
                .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                .collect::<Vec<_>>();

            let pcs = gpu_prover.machine.config().pcs();
            let start = std::time::Instant::now();
            let (mut openings_gpu, opening_proof) = open_prover.open(
                pcs,
                vec![(&gpu_main_data.prover_data, trace_opening_points.clone())],
                &mut challenger,
            );
            println!("device: time to open: {:?}", start.elapsed().as_secs_f64());

            let opening_gpu = openings_gpu.pop().unwrap();
            let mut challenger = gpu_prover.machine.config().challenger();
            let _zeta: Challenge<SC> = challenger.sample_ext_element();
            let domains_and_points = gpu_main_data
                .trace_data
                .domains
                .iter()
                .copied()
                .zip(trace_opening_points)
                .zip(opening_gpu)
                .map(|((domain, point), opening)| {
                    (domain, point.into_iter().zip(opening).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            let verifier_rounds = vec![(gpu_main_data.commit, domains_and_points)];
            <<SC as StarkGenericConfig>::Pcs as Pcs<EF, <SC as StarkGenericConfig>::Challenger>>::verify(
                pcs, verifier_rounds, &opening_proof, &mut challenger,
            ).unwrap();
        }
    }
}
