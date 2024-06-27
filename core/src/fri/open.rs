use std::marker::PhantomData;
use std::sync::mpsc;

use p3_challenger::FieldChallenger;
use p3_field::two_adic_coset_zerofier;
use p3_field::Field;
use tracing::trace_span;

use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;
use p3_challenger::CanObserve;
use p3_challenger::CanSample;
use p3_challenger::CanSampleBits;
use p3_challenger::GrindingChallenger;
use p3_commit::{OpenedValues, Pcs};
use p3_field::cyclic_subgroup_coset_known_order;
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
use p3_util::linear_map::LinearMap;
use p3_util::log2_ceil_usize;
use p3_util::reverse_slice_index_bits;
use sp1_core::stark::Challenge;
use sp1_core::stark::Challenger;
use sp1_core::stark::{OpeningProof, PcsProverData};
use sp1_core::utils::baby_bear_poseidon2::ChallengeMmcs;
use sp1_core::utils::log2_strict_usize;
use sp1_core::utils::BabyBearPoseidon2;
use sp1_core::utils::InnerChallenge;
use sp1_core::utils::{InnerVal, InnerValMmcs};

use crate::device::buffer::DeviceBuffer;
use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::device::CudaSync;
use crate::matrix::ColMajorMatrixDevice;
use crate::matrix::MatrixViewDevice;
use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;
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

#[repr(C)]
pub struct RoundsData<F, EF> {
    mats: *const MatrixViewDevice<F>,
    points: *const EF,
    log_blowup: usize,
}

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

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = tracing::debug_span!("Compute inverse denominators").in_scope(|| {
            let heights_and_points = mats_and_points
                .iter()
                .map(|(mats, points)| (mats.iter().map(|m| m.height).collect_vec(), *points))
                .collect_vec();
            compute_inverse_denominators(&heights_and_points, BabyBear::generator())
        });

        let mut reduced_openings: [Option<Vec<SC::Challenge>>; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let compute_reduce_openings_span = trace_span!("Compute reduced openings").entered();
        let (tx, rx) = mpsc::channel();

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
        let mut point_indices = vec![];
        let mut total_polys = 0;
        let mut points_offset = 0;

        // // Values for inverse denominators.
        // let mut nums_rows = vec![];
        // let mut logs_num_rows = vec![];
        // let mut inv_denom_shifts = vec![];
        // let mut thread_generator_powers = vec![];

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

                    for point in points_for_mat {
                        point_indices.extend((0..cols).map(|_| points_offset));
                        points_offset += coset_height;
                        opening_points.extend((0..cols).map(|_| *point));
                        let zerofier =
                            two_adic_coset_zerofier(coset_log_height, EF::from_base(shift), *point);
                        let barycentric_scalar = zerofier * denominator.inverse();
                        barycentric_scalars.extend((0..cols).map(|_| barycentric_scalar));
                    }

                    for _ in 0..points_for_mat.len() {
                        let poly_evals_iter = unsafe {
                            (0..cols).map(|col| mat.values.as_ptr().add(col * mat.height))
                        };
                        poly_evals.extend(poly_evals_iter);
                    }
                })
        });

        assert_eq!(poly_evals.len(), total_polys);
        assert_eq!(coset_heights.len(), total_polys);
        assert_eq!(coset_log_heights.len(), total_polys);
        assert_eq!(shifts.len(), total_polys);
        assert_eq!(g_values.len(), total_polys * 32);
        assert_eq!(opening_points.len(), total_polys);
        assert_eq!(barycentric_scalars.len(), total_polys);
        assert_eq!(point_indices.len(), total_polys);

        let poly_evals = poly_evals.to_device();
        let coset_heights = coset_heights.to_device();
        let coset_log_heights = coset_log_heights.to_device();
        let shifts = shifts.to_device();
        let g_values = g_values.to_device();
        let opening_points = opening_points.to_device();
        let barycentric_scalars = barycentric_scalars.to_device();
        let point_indices = point_indices.to_device();

        let mut output_buffer = DeviceBuffer::<EF>::with_capacity(total_polys);

        let mut inv_denominators = DeviceBuffer::<EF>::with_capacity(points_offset);

        unsafe {
            output_buffer.set_max_len();
            inv_denominators.set_max_len();
            opening_gpu::interpolate_cosets_raw(
                poly_evals.as_ptr(),
                poly_evals.len(),
                coset_heights.as_ptr(),
                coset_log_heights.as_ptr(),
                shifts.as_ptr(),
                opening_points.as_ptr(),
                barycentric_scalars.as_ptr(),
                g_values.as_ptr(),
                output_buffer.as_mut_ptr(),
            );
        }

        interpolate_cosets_span.exit();

        let ys_out_host_span = tracing::debug_span!("ys_out_host").entered();
        let ys_output = output_buffer.to_host();
        ys_out_host_span.exit();

        assert_eq!(ys_output.len(), total_polys);
        let mut index = 0;

        let compute_openings_span = tracing::debug_span!("Compute openings").entered();
        let mut counter = 0;
        let all_opened_values = {
            //let mut counter_p = 0;
            let all_opened_values = mats_and_points
                .into_iter()
                .map(|(mats, points)| {
                    mats.into_iter()
                        .zip(points)
                        .map(|(mat, points_for_mat)| {
                            let log_height = log2_strict_usize(mat.height);

                            let opened_values_for_mat = points_for_mat
                                .iter()
                                .map(|&point| {
                                    let inv_denoms = &inv_denoms;
                                    let tx = tx.clone();
                                    let num_reduced_at_height = num_reduced[log_height];
                                    num_reduced[log_height] += mat.width();

                                    let point_offset = counter;
                                    counter += 1 << log_height;

                                    let ys = &output_buffer[index..index + mat.width()];
                                    let ys_host = ys_output[index..index + mat.width()].to_vec();
                                    index += mat.width();

                                    let alpha_pow_offset =
                                        alpha.exp_u64(num_reduced_at_height as u64);

                                    let inv_denoms_at_point =
                                        inv_denoms.get(&point).unwrap().to_device();

                                    let span = trace_span!("Compute reduced openings").entered();
                                    let mut reduced_opening_for_log_height_device: DeviceBuffer<
                                        SC::Challenge,
                                    > = DeviceBuffer::with_capacity(mat.height);

                                    unsafe {
                                        reduced_opening_for_log_height_device.set_max_len();
                                        opening_gpu::compute_reduced_openings_for_log_height(
                                            mat.view(),
                                            point,
                                            inv_denoms_at_point.as_ptr(),
                                            alpha,
                                            alpha_pow_offset,
                                            ys.as_ptr(),
                                            reduced_opening_for_log_height_device.as_mut_ptr(),
                                        );
                                    }
                                    let reduced_opening_for_log_height_host =
                                        reduced_opening_for_log_height_device.to_host();
                                    span.exit();

                                    tx.send((log_height, reduced_opening_for_log_height_host))
                                        .unwrap();

                                    ys_host
                                })
                                .collect::<Vec<_>>();
                            opened_values_for_mat
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            drop(tx);

            for (log_height, reduced_opening) in rx.iter() {
                let openings_at_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![SC::Challenge::zero(); 1 << log_height]);
                for (i, x) in reduced_opening.into_iter().enumerate() {
                    openings_at_height[i] += x;
                }
            }

            // Get all opened values by joining all the hadles and collecing the results.
            all_opened_values
                .into_iter()
                .map(|x| {
                    x.into_iter()
                        .map(|y| y.into_iter().collect::<Vec<_>>())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
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

#[allow(clippy::type_complexity)]
pub fn compute_inverse_denominators(
    heights_and_points: &[(Vec<usize>, &Vec<Vec<InnerChallenge>>)],
    coset_shift: BabyBear,
) -> LinearMap<InnerChallenge, Vec<InnerChallenge>> {
    let mut max_log_height_for_point: LinearMap<InnerChallenge, usize> = LinearMap::new();
    for (heights, points) in heights_and_points {
        for (height, points_for_mat) in izip!(heights, *points) {
            let log_height = log2_strict_usize(*height);
            for &z in points_for_mat {
                if let Some(lh) = max_log_height_for_point.get_mut(&z) {
                    *lh = core::cmp::max(*lh, log_height);
                } else {
                    max_log_height_for_point.insert(z, log_height);
                }
            }
        }
    }

    // Compute the largest subgroup we will use, in bitrev order.
    let max_log_height = *max_log_height_for_point.values().max().unwrap();
    let mut subgroup = cyclic_subgroup_coset_known_order(
        BabyBear::two_adic_generator(max_log_height),
        coset_shift,
        1 << max_log_height,
    )
    .collect_vec();
    reverse_slice_index_bits(&mut subgroup);

    max_log_height_for_point
        .into_iter()
        .map(|(z, log_height)| {
            let input = &subgroup[..(1 << log_height)]
                .iter()
                .map(|&x| InnerChallenge::from_base(x) - z)
                .collect_vec()
                .to_device();
            let mut output: DeviceBuffer<InnerChallenge> = DeviceBuffer::with_capacity(input.len());
            unsafe {
                output.set_len(input.len());
                opening_gpu::batch_multiplicative_inverse(
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    input.len(),
                );
            }
            (z, output.to_host())
        })
        .collect()
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
        let tree = FieldMerkleTreeGpu::new(vec![CudaSync::new(
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
    use crate::device::memory::ToHost;
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::two_adic_coset_zerofier;
    use p3_field::AbstractExtensionField;
    use p3_field::AbstractField;
    use p3_field::Field;
    use p3_field::TwoAdicField;
    use p3_util::log2_strict_usize;
    use sp1_core::utils::InnerChallenge;

    use crate::device::buffer::DeviceBuffer;
    use crate::device::memory::ToDevice;
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

        #[link_name = "computeReducedOpeningForLogHeight"]
        pub fn compute_reduced_openings_for_log_height(
            matrix: MatrixViewDevice<F>,
            point: EF,
            inv_denoms: *const EF,
            alpha: EF,
            alpha_pow_offset: EF,
            ys: *const EF,
            reduced_openings_for_log_height: *mut EF,
        );

        #[link_name = "fetchRow"]
        pub fn fetch_row(matrix: MatrixViewDevice<F>, index: usize, output: *mut F);

        #[link_name = "batchMultiplicativeInverse"]
        pub fn batch_multiplicative_inverse(input: *const EF, output: *mut EF, num_elements: usize);
    }
}

#[cfg(test)]
mod tests {
    use crate::device::memory::ToHost;
    use crate::stark::tests::TENDERMINT_BENCHMARK_ELF;
    use crate::stark::StarkGpuProver;
    use p3_baby_bear::BabyBear;
    use p3_challenger::FieldChallenger;
    use p3_commit::Pcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::{bitrev::BitReversalPerm, dense::RowMajorMatrix, Matrix};
    use rand::Rng;

    use sp1_core::stark::StarkGenericConfig;
    use sp1_core::{
        runtime::Program,
        stark::{Challenge, RiscvAir},
        utils::BabyBearPoseidon2,
    };

    use crate::stark::tests::execute_core;
    type SC = BabyBearPoseidon2;
    use crate::{device::memory::ToDevice, fri::opening_gpu};
    use p3_commit::PolynomialSpace;

    use super::FriGpuOpeningProver;
    use crate::utils::init_tracer;

    type EF = BinomialExtensionField<BabyBear, 4>;

    // #[test]
    // pub fn test_interpolate_coset_gpu() {
    //     let mut rng = rand::thread_rng();
    //     let rows = 256;
    //     let log_blowup = 1;
    //     let cols = 100;
    //     let matrix: RowMajorMatrix<BabyBear> =
    //         RowMajorMatrix::rand(&mut rng, rows << log_blowup, cols);

    //     let (low_coset, _) = matrix.split_rows(matrix.height() >> log_blowup);
    //     let shift: BabyBear = rng.gen();
    //     let point: BinomialExtensionField<BabyBear, 4> = rng.gen();
    //     let gt = interpolate_coset(&BitReversalPerm::new_view(low_coset), shift, point);

    //     let coset_evals = matrix.to_device().to_column_major();
    //     let coset_height = rows;
    //     let output = opening_gpu::interpolate_coset(coset_evals.view(), coset_height, shift, point);

    //     for i in 0..output.len() {
    //         assert_eq!(output[i], gt[i]);
    //     }
    //     println!("matched across {:?} elements", output.len());

    //     let coset_evals_host = coset_evals.to_host();
    //     println!("height: {:?}", coset_evals_host.height());
    //     println!("width: {:?}", coset_evals_host.width());
    // }

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
