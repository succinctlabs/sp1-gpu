use std::marker::PhantomData;

use tracing::debug_span;

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
use p3_fri::{BatchOpening, PowersReducer, TwoAdicFriPcsProof};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Hash;
use p3_util::linear_map::LinearMap;
use p3_util::log2_ceil_usize;
use p3_util::reverse_slice_index_bits;
use p3_util::VecExt;
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

impl<SC: BabyBearPoseidon2Config> FriGpuOpeningProver<SC> {
    #[allow(clippy::type_complexity)]
    pub fn open(
        pcs: &SC::Pcs,
        rounds: Vec<(&GpuProverData<SC>, Vec<Vec<SC::Challenge>>)>,
        challenger: &mut SC::Challenger,
    ) -> (OpenedValues<SC::Challenge>, OpeningProof<SC>) {
        let alpha: Challenge<SC> = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| (data.leaves.iter().map(|l| l.view()).collect_vec(), points))
            .collect_vec();
        let mats = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .collect_vec();

        let global_max_width = mats.iter().map(|m| m.width).max().unwrap();
        let global_max_height = mats.iter().map(|m| m.height).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        let alpha_reducer = debug_span!("Reduce powers")
            .in_scope(|| PowersReducer::<InnerVal, InnerChallenge>::new(alpha, global_max_width));

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = debug_span!("Compute inverse denominators")
            .in_scope(|| compute_inverse_denominators(&mats_and_points, BabyBear::generator()));

        let mut all_opened_values: OpenedValues<InnerChallenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let compute_reduce_openings_span = debug_span!("Compute reduced openings").entered();
        for (mats, points) in mats_and_points {
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, points) {
                let log_height = log2_strict_usize(mat.height);
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![SC::Challenge::zero(); mat.height]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height);

                let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);

                for &point in points_for_mat {
                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let mut col_major_mat = mat;
                    col_major_mat.height = mat.width;
                    col_major_mat.width = mat.height;
                    let coset_height = mat.height >> pcs.fri_config().log_blowup;

                    let span = debug_span!("Interpolate coset").entered();
                    let ys = opening_gpu::interpolate_coset(
                        col_major_mat,
                        coset_height,
                        InnerVal::generator(),
                        point,
                    );
                    span.exit();

                    let span = debug_span!("Reduce powers").entered();
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);
                    span.exit();

                    let inv_denoms_at_point = inv_denoms.get(&point).unwrap().to_device();
                    let alpha_powers = alpha_reducer.powers.to_device();

                    let span = debug_span!("Compute reduced openings").entered();
                    let mut reduced_opening_for_log_height_device: DeviceBuffer<SC::Challenge> =
                        DeviceBuffer::with_capacity(reduced_opening_for_log_height.len());
                    unsafe {
                        reduced_opening_for_log_height_device
                            .set_len(reduced_opening_for_log_height.len());
                        opening_gpu::compute_reduced_openings_for_log_height(
                            col_major_mat,
                            inv_denoms_at_point.as_ptr(),
                            alpha_powers.as_ptr(),
                            alpha_pow_offset,
                            sum_alpha_pows_times_y,
                            reduced_opening_for_log_height_device.as_mut_ptr(),
                        );
                    }
                    let reduced_opening_for_log_height_host =
                        reduced_opening_for_log_height_device.to_host();
                    span.exit();

                    let span = debug_span!("Add reduced openings").entered();
                    for i in 0..reduced_opening_for_log_height_host.len() {
                        reduced_opening_for_log_height[i] += reduced_opening_for_log_height_host[i];
                    }
                    span.exit();

                    num_reduced[log_height] += mat.width;
                    opened_values_for_mat.push(ys);
                }
            }
        }
        compute_reduce_openings_span.exit();

        let (fri_proof, query_indices) = debug_span!("Fri Proof")
            .in_scope(|| prove(pcs.fri_config(), &reduced_openings, challenger));

        let query_openings_span = debug_span!("Compute query openings").entered();
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
                        BatchOpening::<InnerVal, InnerValMmcs> {
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
    mats_and_points: &[(Vec<MatrixViewDevice<BabyBear>>, &Vec<Vec<InnerChallenge>>)],
    coset_shift: BabyBear,
) -> LinearMap<InnerChallenge, Vec<InnerChallenge>> {
    let mut max_log_height_for_point: LinearMap<InnerChallenge, usize> = LinearMap::new();
    for (mats, points) in mats_and_points {
        for (mat, points_for_mat) in izip!(mats, *points) {
            let log_height = log2_strict_usize(mat.height);
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

    let commit_phase_result = debug_span!("Commit phase")
        .in_scope(|| commit_phase(config, input, log_max_height, challenger));

    let pow_witness =
        debug_span!("POW witness").in_scope(|| challenger.grind(config.proof_of_work_bits));

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs_span = debug_span!("Compute query proofs").entered();
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
        #[link_name = "interpolateCoset"]
        pub fn interpolate_coset_raw(
            coset_evals: MatrixViewDevice<BabyBear>,
            coset_height: usize,
            coset_log_height: usize,
            shift: F,
            point: EF,
            barycentric_scalar: EF,
            g_powers: *const F,
            output: *mut EF,
        );

        #[link_name = "computeReducedOpeningForLogHeight"]
        pub fn compute_reduced_openings_for_log_height(
            matrix: MatrixViewDevice<F>,
            inv_denoms: *const EF,
            alpha_powers: *const EF,
            alpha_pow_offset: EF,
            sum_alpha_pow_times_y: EF,
            reduced_openings_for_log_height: *mut EF,
        );

        #[link_name = "fetchRow"]
        pub fn fetch_row(matrix: MatrixViewDevice<F>, index: usize, output: *mut F);

        #[link_name = "batchMultiplicativeInverse"]
        pub fn batch_multiplicative_inverse(input: *const EF, output: *mut EF, num_elements: usize);
    }

    pub fn interpolate_coset(
        coset_evals: MatrixViewDevice<BabyBear>,
        coset_height: usize,
        shift: BabyBear,
        point: InnerChallenge,
    ) -> Vec<InnerChallenge> {
        let cols = coset_evals.height;
        let coset_log_height = log2_strict_usize(coset_height);
        let g = BabyBear::two_adic_generator(coset_log_height);
        let g_powers = g.powers().take(coset_height).collect::<Vec<_>>();
        let g_powers_device = g_powers.to_device();

        let zerofier = two_adic_coset_zerofier(coset_log_height, EF::from_base(shift), point);
        let denominator =
            F::from_canonical_usize(coset_height) * shift.exp_u64(coset_height as u64 - 1);
        let barycentric_scalar = zerofier * denominator.inverse();

        let mut output_device: DeviceBuffer<InnerChallenge> = DeviceBuffer::with_capacity(cols);
        unsafe {
            output_device.set_len(cols);
            interpolate_coset_raw(
                coset_evals,
                coset_height,
                coset_log_height,
                shift,
                point,
                barycentric_scalar,
                g_powers_device.as_ptr(),
                output_device.as_mut_ptr(),
            );
        };

        output_device.to_host()
    }
}

#[cfg(test)]
mod tests {
    use crate::device::memory::ToHost;
    use crate::stark::tests::TENDERMINT_BENCHMARK_ELF;
    use crate::stark::FriGpuProver;
    use p3_baby_bear::BabyBear;
    use p3_challenger::FieldChallenger;
    use p3_commit::Pcs;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use p3_fri::PowersReducer;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::{bitrev::BitReversalPerm, dense::RowMajorMatrix, Matrix};
    use rand::Rng;
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};

    use sp1_core::stark::StarkGenericConfig;
    use sp1_core::{
        runtime::Program,
        stark::{Challenge, RiscvAir},
        utils::BabyBearPoseidon2,
    };

    use crate::stark::tests::execute_core;
    type SC = BabyBearPoseidon2;
    use crate::device::buffer::DeviceBuffer;
    use crate::{device::memory::ToDevice, fri::opening_gpu};
    use p3_commit::PolynomialSpace;

    use super::FriGpuOpeningProver;
    use crate::utils::init_tracer;

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    pub fn test_interpolate_coset_gpu() {
        let mut rng = rand::thread_rng();
        let rows = 256;
        let log_blowup = 1;
        let cols = 100;
        let matrix: RowMajorMatrix<BabyBear> =
            RowMajorMatrix::rand(&mut rng, rows << log_blowup, cols);

        let (low_coset, _) = matrix.split_rows(matrix.height() >> log_blowup);
        let shift: BabyBear = rng.gen();
        let point: BinomialExtensionField<BabyBear, 4> = rng.gen();
        let gt = interpolate_coset(&BitReversalPerm::new_view(low_coset), shift, point);

        let coset_evals = matrix.transpose().to_device();
        let coset_height = rows;
        let output = opening_gpu::interpolate_coset(coset_evals.view(), coset_height, shift, point);

        for i in 0..output.len() {
            assert_eq!(output[i], gt[i]);
        }
        println!("matched across {:?} elements", output.len());

        let coset_evals_host = coset_evals.to_host();
        println!("height: {:?}", coset_evals_host.height());
        println!("width: {:?}", coset_evals_host.width());
    }

    #[test]
    pub fn test_compute_reduced_openings_gpu() {
        let mut rng = rand::thread_rng();
        let rows = 4;
        let log_blowup = 1;
        let cols = 4;
        let matrix: RowMajorMatrix<BabyBear> =
            RowMajorMatrix::rand(&mut rng, rows << log_blowup, cols);
        let height = matrix.height();

        let alpha: EF = rng.gen();
        let alpha_reducer = PowersReducer::<F, EF>::new(alpha, matrix.width());
        let mut reduced_opening_for_log_height = vec![EF::zero(); matrix.height()];
        let inv_denoms: Vec<EF> = vec![rng.gen(); matrix.height()];

        let pow: u64 = rng.gen();
        let alpha_pow_offset = alpha.exp_u64(pow);
        let ys: Vec<EF> = vec![rng.gen(); matrix.width()];
        let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);
        reduced_opening_for_log_height
            .par_iter_mut()
            .zip_eq(matrix.par_row_slices())
            .zip(inv_denoms.par_iter())
            .for_each(|((reduced_opening, row), &inv_denom)| {
                let row_sum = alpha_reducer.reduce_base(row);
                *reduced_opening +=
                    inv_denom * alpha_pow_offset * (row_sum - sum_alpha_pows_times_y);
            });

        let matrix_device = matrix.transpose().to_device();
        let inv_denoms_device = inv_denoms.to_device();
        let alpha_powers = alpha_reducer.powers.to_device();
        let mut reduced_openings_for_log_height_device: DeviceBuffer<EF> =
            DeviceBuffer::with_capacity(height);

        unsafe {
            reduced_openings_for_log_height_device.set_len(height);
            opening_gpu::compute_reduced_openings_for_log_height(
                matrix_device.view(),
                inv_denoms_device.as_ptr(),
                alpha_powers.as_ptr(),
                alpha_pow_offset,
                sum_alpha_pows_times_y,
                reduced_openings_for_log_height_device.as_mut_ptr(),
            );
        }

        let output = reduced_openings_for_log_height_device.to_host();

        for i in 0..height {
            assert_eq!(
                output[i], reduced_opening_for_log_height[i],
                "failed at index {}",
                i
            );
        }
        println!("matched across {:?} elements", height);
    }

    #[test]
    fn test_opening_gpu() {
        let program = Program::from(TENDERMINT_BENCHMARK_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = FriGpuProver::new(machine);

        // Execute the program.
        let record = execute_core(program);

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
            let (mut openings_gpu, opening_proof) = FriGpuOpeningProver::<SC>::open(
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
