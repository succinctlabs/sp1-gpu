use std::marker::PhantomData;

use crate::device::buffer::DeviceBuffer;
use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::device::CudaSync;
use crate::matrix::ColMajorMatrixDevice;
use crate::matrix::MatrixViewDevice;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;
use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;
use p3_challenger::CanObserve;
use p3_challenger::CanSample;
use p3_challenger::CanSampleBits;
use p3_challenger::GrindingChallenger;
use p3_commit::Mmcs;
use p3_commit::{OpenedValues, Pcs};
use p3_field::batch_multiplicative_inverse;
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
use p3_fri::{BatchOpening, PowersReducer, TwoAdicFriPcs, TwoAdicFriPcsProof};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::BitReversalPerm;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_symmetric::Hash;
use p3_util::linear_map::LinearMap;
use p3_util::log2_ceil_usize;
use p3_util::reverse_slice_index_bits;
use p3_util::VecExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use sp1_core::stark::Challenge;
use sp1_core::stark::Challenger;
use sp1_core::stark::{OpeningProof, PcsProverData};
use sp1_core::utils::baby_bear_poseidon2::ChallengeMmcs;
use sp1_core::utils::baby_bear_poseidon2::ValMmcs;
use sp1_core::utils::log2_strict_usize;
use sp1_core::utils::BabyBearPoseidon2;
use sp1_core::utils::InnerChallenge;
use sp1_core::utils::{InnerChallengeMmcs, InnerDft, InnerVal, InnerValMmcs};
use std::time::Instant;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;
type SC = BabyBearPoseidon2;

use crate::merkle_tree::FieldMerkleTreeGpu;
use crate::stark::BabyBearPoseidon2Config;

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
        pcs: &TwoAdicFriPcs<InnerVal, InnerDft, InnerValMmcs, InnerChallengeMmcs>,
        rounds: Vec<(
            &FieldMerkleTreeGpu<
                BabyBear,
                [BabyBear; DIGEST_WIDTH],
                CudaSync<ColMajorMatrixDevice<BabyBear>>,
            >,
            Vec<Vec<SC::Challenge>>,
        )>,
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

        let start = Instant::now();
        let alpha_reducer = PowersReducer::<InnerVal, InnerChallenge>::new(alpha, global_max_width);
        println!("device: time to reduce powers: {:?}", start.elapsed());

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let start = Instant::now();
        let inv_denoms = compute_inverse_denominators(&mats_and_points, BabyBear::generator());
        println!(
            "device: time to compute inverse denominators: {:?}",
            start.elapsed()
        );

        let mut all_opened_values: OpenedValues<InnerChallenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        let start = std::time::Instant::now();
        for (mats, points) in mats_and_points {
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, points) {
                let log_height = log2_strict_usize(mat.height);
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![InnerChallenge::zero(); mat.height]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height);

                let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);

                for &point in points_for_mat {
                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let mut col_major_mat = mat;
                    col_major_mat.height = mat.width;
                    col_major_mat.width = mat.height;
                    let coset_height = mat.height >> pcs.fri.log_blowup;

                    let start = Instant::now();
                    let ys = opening_gpu::interpolate_coset(
                        col_major_mat,
                        coset_height,
                        InnerVal::generator(),
                        point,
                    );
                    println!("device: time to interpolate coset: {:?}", start.elapsed());

                    let start = Instant::now();
                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);
                    println!("device: time to reduce powers: {:?}", start.elapsed());

                    let inv_denoms_at_point = inv_denoms.get(&point).unwrap().to_device();
                    let alpha_powers = alpha_reducer.powers.to_device();

                    let start = Instant::now();
                    let mut reduced_opening_for_log_height_device: DeviceBuffer<InnerChallenge> =
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
                    println!(
                        "device: time to compute reduced openings: {:?}",
                        start.elapsed()
                    );

                    let start = Instant::now();
                    for i in 0..reduced_opening_for_log_height_host.len() {
                        reduced_opening_for_log_height[i] += reduced_opening_for_log_height_host[i];
                    }
                    println!(
                        "device: time to add reduced openings: {:?}",
                        start.elapsed()
                    );

                    num_reduced[log_height] += mat.width;
                    opened_values_for_mat.push(ys);
                }
            }
        }
        println!(
            "device: time to compute reduced openings: {:?}",
            start.elapsed()
        );

        let start = std::time::Instant::now();
        let (fri_proof, query_indices) = prove(&pcs.fri, &reduced_openings, challenger);
        println!("device: time to fri proof: {:?}", start.elapsed());

        let start = std::time::Instant::now();
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
        println!(
            "device: time to compute query openings: {:?}",
            start.elapsed()
        );

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

    let start = Instant::now();
    let commit_phase_result = commit_phase(config, input, log_max_height, challenger);

    let start = Instant::now();
    let pow_witness = challenger.grind(config.proof_of_work_bits);

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let start = Instant::now();
    let query_proofs = query_indices
        .iter()
        .map(|&index| answer_query(config, &commit_phase_result.data, index))
        .collect::<Vec<_>>();

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
        let tree = FieldMerkleTreeGpu::new(vec![
            CudaSync::new(leaves_flattened.to_device().to_column_major()).unwrap(),
        ]);
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

pub fn answer_query(
    config: &FriConfig<ChallengeMmcs>,
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
    use rayon::iter::IndexedParallelIterator;
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
    use crate::fri::open::Pcs;
    use crate::stark::tests::TENDERMINT_BENCHMARK_ELF;
    use crate::stark::CpuMainData;
    use crate::stark::CpuQuotientValuesGenerator;
    use crate::stark::FriGpuProver;
    use crate::stark::QuotientValues;
    use p3_baby_bear::BabyBear;
    use p3_challenger::FieldChallenger;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractExtensionField;
    use p3_field::AbstractField;
    use p3_fri::PowersReducer;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::{bitrev::BitReversalPerm, dense::RowMajorMatrix, Matrix};
    use p3_util::log2_strict_usize;
    use rand::Rng;
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
    use sp1_core::air::MachineAir;
    use sp1_core::stark::Verifier;

    use crate::fri::FriCpuOpeningProver;
    use crate::stark::FriCpuProver;
    use p3_challenger::CanObserve;
    use rand::thread_rng;
    use sp1_core::stark::StarkGenericConfig;
    use sp1_core::{
        runtime::{ExecutionRecord, Program, Runtime},
        stark::{Challenge, RiscvAir},
        utils::{tests::FIBONACCI_ELF, BabyBearPoseidon2, SP1CoreOpts},
    };

    use crate::stark::tests::execute_core;
    type SC = BabyBearPoseidon2;
    use crate::device::buffer::DeviceBuffer;
    use crate::{device::memory::ToDevice, fri::opening_gpu};
    use p3_commit::PolynomialSpace;
    use sp1_core::stark::Challenger;

    use super::FriGpuOpeningProver;

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

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let cpu_prover = FriCpuProver::new(machine);

        let config = SC::default();

        let (pk, vk) = gpu_prover.machine.setup(&program);

        // Execute the program.
        let record = execute_core(program);

        let shards = gpu_prover.shard(record);

        for shard in shards {
            // let cpu_main_data = cpu_prover.commit_main(&shard, 1);
            // let main_commit = cpu_main_data.commit;

            // let mut challenger = cpu_prover.machine.config().challenger();
            // let zeta: Challenge<SC> = challenger.sample_ext_element();
            // let trace_opening_points = cpu_main_data
            //     .trace_data
            //     .domains
            //     .iter()
            //     .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
            //     .collect::<Vec<_>>();

            // let start = std::time::Instant::now();
            // let (openings, opening_proof) = <<sp1_core::utils::BabyBearPoseidon2 as sp1_core::stark::StarkGenericConfig>::Pcs as Pcs<Challenge<SC>, Challenger<SC>>>::open(
            //     cpu_prover.machine.config().pcs(),
            //     vec![
            //         (&cpu_main_data.prover_data, trace_opening_points.clone()),
            //     ],
            //     &mut challenger,
            // );
            // println!("host: time to open: {:?}", start.elapsed().as_secs_f64());

            let gpu_main_data = gpu_prover.commit_main(&shard, 1);
            let main_commit = gpu_main_data.commit;

            let mut challenger = gpu_prover.machine.config().challenger();
            let zeta: Challenge<SC> = challenger.sample_ext_element();
            let trace_opening_points = gpu_main_data
                .trace_data
                .domains
                .iter()
                .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                .collect::<Vec<_>>();

            let start = std::time::Instant::now();
            let (openings_gpu, opening_proof) = FriGpuOpeningProver::<SC>::open(
                gpu_prover.machine.config().pcs(),
                vec![(&gpu_main_data.prover_data, trace_opening_points.clone())],
                &mut challenger,
            );

            println!("device: time to open: {:?}", start.elapsed().as_secs_f64());
        }
    }
}
