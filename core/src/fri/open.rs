use std::marker::PhantomData;

use crate::device::memory::ToDevice;
use crate::device::memory::ToHost;
use crate::matrix::ColMajorMatrixDevice;
use crate::matrix::MatrixViewDevice;
use crate::poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH;
use itertools::{izip, Itertools};
use p3_baby_bear::BabyBear;
use p3_challenger::CanSample;
use p3_commit::Mmcs;
use p3_commit::{OpenedValues, Pcs};
use p3_field::batch_multiplicative_inverse;
use p3_field::cyclic_subgroup_coset_known_order;
use p3_field::extension::BinomialExtensionField;
use p3_field::AbstractExtensionField;
use p3_field::AbstractField;
use p3_field::TwoAdicField;
use p3_fri::{BatchOpening, PowersReducer, TwoAdicFriPcs, TwoAdicFriPcsProof};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::BitReversalPerm;
use p3_matrix::Matrix;
use p3_util::linear_map::LinearMap;
use p3_util::log2_ceil_usize;
use p3_util::reverse_slice_index_bits;
use p3_util::VecExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use sp1_core::stark::Challenge;
use sp1_core::stark::{OpeningProof, PcsProverData};
use sp1_core::utils::log2_strict_usize;
use sp1_core::utils::InnerChallenge;
use sp1_core::utils::{InnerChallengeMmcs, InnerDft, InnerVal, InnerValMmcs};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

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
        &self,
        pcs: &TwoAdicFriPcs<InnerVal, InnerDft, InnerValMmcs, InnerChallengeMmcs>,
        rounds: Vec<(
            &FieldMerkleTreeGpu<BabyBear, [BabyBear; DIGEST_WIDTH], ColMajorMatrixDevice<BabyBear>>,
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

        let alpha_reducer = PowersReducer::<InnerVal, InnerChallenge>::new(alpha, global_max_width);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, BabyBear::generator());

        let mut all_opened_values: OpenedValues<InnerChallenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

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
                    let coset_height = mat.height >> pcs.fri.log_blowup;
                    let ys = opening_gpu::interpolate_coset(
                        mat,
                        coset_height,
                        InnerVal::generator(),
                        point,
                    );

                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);

                    let inv_denoms_at_point = inv_denoms.get(&point).unwrap().to_device();
                    let alpha_powers = alpha_reducer.powers.to_device();

                    let mut reduced_opening_for_log_height_device =
                        reduced_opening_for_log_height.to_device();
                    unsafe {
                        opening_gpu::compute_reduced_openings_for_log_height(
                            mat,
                            inv_denoms_at_point.as_ptr(),
                            alpha_powers.as_ptr(),
                            alpha_pow_offset,
                            sum_alpha_pows_times_y,
                            reduced_opening_for_log_height.as_mut_ptr(),
                        );
                    }
                    let reduced_opening_for_log_height_host =
                        reduced_opening_for_log_height_device.to_host();

                    for i in 0..reduced_opening_for_log_height_host.len() {
                        reduced_opening_for_log_height[i] += reduced_opening_for_log_height_host[i];
                    }

                    num_reduced[log_height] += mat.width;
                    opened_values_for_mat.push(ys);
                }
            }
        }

        let (fri_proof, query_indices) =
            p3_fri::prover::prove(&pcs.fri, &reduced_openings, challenger);

        let query_openings = query_indices
            .into_iter()
            .map(|index| {
                rounds
                    .iter()
                    .map(|(data, _)| {
                        let max_height = data.leaves.iter().map(|m| m.width()).max().unwrap();
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
        ColMajorMatrixDevice<BabyBear>,
    >,
) -> (Vec<Vec<F>>, Vec<[F; DIGEST_WIDTH]>) {
    let max_height = prover_data.leaves.iter().map(|m| m.width()).max().unwrap();
    let log_max_height = log2_ceil_usize(max_height);

    let openings = prover_data
        .leaves
        .iter()
        .map(|matrix| {
            let log2_height = log2_ceil_usize(matrix.height());
            let bits_reduced = log_max_height - log2_height;
            let reduced_index = index >> bits_reduced;
            let mut output_device = vec![F::zero(); matrix.height()].to_device();
            unsafe {
                output_device.set_len(matrix.height());
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
            (
                z,
                batch_multiplicative_inverse(
                    &subgroup[..(1 << log_height)]
                        .iter()
                        .map(|&x| InnerChallenge::from_base(x) - z)
                        .collect_vec(),
                ),
            )
        })
        .collect()
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
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::AbstractField;
    use p3_fri::PowersReducer;
    use p3_interpolation::interpolate_coset;
    use p3_matrix::{bitrev::BitReversalPerm, dense::RowMajorMatrix, Matrix};
    use p3_util::log2_strict_usize;
    use rand::Rng;
    use rayon::iter::IntoParallelRefMutIterator;
    use rayon::iter::ParallelIterator;
    use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};

    use crate::device::buffer::DeviceBuffer;
    use crate::{device::memory::ToDevice, fri::opening_gpu};

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
}
