use std::marker::PhantomData;

use itertools::{izip, Itertools};
use p3_challenger::CanSample;
use p3_commit::Mmcs;
use p3_commit::{OpenedValues, Pcs};
use p3_field::AbstractField;
use p3_fri::{
    compute_inverse_denominators, BatchOpening, PowersReducer, TwoAdicFriPcs, TwoAdicFriPcsProof,
};
use p3_interpolation::interpolate_coset;
use p3_matrix::bitrev::BitReversalPerm;
use p3_matrix::Matrix;
use p3_util::VecExt;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use sp1_core::stark::Challenge;
use sp1_core::stark::{OpeningProof, PcsProverData};
use sp1_core::utils::log2_strict_usize;
use sp1_core::utils::InnerChallenge;
use sp1_core::utils::{InnerChallengeMmcs, InnerDft, InnerVal, InnerValMmcs};

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
        // TODO: replace this with FMTGPU
        rounds: Vec<(&PcsProverData<SC>, Vec<Vec<SC::Challenge>>)>,
        challenger: &mut SC::Challenger,
    ) -> (OpenedValues<SC::Challenge>, OpeningProof<SC>) {
        let alpha: Challenge<SC> = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| {
                (
                    pcs.mmcs
                        .get_matrices(data)
                        .into_iter()
                        .map(|m| m.as_view())
                        .collect_vec(),
                    points,
                )
            })
            .collect_vec();
        let mats = mats_and_points
            .iter()
            .flat_map(|(mats, _)| mats)
            .collect_vec();

        let global_max_width = mats.iter().map(|m| m.width()).max().unwrap();
        let global_max_height = mats.iter().map(|m| m.height()).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        let alpha_reducer = PowersReducer::<InnerVal, InnerChallenge>::new(alpha, global_max_width);

        // For each unique opening point z, we will find the largest degree bound
        // for that point, and precompute 1/(X - z) for the largest subgroup (in bitrev order).
        let inv_denoms = compute_inverse_denominators(&mats_and_points, InnerVal::generator());

        let mut all_opened_values: OpenedValues<InnerChallenge> = vec![];
        let mut reduced_openings: [_; 32] = core::array::from_fn(|_| None);
        let mut num_reduced = [0; 32];

        for (mats, points) in mats_and_points {
            let opened_values_for_round = all_opened_values.pushed_mut(vec![]);
            for (mat, points_for_mat) in izip!(mats, points) {
                let log_height = log2_strict_usize(mat.height());
                let reduced_opening_for_log_height = reduced_openings[log_height]
                    .get_or_insert_with(|| vec![InnerChallenge::zero(); mat.height()]);
                debug_assert_eq!(reduced_opening_for_log_height.len(), mat.height());

                let opened_values_for_mat = opened_values_for_round.pushed_mut(vec![]);
                for &point in points_for_mat {
                    // Use Barycentric interpolation to evaluate the matrix at the given point.
                    let ys = {
                        let (low_coset, _) = mat.split_rows(mat.height() >> pcs.fri.log_blowup);
                        interpolate_coset(
                            &BitReversalPerm::new_view(low_coset),
                            InnerVal::generator(),
                            point,
                        )
                    };

                    let alpha_pow_offset = alpha.exp_u64(num_reduced[log_height] as u64);
                    let sum_alpha_pows_times_y = alpha_reducer.reduce_ext(&ys);

                    reduced_opening_for_log_height
                        .par_iter_mut()
                        .zip_eq(mat.par_row_slices())
                        // This might be longer, but zip will truncate to smaller subgroup
                        // (which is ok because it's bitrev)
                        .zip(inv_denoms.get(&point).unwrap())
                        .for_each(|((reduced_opening, row), &inv_denom)| {
                            let row_sum = alpha_reducer.reduce_base(row);
                            *reduced_opening +=
                                inv_denom * alpha_pow_offset * (row_sum - sum_alpha_pows_times_y);
                        });

                    num_reduced[log_height] += mat.width();
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
                        let log_max_height = log2_strict_usize(pcs.mmcs.get_max_height(data));
                        let bits_reduced = log_global_max_height - log_max_height;
                        let reduced_index = index >> bits_reduced;
                        let (opened_values, opening_proof) =
                            pcs.mmcs.open_batch(reduced_index, data);
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
