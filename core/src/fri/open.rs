use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
};

use p3_challenger::FieldChallenger;
use p3_commit::Mmcs;
use p3_field::{two_adic_coset_zerofier, Field};
use sp1_core_machine::utils::log2_strict_usize;
use sp1_stark::{Challenge, Com, OpeningProof};
use tracing::trace_span;

use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, CanSampleBits, GrindingChallenger};
use p3_commit::OpenedValues;
use p3_field::{AbstractExtensionField, AbstractField, TwoAdicField};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof, TwoAdicFriPcsProof};
use sp1_stark::{Challenger, InnerVal};

use crate::{
    device::{
        memory::{ToDevice, ToHost},
        DeviceBuffer,
    },
    fri::TwoAdicFriCommitter,
    matrix::{ColMajorMatrixDevice, MatrixViewDevice},
    merkle_tree::{MmcsCommitter, MmcsProverData},
    stark::{BabyBearFriConfig, FriMmcs, PcsConfig},
};

#[derive(Clone, Copy, Debug)]
pub struct FriOpeningProver<SC>(PhantomData<SC>);

pub trait FriQueryProver<F: Field, ValMmcs: Mmcs<F>>: MmcsCommitter<F, ValMmcs> {
    fn query_open_batch(
        &self,
        query_indices: &[usize],
        prover_data_slice: &[&Self::ProverData],
        log_global_max_height: usize,
        is_answering: bool,
    ) -> Vec<Vec<BatchOpening<F, ValMmcs>>>;
}

impl<SC: BabyBearFriConfig> FriOpeningProver<SC> {
    pub fn batch_update(
        &self,
        leaf_matrix: &mut ColMajorMatrixDevice<SC::Val>,
        polynomial_batch: &ColMajorMatrixDevice<SC::Val>,
        evaluations: &DeviceBuffer<SC::Challenge>,
        evaluation_point: SC::Challenge,
        batching_challenge: SC::Challenge,
        batching_challenge_offset: &mut SC::Challenge,
    ) {
        let log_height = polynomial_batch.height().ilog2() as usize;
        let domain_generator = BabyBear::two_adic_generator(log_height);
        let width = polynomial_batch.width();
        let shift = BabyBear::generator();
        unsafe {
            opening_gpu::batch_fri_update(
                leaf_matrix.view_mut(),
                polynomial_batch.values.as_ptr(),
                evaluations.as_ptr(),
                domain_generator,
                shift,
                evaluation_point,
                batching_challenge,
                *batching_challenge_offset,
                width,
                log_height,
                crate::cuda_runtime::ffi::DEFAULT_STREAM,
            );
        }
        *batching_challenge_offset *= batching_challenge.exp_u64(width as u64);
    }

    #[allow(clippy::type_complexity)]
    pub fn open<C>(
        &self,
        committer: &TwoAdicFriCommitter<SC, C>,
        pcs: &SC::Pcs,
        rounds: Vec<(&C::ProverData, Vec<Vec<SC::Challenge>>)>,
        challenger: &mut SC::Challenger,
    ) -> (OpenedValues<SC::Challenge>, OpeningProof<SC>)
    where
        C: FriQueryProver<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
    {
        let alpha: Challenge<SC> = challenger.sample();

        let mats_and_points = rounds
            .iter()
            .map(|(data, points)| (data.matrices().iter().collect_vec(), points))
            .collect_vec();
        let mats = mats_and_points.iter().flat_map(|(mats, _)| mats).collect_vec();

        let global_max_height = mats.iter().map(|m| m.height).max().unwrap();
        let log_global_max_height = log2_strict_usize(global_max_height);

        let mut num_reduced = [0; 32];

        let compute_reduce_openings_span = trace_span!("Compute reduced openings").entered();

        // Interpolate cosets.

        let interpolate_cosets_span = tracing::trace_span!("Interpolate cosets").entered();

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
        let mut height_indices = [0usize; 32];
        let mut height_index_map = HashMap::new();
        let mut reduced_leaves = vec![];

        let mut reduced_leaf = ColMajorMatrixDevice::<SC::Val>::with_capacity(
            2 * <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
            1,
        )
        .unwrap();
        unsafe {
            reduced_leaf.set_max_width();
        }
        reduced_leaves.push(reduced_leaf);

        let get_data_for_device_span = tracing::trace_span!("get data for device").entered();
        mats_and_points.iter().for_each(|(mats, points)| {
            mats.iter().zip(points.iter()).for_each(|(mat, points_for_mat)| {
                let log_height = log2_strict_usize(mat.height);
                height_index_map.entry(log_height).or_insert_with_key(|log_height| {
                    let idx = reduced_leaves.len();
                    height_indices[*log_height] = idx;
                    let mut reduced_leaf = ColMajorMatrixDevice::<SC::Val>::with_capacity(
                        2 * <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
                        1 << (log_height - 1),
                    )
                    .unwrap();
                    unsafe {
                        reduced_leaf.set_max_width();
                    }
                    reduced_leaves.push(reduced_leaf);

                    idx
                });
                // Use Barycentric interpolation to evaluate the matrix at the given point.
                let coset_height = mat.height() >> pcs.fri_config().log_blowup;
                let cols = mat.width();
                let num_polys = cols * points_for_mat.len();
                total_polys += num_polys;
                let coset_log_height = log2_strict_usize(coset_height);
                let g = BabyBear::two_adic_generator(coset_log_height);
                let denominator = SC::Val::from_canonical_usize(coset_height)
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
                    let zerofier = two_adic_coset_zerofier(
                        coset_log_height,
                        SC::Challenge::from_base(shift),
                        *point,
                    );
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
                    let poly_evals_iter =
                        unsafe { (0..cols).map(|col| mat.values.as_ptr().add(col * mat.height)) };
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
        let mut inv_denominators =
            DeviceBuffer::<SC::Challenge>::with_capacity(inv_offset).unwrap();
        let inv_indices_device = inv_indices.to_device().unwrap();
        unsafe {
            inv_denominators.set_max_len();
            let nums_rows = nums_rows.to_device().unwrap();
            let log_nums_rows = log_nums_rows.to_device().unwrap();
            let thread_generator_powers = thread_generator_powers.to_device().unwrap();
            let points_for_inv = points_for_inv.to_device().unwrap();
            let shifts_for_inv = shifts_for_inv.to_device().unwrap();
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
            let mut ys_output_buffer =
                DeviceBuffer::<SC::Challenge>::with_capacity(total_polys).unwrap();

            let poly_evals = poly_evals.to_device().unwrap();
            let coset_heights = coset_heights.to_device().unwrap();
            let coset_log_heights = coset_log_heights.to_device().unwrap();
            let shifts = shifts.to_device().unwrap();
            let g_values = g_values.to_device().unwrap();
            let opening_points = opening_points.to_device().unwrap();
            let barycentric_scalars = barycentric_scalars.to_device().unwrap();

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

        let ys_out_host_span = tracing::trace_span!("ys_out_host").entered();
        let ys_output = ys_output_buffer.to_host();
        ys_out_host_span.exit();

        assert_eq!(ys_output.len(), total_polys);

        let compute_openings_span = tracing::trace_span!("compute opened values").entered();
        let mut point_index = 0;
        let all_opened_values = {
            let mut reduced_openings_device =
                DeviceBuffer::<SC::Challenge>::with_capacity(inv_offset).unwrap();

            // Compute openings fused.
            let compute_reduced_openings_span =
                tracing::trace_span!("compute reduced openings on device").entered();
            let alpha_pow_offsets_device = alpha_pow_offsets.to_device().unwrap();
            let log_heights = matrices_for_openings
                .iter()
                .map(|mat| log2_strict_usize(mat.height))
                .collect::<Vec<_>>()
                .to_device()
                .unwrap();
            let matrices_for_openings = matrices_for_openings.to_device().unwrap();
            let ys_indices = ys_indices.to_device().unwrap();

            let mut reduce_leaves_raw = reduced_leaves
                .iter_mut()
                .map(|mat| mat.view_mut())
                .collect::<Vec<_>>()
                .to_device()
                .unwrap();

            unsafe {
                reduced_openings_device.set_max_len();
                opening_gpu::compute_reduced_openings(
                    matrices_for_openings.as_ptr(),
                    log_heights.as_ptr(),
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

            let height_indices = height_indices.to_device().unwrap();
            unsafe {
                opening_gpu::reduce_sums(
                    log_heights.as_ptr(),
                    global_max_height,
                    inv_indices_device.as_ptr(),
                    reduced_openings_device.as_ptr(),
                    reduce_leaves_raw.as_mut_ptr(),
                    height_indices.as_ptr(),
                    num_points,
                );
            }

            let copy_values_span = tracing::trace_span!("Copy opened values to host").entered();
            let mut index = 0;
            let all_opened_values = mats_and_points
                .into_iter()
                .map(|(mats, points)| {
                    mats.into_iter()
                        .zip(points)
                        .map(|(mat, points_for_mat)| {
                            let opened_values_for_mat = points_for_mat
                                .iter()
                                .map(|_| {
                                    let ys_host = ys_output[index..index + mat.width()].to_vec();
                                    index += mat.width();
                                    point_index += 1;
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

        let inverse_index_map: BTreeMap<_, _> =
            height_index_map.iter().map(|(k, v)| (*v, *k)).collect();

        let leaves: BTreeMap<_, _> = reduced_leaves
            .into_iter()
            .enumerate()
            .skip(1)
            .map(|(i, m)| (inverse_index_map[&i], m))
            .collect();

        let (fri_proof, query_indices) = tracing::trace_span!("fri Proof")
            .in_scope(|| prove(committer, pcs.fri_config(), leaves, challenger));

        let query_openings_span = tracing::trace_span!("compute query openings").entered();

        let data = rounds.into_iter().map(|(data, _)| data).collect::<Vec<_>>();
        let query_openings = committer.mmcs_committer.query_open_batch(
            &query_indices,
            &data,
            log_global_max_height,
            false,
        );

        query_openings_span.exit();

        (all_opened_values, TwoAdicFriPcsProof { fri_proof, query_openings })
    }
}

pub(super) mod merkle_tree_opening_prover {
    use std::any::TypeId;

    use serde::{de::DeserializeOwned, Serialize};

    use p3_baby_bear::BabyBear;
    use p3_bn254_fr::Bn254Fr;
    use p3_field::{PackedField, PackedValue};
    use p3_fri::BatchOpening;
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_symmetric::{CryptographicHasher, PseudoCompressionFunction};
    use p3_util::log2_ceil_usize;

    use crate::{
        device::memory::ToDevice,
        matrix::MatrixViewDevice,
        merkle_tree::{FieldMerkleTreeDeviceCommitter, FieldMerkleTreeHasher, MmcsProverData},
    };

    use super::*;

    pub enum FieldId {
        BabyBear = 0,
        Bn254 = 1,
    }

    impl<Hasher, P, PW, H, C, const DIGEST_ELEMS: usize>
        FriQueryProver<BabyBear, FieldMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>>
        for FieldMerkleTreeDeviceCommitter<Hasher>
    where
        Hasher: FieldMerkleTreeHasher<BabyBear, Digest = [PW::Value; DIGEST_ELEMS]>,
        P: PackedField<Scalar = BabyBear>,
        PW: PackedValue,
        H: CryptographicHasher<P::Scalar, [PW::Value; DIGEST_ELEMS]>,
        H: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
        H: Sync,
        C: PseudoCompressionFunction<[PW::Value; DIGEST_ELEMS], 2>,
        C: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
        C: Sync,
        PW::Value: Eq,
        [PW::Value; DIGEST_ELEMS]: Serialize + DeserializeOwned,
    {
        fn query_open_batch(
            &self,
            query_indices: &[usize],
            prover_data_slice: &[&Self::ProverData],
            log_global_max_height: usize,
            is_answering: bool,
        ) -> Vec<Vec<BatchOpening<BabyBear, FieldMerkleTreeMmcs<P, PW, H, C, DIGEST_ELEMS>>>>
        {
            // Function runs one kernel for all query indices and all matrices.
            //
            // 1. Collect relevant data and calculate offsets based on matrix.width.
            // 2. Run kernel that returns one output buffer full of data:
            //  Output buffer is 1D representation of 4D:
            // [query_index][data_index][matrix_index][matrix_width]
            // 3. Slice buffer to proper structure.
            // 4. Calculate proofs for each data.
            let total_data = prover_data_slice.len();
            let total_matrices: usize =
                prover_data_slice.iter().map(|data| data.matrices().len()).sum();
            let mut matrix_views: Vec<MatrixViewDevice<BabyBear>> =
                Vec::with_capacity(total_matrices);
            let mut width_offsets: Vec<usize> = Vec::with_capacity(total_matrices + 1);
            let mut log_max_heights: Vec<usize> = Vec::with_capacity(total_data);
            let mut log_max_heights_offsets: Vec<usize> = Vec::with_capacity(total_data);
            let mut data_matrix_offsets: Vec<usize> = Vec::with_capacity(total_data);
            let mut digests = Vec::new();
            let mut total_width = 0;
            let mut data_matrix_offset = 0;
            let mut max_width = 0;
            let mut log_max_height = 0;
            let mut total_log_max_heights = 0;

            prover_data_slice.iter().for_each(|data| {
                let mut max_height = 0;
                data_matrix_offsets.push(data_matrix_offset);
                data_matrix_offset += data.matrices().len();

                data.matrices().iter().for_each(|matrix| {
                    matrix_views.push(matrix.view());
                    width_offsets.push(total_width);
                    let matrix_width = matrix.width();
                    total_width += matrix_width;
                    max_width = std::cmp::max(max_width, matrix_width);
                    max_height = std::cmp::max(max_height, matrix.height());
                });
                log_max_height = log2_ceil_usize(max_height);
                log_max_heights.push(log_max_height);
                log_max_heights_offsets.push(total_log_max_heights);
                total_log_max_heights += log_max_height;

                digests.extend(data.digest_layers[0..log_max_height].iter().map(|d| d.as_ptr()));
            });

            width_offsets.push(total_width);
            assert_eq!(data_matrix_offset, total_matrices);
            assert_eq!(total_log_max_heights, digests.len());

            let matrix_views_device = matrix_views.to_device().unwrap();
            let width_offsets_device = width_offsets.to_device().unwrap();
            let query_indices_device = query_indices.to_vec().to_device().unwrap();

            let total_query_indices = query_indices_device.len();
            let openings_capacity = total_width * total_query_indices;
            let mut total_openings_device: DeviceBuffer<BabyBear> =
                DeviceBuffer::with_capacity(openings_capacity).unwrap();

            let log_max_heights_device = log_max_heights.to_device().unwrap();
            let log_max_heights_offsets_device = log_max_heights_offsets.to_device().unwrap();
            let digests_device = digests.to_device().unwrap();
            let mut total_proofs_device: DeviceBuffer<[PW::Value; DIGEST_ELEMS]> =
                DeviceBuffer::with_capacity(total_log_max_heights * total_query_indices).unwrap();

            unsafe {
                total_openings_device.set_len(openings_capacity);
                opening_gpu::calculate_openings(
                    matrix_views_device.as_ptr(),
                    width_offsets_device.as_ptr(),
                    query_indices_device.as_ptr(),
                    total_matrices,
                    total_width,
                    max_width,
                    total_query_indices,
                    log_global_max_height,
                    is_answering,
                    total_openings_device.as_mut_ptr(),
                );
                total_proofs_device.set_len(total_log_max_heights * total_query_indices);

                let field_id = match TypeId::of::<PW::Value>() {
                    x if x == TypeId::of::<BabyBear>() => FieldId::BabyBear,
                    x if x == TypeId::of::<Bn254Fr>() => FieldId::Bn254,
                    _ => panic!("Unsupported field"),
                };
                opening_gpu::calculate_proofs(
                    query_indices_device.as_ptr(),
                    log_max_heights_device.as_ptr(),
                    log_max_heights_offsets_device.as_ptr(),
                    total_query_indices,
                    total_data,
                    log_global_max_height,
                    total_log_max_heights,
                    digests_device.as_ptr() as *const *const *const std::ffi::c_void,
                    total_proofs_device.as_mut_ptr() as *mut *mut std::ffi::c_void,
                    is_answering,
                    field_id as usize,
                );
            }
            let total_openings_host = total_openings_device.to_host();
            let total_proofs_host = total_proofs_device.to_host();

            query_indices
                .iter()
                .enumerate()
                .map(|(index_i, _)| {
                    let index_offset = index_i * total_width;
                    prover_data_slice
                        .iter()
                        .enumerate()
                        .map(|(data_i, data)| {
                            let data_offset = data_matrix_offsets[data_i];
                            let openings: Vec<Vec<BabyBear>> = data
                                .leaves
                                .iter()
                                .enumerate()
                                .map(|(matrix_i, _)| {
                                    let start =
                                        index_offset + width_offsets[data_offset + matrix_i];
                                    let end =
                                        index_offset + width_offsets[data_offset + matrix_i + 1];
                                    total_openings_host[start..end].to_vec()
                                })
                                .collect();

                            let log_max_height = log_max_heights[data_i];

                            let proof_start =
                                index_i * total_log_max_heights + log_max_heights_offsets[data_i];
                            let proof_end = proof_start + log_max_height;
                            let proof = total_proofs_host[proof_start..proof_end].to_vec();

                            BatchOpening { opened_values: openings, opening_proof: proof }
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn prove<SC, C>(
    committer: &TwoAdicFriCommitter<SC, C>,
    config: &PcsConfig<SC>,
    input: BTreeMap<usize, ColMajorMatrixDevice<SC::Val>>,
    challenger: &mut Challenger<SC>,
) -> (FriProof<SC::Challenge, FriMmcs<SC>, SC::Val>, Vec<usize>)
where
    SC: BabyBearFriConfig,
    C: FriQueryProver<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
{
    let log_max_height = input.keys().max().copied().unwrap();

    debug_assert_eq!(committer.log_blowup, config.log_blowup);
    let commit_phase_result = trace_span!("Commit phase")
        .in_scope(|| commit_phase(committer, input, log_max_height, challenger));

    let pow_witness =
        trace_span!("POW witness").in_scope(|| challenger.grind(config.proof_of_work_bits));

    let query_indices: Vec<usize> =
        (0..config.num_queries).map(|_| challenger.sample_bits(log_max_height)).collect();

    let query_proofs_span = trace_span!("Compute query proofs").entered();

    let query_proofs_data = committer.mmcs_committer.query_open_batch(
        &query_indices,
        commit_phase_result.data.iter().collect::<Vec<_>>().as_slice(),
        log_max_height,
        true,
    );
    let query_proofs = query_proofs_data
        .into_iter()
        .enumerate()
        .map(|(q, per_query)| {
            let commit_phase_openings = per_query
                .into_iter()
                .enumerate()
                .map(|(i, batch_opening)| {
                    let BatchOpening { opened_values, opening_proof } = batch_opening;
                    let index_i = query_indices[q] >> i;
                    let index_i_sibling = index_i ^ 1;

                    let (mut opened_rows, opening_proof) = (opened_values, opening_proof);
                    assert_eq!(opened_rows.len(), 1);

                    let opened_row = opened_rows.pop().unwrap();
                    let opened_row_ext = (0..opened_row.len() / 4)
                        .map(|j| SC::Challenge::from_base_slice(&opened_row[j * 4..(j + 1) * 4]))
                        .collect::<Vec<_>>();
                    assert_eq!(opened_row_ext.len(), 2, "Committed data should be in pairs");
                    let sibling_value = opened_row_ext[index_i_sibling % 2];

                    CommitPhaseProofStep { sibling_value, opening_proof }
                })
                .collect();

            QueryProof { commit_phase_openings }
        })
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

pub fn fold_even_odd<SC: BabyBearFriConfig>(
    evaluations: &ColMajorMatrixDevice<SC::Val>,
    input_leaves: Option<ColMajorMatrixDevice<SC::Val>>,
    beta: SC::Challenge,
) -> ColMajorMatrixDevice<BabyBear> {
    let mut output =
        ColMajorMatrixDevice::with_capacity(evaluations.width(), evaluations.height() / 2).unwrap();

    let g_inv = SC::Val::two_adic_generator(log2_strict_usize(evaluations.height()) + 1).inverse();
    let one_half = SC::Val::two().inverse();
    let half_beta = beta * one_half;

    let mut powers = shifted_powers::<SC>(g_inv, half_beta, evaluations.height());
    powers.bit_reverse_rows().unwrap();

    unsafe {
        let input_view = input_leaves
            .as_ref()
            .map(|leaves| leaves.view())
            .unwrap_or(MatrixViewDevice::null(false));
        output.set_max_width();
        opening_gpu::fold_even_odd_raw(
            evaluations.view(),
            input_view,
            output.view_mut(),
            powers.view(),
            BabyBear::two().inverse(),
            input_leaves.is_some(),
        );
    }

    output
}

pub fn shifted_powers<SC: BabyBearFriConfig>(
    g: SC::Val,
    shift: SC::Challenge,
    n: usize,
) -> ColMajorMatrixDevice<SC::Val> {
    let mut output = ColMajorMatrixDevice::with_capacity(
        <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
        n,
    )
    .unwrap();

    let num_threads = 256;
    let num_blocks = n.div_ceil(num_threads);

    assert!(num_blocks > 0);

    let block_powers = g.powers().take(num_threads).collect::<Vec<_>>().to_device().unwrap();

    unsafe {
        output.set_max_width();
        opening_gpu::shifted_powers_raw(
            block_powers.as_ptr(),
            shift,
            output.view_mut(),
            n,
            num_threads,
            num_blocks,
        );
    }

    output
}

pub fn commit_phase<SC, C>(
    committer: &TwoAdicFriCommitter<SC, C>,
    mut input: BTreeMap<usize, ColMajorMatrixDevice<SC::Val>>,
    log_max_height: usize,
    challenger: &mut Challenger<SC>,
) -> CommitPhaseResult<SC, C>
where
    SC: BabyBearFriConfig,
    C: MmcsCommitter<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
{
    let mut leaves = input.remove(&log_max_height).unwrap();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (committer.log_blowup..log_max_height).rev() {
        let temp = core::mem::replace(&mut leaves, ColMajorMatrixDevice::null());
        let (commit, prover_data) = committer.mmcs_commit(vec![temp]);
        challenger.observe(commit.clone());

        let beta: SC::Challenge = challenger.sample();

        let injected_input = input.remove(&log_folded_height);
        leaves = fold_even_odd::<SC>(&prover_data.matrices()[0], injected_input, beta);

        commits.push(commit);
        data.push(prover_data);
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    let leaves = leaves.to_host();
    assert_eq!(
        leaves.values.len(),
        (1 << committer.log_blowup) * <SC::Challenge as AbstractExtensionField<SC::Val>>::D
    );
    let final_poly = SC::Challenge::from_base_slice(
        &leaves.values[0..<SC::Challenge as AbstractExtensionField<SC::Val>>::D],
    );
    challenger.observe_ext_element(final_poly);

    CommitPhaseResult { commits, data, final_poly }
}

pub struct CommitPhaseResult<SC, C>
where
    SC: BabyBearFriConfig,
    C: MmcsCommitter<SC::Val, SC::ValMmcs>,
{
    commits: Vec<Com<SC>>,
    data: Vec<C::ProverData>,
    final_poly: SC::Challenge,
}

impl<SC> Default for FriOpeningProver<SC> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

pub mod opening_gpu {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use crate::{
        cuda_runtime::stream::CudaStreamHandle,
        matrix::{MatrixViewDevice, MatrixViewMutDevice},
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[link_name = "opening_gpu"]
    #[allow(unused_attributes)]
    extern "C" {

        #[link_name = "shiftedPowers"]
        pub fn shifted_powers_raw(
            block_powers: *const F,
            shift: EF,
            output: MatrixViewMutDevice<F>,
            n: usize,
            num_threads: usize,
            num_blocks: usize,
        );

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
            log_heights: *const usize,
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
            log_heights: *const usize,
            max_height: usize,
            inv_indices: *const usize,
            reduced_openings: *const EF,
            reduced_openings_leaves: *mut MatrixViewMutDevice<F>,
            height_indices: *const usize,
            num_points: usize,
        );

        #[link_name = "numBlocksSums"]
        pub fn num_block_sums(max_height: usize) -> usize;

        #[link_name = "calculateOpenings"]
        pub fn calculate_openings(
            matrix_ptr: *const MatrixViewDevice<F>,
            width_offsets: *const usize,
            query_indices: *const usize,
            total_matrices: usize,
            total_width: usize,
            max_width: usize,
            total_indices: usize,
            log_max_height: usize,
            is_answering: bool,
            output: *mut F,
        );

        #[link_name = "calculateProof"]
        pub fn calculate_proofs(
            query_indices: *const usize,
            log_max_heights: *const usize,
            log_max_heights_offsets: *const usize,
            total_indices: usize,
            total_data: usize,
            log_max_height: usize,
            sum_log_max_height: usize,
            digests: *const *const *const std::ffi::c_void,
            output: *mut *mut std::ffi::c_void,
            is_answering: bool,
            field_id: usize,
        );

        #[link_name = "batchMultiplicativeInverse"]
        pub fn batch_multiplicative_inverse(input: *const EF, output: *mut EF, num_elements: usize);

        #[link_name = "foldEvenOdd"]
        pub fn fold_even_odd_raw(
            evaluations: MatrixViewDevice<F>,
            input_leaves: MatrixViewDevice<F>,
            output: MatrixViewMutDevice<F>,
            powers: MatrixViewDevice<F>,
            one_half: F,
            input_exists: bool,
        );
    }

    #[link_name = "fri_batch"]
    #[allow(unused_attributes)]
    extern "C" {
        #[link_name = "batchFri"]
        pub fn batch_fri_update(
            leaf_matrix: MatrixViewMutDevice<F>,
            polynomial_batch: *const F,
            evaluations: *const EF,
            domain_generator: F,
            shift: F,
            evaluation_point: EF,
            batching_challenge: EF,
            batching_challenge_offset: EF,
            width: usize,
            log_height: usize,
            stream: CudaStreamHandle,
        );
    }
}
