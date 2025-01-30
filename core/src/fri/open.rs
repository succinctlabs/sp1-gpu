use std::{collections::BTreeMap, marker::PhantomData};

use p3_challenger::FieldChallenger;
use p3_commit::{Mmcs, TwoAdicMultiplicativeCoset};
use p3_field::Field;
use sp1_core_machine::utils::log2_strict_usize;
use sp1_stark::Com;

use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_field::{AbstractExtensionField, AbstractField, TwoAdicField};
use p3_fri::{BatchOpening, CommitPhaseProofStep, FriProof, QueryProof};
use sp1_stark::Challenger;

use crate::{
    challenger::grinding_challenger::DeviceGrindingChallenger,
    cuda_runtime::stream::CudaStream,
    device::{
        memory::{ToDevice, ToHost},
        DeviceBuffer,
    },
    fri::TwoAdicFriCommitter,
    matrix::{ColMajorMatrixDevice, MatrixViewDevice},
    merkle_tree::{MmcsCommitterAsync, MmcsProverData},
    stark::{BabyBearFriConfig, FriMmcs, PcsConfig},
};

#[derive(Clone, Debug)]
pub struct FriOpeningProver<SC> {
    pub(crate) domain_normalizers: Vec<BabyBear>,
    _marker: PhantomData<SC>,
}

impl<SC> FriOpeningProver<SC> {
    pub const fn new(domain_normalizers: Vec<BabyBear>) -> Self {
        Self { domain_normalizers, _marker: PhantomData }
    }
}

pub trait FriQueryProver<F: Field, ValMmcs: Mmcs<F>>: MmcsCommitterAsync<F, ValMmcs> {
    fn query_open_batch(
        &self,
        query_indices: &[usize],
        prover_data_slice: &[&Self::ProverData],
        log_global_max_height: usize,
        is_answering: bool,
        stream: &CudaStream,
    ) -> Vec<Vec<BatchOpening<F, ValMmcs>>>;
}

impl<SC: BabyBearFriConfig> FriOpeningProver<SC> {
    #[inline]
    pub fn eval(
        &self,
        domain: TwoAdicMultiplicativeCoset<SC::Val>,
        trace: &ColMajorMatrixDevice<SC::Val>,
        point: SC::Challenge,
    ) -> DeviceBuffer<SC::Challenge> {
        let mut open =
            DeviceBuffer::<SC::Challenge>::with_capacity_in(trace.width(), trace.stream()).unwrap();
        unsafe {
            open.set_max_len();
        }
        let normalizer = self.domain_normalizers[domain.log_n];
        trace.eval(&mut open, normalizer, domain.shift, point).unwrap();
        open
    }

    pub fn prove<C>(
        &self,
        committer: &TwoAdicFriCommitter<SC, C>,
        config: &PcsConfig<SC>,
        input: BTreeMap<usize, ColMajorMatrixDevice<SC::Val>>,
        challenger: &mut Challenger<SC>,
        main_stream: &CudaStream,
    ) -> (FriProof<SC::Challenge, FriMmcs<SC>, SC::Val>, Vec<usize>)
    where
        C: FriQueryProver<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
    {
        let log_max_height = Iterator::max(input.keys()).copied().unwrap();

        debug_assert_eq!(committer.log_blowup, config.log_blowup);
        let commit_phase_result = tracing::debug_span!("commit phase")
            .in_scope(|| commit_phase(committer, input, log_max_height, challenger));

        let pow_witness = tracing::debug_span!("pow witness")
            .in_scope(|| challenger.grind_device(config.proof_of_work_bits, main_stream));

        let query_indices: Vec<usize> =
            (0..config.num_queries).map(|_| challenger.sample_bits(log_max_height)).collect();

        let query_proofs_span = tracing::debug_span!("compute query proofs").entered();

        let query_proofs_data = committer.mmcs_committer.query_open_batch(
            &query_indices,
            commit_phase_result.data.iter().collect::<Vec<_>>().as_slice(),
            log_max_height,
            true,
            main_stream,
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
                            .map(|j| {
                                SC::Challenge::from_base_slice(&opened_row[j * 4..(j + 1) * 4])
                            })
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

    pub fn batch_update(
        &self,
        reduced_openings: &mut DeviceBuffer<SC::Challenge>,
        polynomial_batch: &ColMajorMatrixDevice<SC::Val>,
        shift: SC::Val,
        evaluations: &DeviceBuffer<SC::Challenge>,
        evaluation_point: SC::Challenge,
        batching_challenge: SC::Challenge,
        batching_challenge_offset: &mut SC::Challenge,
    ) {
        let log_height = polynomial_batch.height().ilog2() as usize;
        let domain_generator = BabyBear::two_adic_generator(log_height);
        let width = polynomial_batch.width();
        unsafe {
            opening_gpu::batch_fri_update(
                reduced_openings.as_mut_ptr(),
                polynomial_batch.values.as_ptr(),
                evaluations.as_ptr(),
                domain_generator,
                shift,
                evaluation_point,
                batching_challenge,
                *batching_challenge_offset,
                width,
                log_height,
                polynomial_batch.stream().handle(),
            );
        }
        *batching_challenge_offset *= batching_challenge.exp_u64(width as u64);
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
            stream: &CudaStream,
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

            let matrix_views_device = matrix_views.to_device_async(stream).unwrap();
            let width_offsets_device = width_offsets.to_device_async(stream).unwrap();
            let query_indices_device = query_indices.to_device_async(stream).unwrap();

            let total_query_indices = query_indices_device.len();
            let openings_capacity = total_width * total_query_indices;
            let mut total_openings_device: DeviceBuffer<BabyBear> =
                DeviceBuffer::with_capacity_in(openings_capacity, stream).unwrap();

            let log_max_heights_device = log_max_heights.to_device_async(stream).unwrap();
            let log_max_heights_offsets_device =
                log_max_heights_offsets.to_device_async(stream).unwrap();
            let digests_device = digests.to_device_async(stream).unwrap();
            let mut total_proofs_device: DeviceBuffer<[PW::Value; DIGEST_ELEMS]> =
                DeviceBuffer::with_capacity_in(total_log_max_heights * total_query_indices, stream)
                    .unwrap();

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
                    stream.handle(),
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
                    stream.handle(),
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

pub fn fold_even_odd<SC: BabyBearFriConfig>(
    evaluations: &ColMajorMatrixDevice<SC::Val>,
    input_leaves: Option<ColMajorMatrixDevice<SC::Val>>,
    beta: SC::Challenge,
) -> ColMajorMatrixDevice<BabyBear> {
    let stream = evaluations.stream();
    let mut output = ColMajorMatrixDevice::with_capacity_in(
        evaluations.width(),
        evaluations.height() / 2,
        stream,
    )
    .unwrap();

    let g_inv = SC::Val::two_adic_generator(log2_strict_usize(evaluations.height()) + 1).inverse();
    let one_half = SC::Val::two().inverse();
    let half_beta = beta * one_half;

    let mut powers = shifted_powers::<SC>(g_inv, half_beta, evaluations.height(), stream);
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
            stream.handle(),
        );
    }

    output
}

pub fn shifted_powers<SC: BabyBearFriConfig>(
    g: SC::Val,
    shift: SC::Challenge,
    n: usize,
    stream: &CudaStream,
) -> ColMajorMatrixDevice<SC::Val> {
    let mut output = ColMajorMatrixDevice::with_capacity_in(
        <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
        n,
        stream,
    )
    .unwrap();

    let num_threads = 256;
    let num_blocks = n.div_ceil(num_threads);

    assert!(num_blocks > 0);

    let block_powers =
        g.powers().take(num_threads).collect::<Vec<_>>().to_device_async(stream).unwrap();

    unsafe {
        output.set_max_width();
        opening_gpu::shifted_powers_raw(
            block_powers.as_ptr(),
            shift,
            output.view_mut(),
            n,
            num_threads,
            num_blocks,
            stream.handle(),
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
    C: MmcsCommitterAsync<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
{
    let mut leaves = input.remove(&log_max_height).unwrap();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (committer.log_blowup..log_max_height).rev() {
        let temp = core::mem::replace(&mut leaves, ColMajorMatrixDevice::null());
        let stream = temp.stream().clone();
        let (commit, prover_data) = committer.mmcs_commit(vec![temp], &stream);

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
    C: MmcsCommitterAsync<SC::Val, SC::ValMmcs>,
{
    commits: Vec<Com<SC>>,
    data: Vec<C::ProverData>,
    final_poly: SC::Challenge,
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
    /// cbindgen:ignore
    extern "C" {

        #[link_name = "shiftedPowers"]
        pub fn shifted_powers_raw(
            block_powers: *const F,
            shift: EF,
            output: MatrixViewMutDevice<F>,
            n: usize,
            num_threads: usize,
            num_blocks: usize,
            stream: CudaStreamHandle,
        );

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
            stream: CudaStreamHandle,
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
            stream: CudaStreamHandle,
        );

        #[link_name = "foldEvenOdd"]
        pub fn fold_even_odd_raw(
            evaluations: MatrixViewDevice<F>,
            input_leaves: MatrixViewDevice<F>,
            output: MatrixViewMutDevice<F>,
            powers: MatrixViewDevice<F>,
            one_half: F,
            input_exists: bool,
            stream: CudaStreamHandle,
        );
    }

    #[link_name = "fri_batch"]
    #[allow(unused_attributes)]
    /// cbindgen:ignore
    extern "C" {
        #[link_name = "batchFri"]
        pub fn batch_fri_update(
            reduced_openings: *mut EF,
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

#[cfg(test)]
mod tests {
    use crate::{
        device::memory::ToDevice, matrix::RowMajorMatrixDevice, stark::observe_device_buffer,
        univariate::subgroup_normalizer,
    };

    use super::*;
    use crate::cuda_runtime::event::CudaEvent;
    use p3_field::{extension::BinomialExtensionField, AbstractField};
    use p3_fri::TwoAdicFriPcsProof;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;
    use sp1_stark::{
        baby_bear_poseidon2::{BabyBearPoseidon2, ValMmcs},
        StarkGenericConfig,
    };

    use crate::merkle_tree::Poseidon2BabyBearCommitter;

    #[test]
    fn test_fri_pcs() {
        let log_blowup = 1;
        let log_degree = 21;
        let lde_log_height = log_degree + log_blowup;

        type SC = BabyBearPoseidon2;
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let mut rng = thread_rng();

        let main_stream = CudaStream::create().unwrap();

        for total_degree in [21, 24, 27, 29] {
            let width = 1 << (total_degree - log_degree);
            let domain_normalizers = (0..26).map(subgroup_normalizer).collect::<Vec<BabyBear>>();
            let opening_prover = FriOpeningProver::<SC>::new(domain_normalizers);

            let trace = RowMajorMatrix::<BabyBear>::rand(&mut rng, 1 << log_degree, width);

            let domain = TwoAdicMultiplicativeCoset::<BabyBear> {
                log_n: log_degree,
                shift: BabyBear::one(),
            };
            let trace = trace.to_device_async(&main_stream).unwrap().to_column_major();
            let eval = (domain, &trace, CudaEvent::new().unwrap());
            let pcs = TwoAdicFriCommitter::<SC, Poseidon2BabyBearCommitter>::new(log_blowup);

            main_stream.synchronize().unwrap();
            let time = std::time::Instant::now();
            let (commit, prover_data) = pcs.commit(&[eval], &main_stream);

            let config = SC::default();
            let mut challenger = config.challenger();
            challenger.observe(commit);
            let zeta: EF = challenger.sample_ext_element();
            let mut batched_openings =
                DeviceBuffer::<EF>::with_capacity_in(1 << lde_log_height, &main_stream).unwrap();
            unsafe {
                batched_openings.set_max_len();
                batched_openings.set(0).unwrap();
            }
            let open = opening_prover.eval(domain, &trace, zeta);
            observe_device_buffer::<SC>(&mut challenger, &open);

            let mut alpha_offset = EF::one();
            let alpha = challenger.sample_ext_element();
            let lde = &prover_data.matrices()[0];
            opening_prover.batch_update(
                &mut batched_openings,
                lde,
                BabyBear::generator(),
                &open,
                zeta,
                alpha,
                &mut alpha_offset,
            );

            let base_values = unsafe { batched_openings.flatten_to_base::<BabyBear>() };
            let leaf_matrix = RowMajorMatrixDevice::new(
                base_values,
                2 * <EF as AbstractExtensionField<BabyBear>>::D,
            )
            .to_column_major();

            let input_leaves = BTreeMap::from([(lde_log_height, leaf_matrix)]);

            let (fri_proof, query_indices) = opening_prover.prove(
                &pcs,
                config.pcs().fri_config(),
                input_leaves,
                &mut challenger,
                &main_stream,
            );

            let query_openings: Vec<Vec<BatchOpening<F, ValMmcs>>> =
                pcs.mmcs_committer.query_open_batch(
                    &query_indices,
                    &[&prover_data],
                    lde_log_height,
                    false,
                    &main_stream,
                );

            let opening_proof = TwoAdicFriPcsProof { fri_proof, query_openings };
            main_stream.synchronize().unwrap();
            let elapsed = time.elapsed();
            println!("time for total degree {}: {:?}", total_degree, elapsed);
            drop(opening_proof);
        }
    }
}
