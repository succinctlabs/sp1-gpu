use std::collections::BTreeMap;
use std::collections::HashMap;
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
use p3_fri::CommitPhaseProofStep;
use p3_fri::FriConfig;
use p3_fri::FriProof;
use p3_fri::QueryProof;
use p3_fri::{BatchOpening, TwoAdicFriPcsProof};
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
use crate::matrix::DeviceMatrix;
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

        let mut reduced_leaf =
            ColMajorMatrixDevice::<F>::with_capacity(2 * <EF as AbstractExtensionField<F>>::D, 1);
        unsafe {
            reduced_leaf.set_max_width();
        }
        reduced_leaves.push(reduced_leaf);

        let get_data_for_device_span = tracing::trace_span!("Get data for device").entered();
        mats_and_points.iter().for_each(|(mats, points)| {
            mats.iter()
                .zip(points.iter())
                .for_each(|(mat, points_for_mat)| {
                    let log_height = log2_strict_usize(mat.height);
                    height_index_map
                        .entry(log_height)
                        .or_insert_with_key(|log_height| {
                            let idx = reduced_leaves.len();
                            height_indices[*log_height] = idx;
                            let mut reduced_leaf = ColMajorMatrixDevice::<F>::with_capacity(
                                2 * <EF as AbstractExtensionField<F>>::D,
                                1 << (log_height - 1),
                            );
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

        let ys_out_host_span = tracing::trace_span!("ys_out_host").entered();
        let ys_output = ys_output_buffer.to_host();
        ys_out_host_span.exit();

        assert_eq!(ys_output.len(), total_polys);

        let compute_openings_span = tracing::trace_span!("Compute opened values").entered();
        let mut point_index = 0;
        let all_opened_values = {
            let mut reduced_openings_device = DeviceBuffer::<EF>::with_capacity(inv_offset);

            // Compute openings fused.
            let compute_reduced_openings_span =
                tracing::trace_span!("Compute reduced openings on device").entered();
            let alpha_pow_offsets_device = alpha_pow_offsets.to_device();
            let log_heights = matrices_for_openings
                .iter()
                .map(|mat| log2_strict_usize(mat.height))
                .collect::<Vec<_>>()
                .to_device();
            let matrices_for_openings = matrices_for_openings.to_device();
            let ys_indices = ys_indices.to_device();

            let mut reduce_leaves_raw = reduced_leaves
                .iter_mut()
                .map(|mat| mat.view_mut())
                .collect::<Vec<_>>()
                .to_device();

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

            let height_indices = height_indices.to_device();
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

        let (fri_proof, query_indices) = tracing::trace_span!("Fri Proof")
            .in_scope(|| prove(pcs.fri_config(), leaves, challenger));

        let query_openings_span = tracing::trace_span!("Compute query openings").entered();

        let query_openings_data = query_open_batch(
            &query_indices, 
            rounds.iter().map(|(data, _)| *data).collect(),
            log_global_max_height,
            false
        );
        let query_openings = query_openings_data.into_iter().map(|per_query| {
            per_query.iter().map(|(openings, proof)| {
                BatchOpening::<SC::Val, InnerValMmcs> {
                    opened_values: openings.to_vec(),
                    opening_proof: proof.to_vec(),
                }
            }).collect::<Vec<_>>()
        }).collect::<Vec<_>>();

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

fn query_open_batch(
    query_indices: &[usize],
    prover_datas: Vec<&GpuProverData<SC>>,
    log_global_max_height: usize,
    is_answering: bool,
) -> Vec<Vec<(Vec<Vec<F>>, Vec<[F; DIGEST_WIDTH]>)>> {
    let total_matrices: usize = prover_datas.iter().map(|data| data.leaves.len()).sum();
    let mut matrix_views: Vec<MatrixViewDevice<F>> = Vec::with_capacity(total_matrices);
    let mut width_offsets: Vec<usize> = Vec::with_capacity(total_matrices+1);
    let mut log2_max_heights: Vec<usize> = Vec::with_capacity(prover_datas.len());
    let mut data_matrix_offsets: Vec<usize> = Vec::with_capacity(prover_datas.len());
    let mut total_width = 0;
    let mut data_matrix_offset = 0;
    let mut max_width = 0;
    prover_datas.iter().for_each(|data| {
        let mut max_height = 0;           
        data_matrix_offsets.push(data_matrix_offset);
        data_matrix_offset += data.leaves.len();

        data.leaves.iter().for_each(|matrix| {
            matrix_views.push(matrix.view());
            width_offsets.push(total_width);
            let matrix_width = matrix.width();
            total_width += matrix_width;
            max_width = std::cmp::max(max_width, matrix_width);
            max_height = std::cmp::max(max_height, matrix.height());
        });
        log2_max_heights.push(log2_ceil_usize(max_height));
    });
    width_offsets.push(total_width);
    assert_eq!(data_matrix_offset, total_matrices);

    let matrix_views_device = matrix_views.to_device();
    let width_offsets_device = width_offsets.to_device();
    let query_indices_device = query_indices.to_vec().to_device();
    
    let total_query_indices = query_indices_device.len();
    let output_capacity = total_width * total_query_indices;
    let mut total_output_device: DeviceBuffer<F> = DeviceBuffer::with_capacity(output_capacity);
    unsafe {
        total_output_device.set_len(output_capacity);
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
            total_output_device.as_mut_ptr()
        );
    }
    let total_output_host = total_output_device.to_host();

    query_indices.iter().enumerate().map(|(index_i, &index)| {
        let index_offset = index_i * total_width;
        prover_datas.iter().enumerate().map(|(data_i, data)| {
            let data_offset = data_matrix_offsets[data_i];
            let openings: Vec<Vec<F>> = data.leaves.iter().enumerate().map(|(matrix_i, _)| {
                let start = index_offset + width_offsets[data_offset + matrix_i];
                let end	  = index_offset + width_offsets[data_offset + matrix_i + 1];
                total_output_host[start..end].to_vec()
            }).collect();

            let log_max_height = log2_max_heights[data_i];
            let bits_reduced = if is_answering {
                data_i + 1
            } else {
                log_global_max_height - log_max_height
            };
            let data_index = index >> bits_reduced;
            let proof = (0..log_max_height)
                .map(|i| {
                    let start = (data_index >> i) ^ 1;
                    let end = start + 1;
                    data.digest_layers[i][start..end].to_host()[0]
                }).collect();

            (openings, proof)
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>()
}

fn open_batch(
    index: usize,
    prover_data: &FieldMerkleTreeGpu<
        BabyBear,
        [BabyBear; DIGEST_WIDTH],
        CudaSync<ColMajorMatrixDevice<BabyBear>>,
    >,
) -> (Vec<Vec<F>>, Vec<[F; DIGEST_WIDTH]>) {
    // 1. For each matrix in prover_data.leaves this function prepares
    // * vector of matrix widths and it's offsets in output vec 
    //     offset[i] = sum(widths[j]), j = 0..i
    // * vector of matrix views and corresponding indexes 
    //     matrix_idxs[thread_idx] = matrix idx
    //
    // 2. Calculate sum of all widths to allocate proper output buffer (total_width)
    // 3. Run kernel that take all matrices into one output buffer
    // 4. Split output vector into Vec<Vec<>> using offsets
    // 5. Copy proof to host (not changed) 
    let max_height = prover_data.leaves.iter().map(|m| m.height()).max().unwrap();
    let log_max_height = log2_ceil_usize(max_height);

    // Calculate and allocate all vectors
    let widths: Vec<usize> = prover_data.leaves.iter().map(|m| m.width()).collect();
    let total_width: usize = widths.iter().sum();
    let mut total_output_device: DeviceBuffer<F> = DeviceBuffer::with_capacity(total_width);

    let matrix_num = prover_data.leaves.len();
    let mut matrix_views: Vec<MatrixViewDevice<F>> = Vec::with_capacity(matrix_num);
    let mut matrix_idxs: Vec<usize> = Vec::with_capacity(total_width);
    let mut width_offsets: Vec<usize> = Vec::with_capacity(matrix_num);
    
    let mut current_offset = 0;
    prover_data.leaves.iter().enumerate().for_each(|(i, matrix)| {
        matrix_views.push(matrix.view());
        matrix_idxs.extend(std::iter::repeat(i).take(matrix.width()));
        width_offsets.push(current_offset);
        current_offset += matrix.width();
    });

    // Run kernel for all matrices
    let matrix_views_device = matrix_views.to_device();
    let matrix_idxs_device = matrix_idxs.to_device();
    let width_offsets_device = width_offsets.to_device();
    unsafe {
        total_output_device.set_len(total_width);
        opening_gpu::total_fetch_row(
            matrix_views_device.as_ptr(),
            matrix_idxs_device.as_ptr(),
            width_offsets_device.as_ptr(),
            total_width,
            index,
            log_max_height,
            total_output_device.as_mut_ptr()
        );
    }
    let total_output_host = total_output_device.to_host();

    // Split the total_output_host into chunks corresponding to each matrix width (Use offsets for slicing)
    let openings: Vec<Vec<F>> = width_offsets.iter().enumerate().map(|(i, &offset)| {
        let width = widths[i];
        total_output_host[offset..offset + width].to_vec()
    }).collect();

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
    input: BTreeMap<usize, ColMajorMatrixDevice<F>>,
    challenger: &mut Challenger<SC>,
) -> (FriProof<EF, ChallengeMmcs, F>, Vec<usize>) {
    let log_max_height = input.keys().max().copied().unwrap();

    let commit_phase_result = trace_span!("Commit phase")
        .in_scope(|| commit_phase(config, input, log_max_height, challenger));

    let pow_witness =
        trace_span!("POW witness").in_scope(|| challenger.grind(config.proof_of_work_bits));

    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| challenger.sample_bits(log_max_height))
        .collect();

    let query_proofs_span = trace_span!("Compute query proofs").entered();
/*
    let query_proofs = query_indices
        .iter()
        .map(|&index| answer_query(&commit_phase_result.data, index))
        .collect::<Vec<_>>();
*/
    let query_proofs_data = query_open_batch(
        &query_indices, 
        commit_phase_result.data.iter().map(|commit| commit as &GpuProverData<SC>).collect(),
        log_max_height,
        true
    );
    let query_proofs = query_proofs_data.iter().enumerate().map(|(q, per_query)| {
        let commit_phase_openings = per_query.iter().enumerate().map(|(i, (openings, proof))| {
            let index_i = query_indices[q] >> i;
            let index_i_sibling = index_i ^ 1;

            let (mut opened_rows, opening_proof) = (openings.clone(), proof.clone());
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
        }).collect();
        
        QueryProof {
            commit_phase_openings,
        }
    }).collect::<Vec<_>>();

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

pub fn fold_even_odd(
    evaluations: &ColMajorMatrixDevice<F>,
    input_leaves: Option<ColMajorMatrixDevice<F>>,
    beta: EF,
) -> ColMajorMatrixDevice<F> {
    let mut output =
        ColMajorMatrixDevice::with_capacity(evaluations.width(), evaluations.height() / 2);

    let g_inv = F::two_adic_generator(log2_strict_usize(evaluations.height()) + 1).inverse();
    let one_half = F::two().inverse();
    let half_beta = beta * one_half;

    let mut powers = shifted_powers(g_inv, half_beta, evaluations.height());
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

pub fn shifted_powers(g: F, shift: EF, n: usize) -> ColMajorMatrixDevice<F> {
    let mut output = ColMajorMatrixDevice::with_capacity(<EF as AbstractExtensionField<F>>::D, n);

    let num_threads = 256;
    let num_blocks = n.div_ceil(num_threads);

    assert!(num_blocks > 0);

    let block_powers = g.powers().take(num_threads).collect::<Vec<_>>().to_device();

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

pub fn commit_phase(
    config: &FriConfig<ChallengeMmcs>,
    mut input: BTreeMap<usize, ColMajorMatrixDevice<F>>,
    log_max_height: usize,
    challenger: &mut Challenger<SC>,
) -> CommitPhaseResult {
    let mut leaves = input.remove(&log_max_height).unwrap();

    let mut commits = vec![];
    let mut data = vec![];

    for log_folded_height in (config.log_blowup..log_max_height).rev() {
        let temp = core::mem::replace(&mut leaves, ColMajorMatrixDevice::null());
        let tree = FieldMerkleTreeGpu::new(vec![CudaSync::new(temp).unwrap()]);
        let commit: Hash<F, F, DIGEST_WIDTH> = tree.root().into();
        challenger.observe(commit);

        let beta: EF = challenger.sample();

        let injected_input = input.remove(&log_folded_height);
        leaves = fold_even_odd(&tree.leaves[0], injected_input, beta);

        commits.push(commit);
        data.push(tree);
    }

    // We should be left with `blowup` evaluations of a constant polynomial.
    let leaves = leaves.to_host();
    assert_eq!(
        leaves.values.len(),
        config.blowup() * <EF as AbstractExtensionField<F>>::D
    );
    let final_poly = EF::from_base_slice(&leaves.values[0..<EF as AbstractExtensionField<F>>::D]);
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

    use crate::matrix::{MatrixViewDevice, MatrixViewMutDevice};

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

        #[link_name = "fetchRow"]
        pub fn fetch_row(matrix: MatrixViewDevice<F>, index: usize, output: *mut F);

        #[link_name = "fetchRowTotal"]
        pub fn total_fetch_row(
            matrix_ptr: *const MatrixViewDevice<F>,
            matrix_idxs: *const usize,
            width_offsets: *const usize,
            total_width: usize,
            index: usize, 
            log_max_height: usize,
            output: *mut F);
        
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
            output: *mut F);

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
}
