use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};

use itertools::Itertools;

use sp1_core::stark::ShardMainData;
use tracing::info;

use p3_baby_bear::BabyBear;
use p3_commit::PolynomialSpace;
use p3_field::AbstractExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core::stark::AirOpenedValues;

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use sp1_core::stark::Chip;
use sp1_core::stark::ChipOpenedValues;
use sp1_core::stark::DebugConstraintBuilder;
use sp1_core::stark::MachineProof;
use sp1_core::stark::MachineProver;
use sp1_core::stark::ShardCommitment;
use sp1_core::stark::ShardOpenedValues;
use sp1_core::stark::ShardProof;
use sp1_core::utils::SP1CoreOpts;
use sp1_core::{
    air::MachineAir,
    stark::{
        MachineRecord, PcsProverData, ProverConstraintFolder, StarkGenericConfig, StarkMachine,
        StarkProvingKey, Val,
    },
};
use std::cmp::Reverse;

use air::P3EvalFolder;

use crate::fri::FriGpuOpeningProver;
use crate::poseidon2::baby_bear::DeviceHasherBabyBear;
use crate::runtime::scope;
use crate::stark::DeviceQuotientValues;
use crate::stark::DeviceQuotientValuesGenerator;
use crate::utils::ChipStatistics;
use crate::{
    device::{
        error::CudaError,
        memory::{ToDevice, ToHost},
        CudaSync,
    },
    fri::TwoAdicFriCommitter,
    matrix::ColMajorMatrixDevice,
    merkle_tree::FieldMerkleTreeGpu,
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH,
};

use super::{BabyBearPoseidon2Config, PermutationTraceGenerator};

use super::natural_domain_for_degree;

const LDE_MEM_THRESHOLD: usize = 1e10 as usize;

pub struct StarkGpuProver<SC: StarkGenericConfig, H, A> {
    pub(crate) machine: StarkMachine<SC, A>,
    permutation_trace_generator: PermutationTraceGenerator<SC::Val, SC::Challenge, A>,
    quotient_generator: DeviceQuotientValuesGenerator<SC, A>,
    committer: TwoAdicFriCommitter<SC::Val, H>,
    opening_prover: FriGpuOpeningProver<SC>,
}

pub type GpuMatrix<F> = CudaSync<ColMajorMatrixDevice<F>>;

pub type GpuProverData<SC> =
    FieldMerkleTreeGpu<Val<SC>, [Val<SC>; DIGEST_WIDTH], GpuMatrix<Val<SC>>>;

pub type CpuProverData<SC> = PcsProverData<SC>;

pub type CpuMatrix<F> = RowMajorMatrix<F>;

impl<SC, A> MachineProver<SC, A> for StarkGpuProver<SC, DeviceHasherBabyBear, A>
where
    SC: BabyBearPoseidon2Config,
    A: for<'a> Air<P3EvalFolder<'a>>
        + for<'a> Air<ProverConstraintFolder<'a, SC>>
        + MachineAir<BabyBear>,
    A::Record: MachineRecord<Config = SP1CoreOpts> + Sync,
{
    type DeviceMatrix = CudaSync<ColMajorMatrixDevice<Val<SC>>>;
    type DeviceProverData = GpuProverData<SC>;
    type Error = CudaError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        let log_blowup = machine.config().pcs().fri_config().log_blowup;
        let quotient_generator = DeviceQuotientValuesGenerator::new(&machine);
        Self {
            machine,
            committer: TwoAdicFriCommitter::new(log_blowup),
            permutation_trace_generator: PermutationTraceGenerator::default(),
            opening_prover: FriGpuOpeningProver::default(),
            quotient_generator,
        }
    }

    fn machine(&self) -> &StarkMachine<SC, A> {
        &self.machine
    }

    fn generate_traces(&self, record: &A::Record) -> Vec<(String, RowMajorMatrix<Val<SC>>)> {
        let shard_chips = self.machine.shard_chips(record).collect::<Vec<_>>();
        shard_chips
            .par_iter()
            .map(|chip| {
                let trace = chip.generate_trace(record, &mut A::Record::default());
                (chip.name(), trace)
            })
            .collect::<Vec<_>>()
    }

    fn commit(
        &self,
        shard: A::Record,
        mut named_traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData> {
        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        // Get the chip ordering.
        let chip_ordering = named_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect();

        // Get the domains.
        let config = self.machine.config();
        let (domains, traces): (Vec<_>, Vec<_>) = named_traces
            .into_iter()
            .map(|(_, trace)| (natural_domain_for_degree(config, trace.height()), trace))
            .unzip();

        let span = tracing::Span::current();
        scope(|s| {
            let _span = span.enter();
            let span = tracing::Span::current();
            s.spawn(move || {
                let _span = span.enter();

                // Copy the traces to device.
                let traces: Vec<_> = traces
                    .iter()
                    .map(|trace| CudaSync::new(trace.to_device().to_column_major()).unwrap())
                    .collect();

                // Commit to the traces.
                let domains_and_traces = domains
                    .iter()
                    .copied()
                    .zip(traces.iter())
                    .collect::<Vec<_>>();
                let (commit, data) = self.committer.commit(&domains_and_traces);

                ShardMainData {
                    traces,
                    main_commit: commit,
                    main_data: data,
                    chip_ordering,
                    public_values: shard.public_values(),
                }
            })
            .sync_join()
            .unwrap()
        })
    }

    fn open(
        &self,
        pk: &StarkProvingKey<SC>,
        data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut <SC as StarkGenericConfig>::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error> {
        let span = tracing::Span::current();
        scope(move |s| {
            let _span = span.enter();
            let span = tracing::Span::current();
            s.spawn(move || {
                let _span = span.enter();
                let ShardMainData {
                    traces,
                    main_commit,
                    mut main_data,
                    chip_ordering,
                    public_values,
                } = data;

                let shard_chips = self
                    .machine
                    .shard_chips_ordered(&chip_ordering)
                    .collect::<Vec<_>>();

                let domains = traces
                    .iter()
                    .map(|trace| {
                        let config = self.machine.config();
                        natural_domain_for_degree(config, trace.height())
                    })
                    .collect::<Vec<_>>();

                // Compute some statistics.
                let mut total_lde_size = 0;
                let log_blowup = self.committer.log_blowup();
                for (chip, domain) in shard_chips.iter().zip(domains.iter()) {
                    let height = domain.size();
                    let stats = ChipStatistics::new::<SC::Challenge, _>(chip, height);
                    total_lde_size += stats.lde_memory_size(log_blowup);
                }
                info!("Total LDE size: {:.4} GB", (total_lde_size as f64) * 1e-9);

                let recompute_ldes = total_lde_size > LDE_MEM_THRESHOLD;

                // Delete the ldes of the main prover data.
                if recompute_ldes {
                    main_data.leaves.clear();
                }

                // Get the permutation challenges.
                let permutation_challenges = (0..2)
                    .map(|_| challenger.sample_ext_element())
                    .collect::<Vec<_>>();
                // Generate permutation traces.
                let permutation_traces = tracing::debug_span!("generate permutation traces")
                    .in_scope(|| {
                        self.generate_permutation_traces(
                            pk,
                            &shard_chips,
                            &traces,
                            &permutation_challenges,
                        )
                    })?;

                info!(
                    "Shard: [{}]",
                    shard_chips
                        .iter()
                        .map(|c| c.name())
                        .collect::<Vec<_>>()
                        .join(", ")
                );

                for (i, chip) in shard_chips.iter().enumerate() {
                    let width = traces[i].width();
                    let height = traces[i].height();
                    let permutation_width = permutation_traces[i].width();
                    let total_width = width + permutation_width;
                    info!(
                        "Chip {:<20}: {:<20} = {:>10}W x {:>10}H",
                        chip.name(),
                        total_width * height,
                        total_width,
                        height,
                    );
                }

                // Commit to the permutation traces.
                let perm_domains_and_traces = domains
                    .iter()
                    .copied()
                    .zip(permutation_traces)
                    .collect::<Vec<_>>();
                let (permutation_commit, mut perm_prover_data) =
                    self.committer.commit(&perm_domains_and_traces);

                // Observe the permutation commitment.
                challenger.observe(permutation_commit);

                // Get the cumulative sums from device.
                let cumulative_sums = perm_domains_and_traces
                    .iter()
                    .map(|(_, trace)| {
                        let row_idx = trace.height() - 1;
                        let start_col_idx =
                            trace.width() - <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
                        SC::Challenge::from_base_fn(|i| {
                            let index = (start_col_idx + i) * trace.height() + row_idx;
                            let val = trace.values[index..index + 1].to_host();
                            val[0]
                        })
                    })
                    .collect::<Vec<_>>();
                // Delete the ldes of the permutation prover data.
                if recompute_ldes {
                    perm_prover_data.leaves.clear();
                }

                // Get a challenge for folding the constraints.
                //
                // *Remark*: this is called `alpha` in [sp1_core].
                let folding_challenge: SC::Challenge = challenger.sample_ext_element();

                // Compute quotient values.

                // Compute values
                let quotient_values = tracing::debug_span!("quotient").in_scope(|| {
                    self.quotient_generator.generate_quotient_values(
                        &self.committer,
                        &shard_chips,
                        pk,
                        &traces,
                        &perm_domains_and_traces,
                        &permutation_challenges,
                        folding_challenge,
                        &public_values,
                        &cumulative_sums,
                    )
                })?;

                // Commit to the quotient values
                let quotient_domains_and_chunks = quotient_values
                    .into_iter()
                    .flat_map(|values| {
                        let DeviceQuotientValues {
                            quotient_chunks,
                            quotient_chunk_domains,
                        } = values;

                        quotient_chunk_domains.into_iter().zip(quotient_chunks)
                    })
                    .collect::<Vec<_>>();
                let (quotient_commit, quotient_prover_data) =
                    tracing::debug_span!("commit to quotient")
                        .in_scope(|| self.committer.commit(&quotient_domains_and_chunks));
                let num_quotient_chunks = quotient_domains_and_chunks.len();
                drop(quotient_domains_and_chunks);
                // Observe the quotient commitment.
                challenger.observe(quotient_commit);

                // Generate the opening proof and assemble the shard proof.

                // Compute the opening challenge.
                let zeta: SC::Challenge = challenger.sample_ext_element();

                let preprocessed_opening_points = pk
                    .traces
                    .iter()
                    .map(|trace| {
                        let domain =
                            natural_domain_for_degree(self.machine.config(), trace.height());
                        vec![zeta, domain.next_point(zeta).unwrap()]
                    })
                    .collect::<Vec<_>>();

                let trace_opening_points = domains
                    .iter()
                    .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
                    .collect::<Vec<_>>();

                // Compute quotient openning points, open every chunk at zeta.
                let quotient_opening_points = (0..num_quotient_chunks)
                    .map(|_| vec![zeta])
                    .collect::<Vec<_>>();

                // Recompute main and permutation LDE and insert into the prover data.
                if recompute_ldes {
                    for (domain, perm_trace) in perm_domains_and_traces {
                        let perm_lde = self.committer.encode(domain, &perm_trace, true)?;
                        perm_prover_data.leaves.push(CudaSync::new(perm_lde)?);
                    }
                    for (domain, trace) in domains.iter().zip(traces) {
                        let main_lde = self.committer.encode(*domain, &trace, true)?;
                        main_data.leaves.push(CudaSync::new(main_lde)?);
                    }
                }

                let pk_data_device = pk.data.to_device();

                let (openings, opening_proof) =
                    tracing::debug_span!("compute opening").in_scope(|| {
                        self.opening_prover.open(
                            &self.committer,
                            self.pcs(),
                            vec![
                                (&pk_data_device, preprocessed_opening_points),
                                (&main_data, trace_opening_points.clone()),
                                (&perm_prover_data, trace_opening_points),
                                (&quotient_prover_data, quotient_opening_points),
                            ],
                            challenger,
                        )
                    });
                drop(pk_data_device);
                drop(main_data);
                drop(perm_prover_data);
                drop(quotient_prover_data);

                // Collect the opened values for each chip.
                let [preprocessed_values, main_values, permutation_values, mut quotient_values] =
                    openings.try_into().unwrap();
                assert!(main_values.len() == shard_chips.len());
                let preprocessed_opened_values = preprocessed_values
                    .into_iter()
                    .map(|op| {
                        let [local, next] = op.try_into().unwrap();
                        AirOpenedValues { local, next }
                    })
                    .collect::<Vec<_>>();

                let main_opened_values = main_values
                    .into_iter()
                    .map(|op| {
                        let [local, next] = op.try_into().unwrap();
                        AirOpenedValues { local, next }
                    })
                    .collect::<Vec<_>>();

                let permutation_opened_values = permutation_values
                    .into_iter()
                    .map(|op| {
                        let [local, next] = op.try_into().unwrap();
                        AirOpenedValues { local, next }
                    })
                    .collect::<Vec<_>>();

                let mut quotient_opened_values = Vec::with_capacity(shard_chips.len());
                for chip in shard_chips.iter() {
                    let log_quotient_degree = chip.log_quotient_degree();
                    let degree = 1 << log_quotient_degree;
                    let slice = quotient_values.drain(0..degree);
                    quotient_opened_values
                        .push(slice.map(|mut op| op.pop().unwrap()).collect::<Vec<_>>());
                }

                let opened_values = main_opened_values
                    .into_iter()
                    .zip_eq(permutation_opened_values)
                    .zip_eq(quotient_opened_values)
                    .zip_eq(cumulative_sums)
                    .zip_eq(shard_chips.iter())
                    .enumerate()
                    .map(
                        |(i, ((((main, permutation), quotient), cumulative_sum), chip))| {
                            let preprocessed = pk
                                .chip_ordering
                                .get(&chip.name())
                                .map(|&index| preprocessed_opened_values[index].clone())
                                .unwrap_or(AirOpenedValues {
                                    local: vec![],
                                    next: vec![],
                                });
                            let log_degree = domains[i].size().ilog2() as usize;
                            ChipOpenedValues {
                                preprocessed,
                                main,
                                permutation,
                                quotient,
                                cumulative_sum,
                                log_degree,
                            }
                        },
                    )
                    .collect::<Vec<_>>();

                Ok(ShardProof::<SC> {
                    commitment: ShardCommitment {
                        main_commit,
                        permutation_commit,
                        quotient_commit,
                    },
                    opened_values: ShardOpenedValues {
                        chips: opened_values,
                    },
                    opening_proof,
                    chip_ordering,
                    public_values,
                })
            })
            .sync_join()
            .unwrap()
        })
    }

    /// Prove the execution record is valid.
    ///
    /// Given a proving key `pk` and a matching execution record `record`, this function generates
    /// a STARK proof that the execution record is valid.
    fn prove(
        &self,
        pk: &StarkProvingKey<SC>,
        mut records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        self.machine().generate_dependencies(&mut records, &opts);

        // Observe the preprocessed commitment.
        pk.observe_into(challenger);

        // Generate and commit the traces for each shard.
        let shard_data: Vec<_> = records
            .into_iter()
            .map(|record| {
                let traces = self.generate_traces(&record);
                self.commit(record, traces)
            })
            .collect();

        // Observe the challenges for each segment.
        shard_data.iter().for_each(|data| {
            challenger.observe(data.main_commit);
            challenger.observe_slice(&data.public_values[0..self.num_pv_elts()]);
        });

        let shard_proofs = shard_data
            .into_iter()
            .map(|data| {
                let mut challenger = challenger.clone();
                let span = tracing::Span::current();
                let _span = span.enter();
                self.open(pk, data, &mut challenger)
            })
            .collect::<Result<Vec<_>, CudaError>>()?;

        Ok(MachineProof { shard_proofs })
    }
}

impl<SC, H, A> StarkGpuProver<SC, H, A>
where
    SC: BabyBearPoseidon2Config,
    A: for<'a> Air<P3EvalFolder<'a>>
        + for<'a> Air<ProverConstraintFolder<'a, SC>>
        + MachineAir<BabyBear>,
    A::Record: Sync,
{
    pub fn pcs(&self) -> &SC::Pcs {
        self.machine.config().pcs()
    }

    pub fn generate_permutation_traces(
        &self,
        pk: &StarkProvingKey<SC>,
        chips: &[&Chip<SC::Val, A>],
        main_traces: &[GpuMatrix<SC::Val>],
        random_elements: &[SC::Challenge],
    ) -> Result<Vec<GpuMatrix<SC::Val>>, CudaError> {
        chips
            .iter()
            .zip(main_traces.iter())
            .map(|(chip, main_trace)| {
                let preprocessed_trace = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&index| pk.traces[index].to_device().to_column_major());

                let flatenned_trace = self
                    .permutation_trace_generator
                    .generate_flattened_permutation_trace(
                        chip,
                        preprocessed_trace.as_ref(),
                        main_trace,
                        random_elements,
                    )?;
                CudaSync::new(flatenned_trace)
            })
            .collect::<Result<Vec<_>, CudaError>>()
    }
}

#[cfg(test)]
pub mod tests {
    use sp1_core::{
        runtime::{ExecutionRecord, Program, Runtime},
        utils::{
            run_test,
            tests::{FIBONACCI_ELF, SSZ_WITHDRAWALS_ELF},
            SP1CoreOpts,
        },
    };

    use crate::{poseidon2::baby_bear::DeviceHasherBabyBear, utils::init_tracer};

    use super::*;

    pub fn execute_core(program: Program) -> ExecutionRecord {
        let opts = SP1CoreOpts::default();
        let mut runtime = Runtime::new(program, opts);
        runtime.run().unwrap();
        runtime.record
    }

    #[test]
    fn test_fibonacci_prove() {
        let program = Program::from(FIBONACCI_ELF);

        init_tracer();
        // Execute the program.
        run_test::<StarkGpuProver<_, DeviceHasherBabyBear, _>>(program).unwrap();
    }

    #[test]
    #[ignore]
    fn test_ssz_withdrawals_prove() {
        let program = Program::from(SSZ_WITHDRAWALS_ELF);

        init_tracer();
        // Execute the program.
        run_test::<StarkGpuProver<_, DeviceHasherBabyBear, _>>(program).unwrap();
    }
}
