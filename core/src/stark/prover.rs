use hashbrown::HashMap;

use rayon::prelude::*;

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Mmcs;
use p3_commit::PolynomialSpace;
use p3_field::AbstractExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core::stark::AirOpenedValues;

use itertools::Itertools;

use sp1_core::stark::ShardMainData;
use tracing::info;

use sp1_core::air::MachineProgram;
use sp1_core::stark::Chip;
use sp1_core::stark::ChipOpenedValues;
use sp1_core::stark::Com;
use sp1_core::stark::DebugConstraintBuilder;
use sp1_core::stark::MachineProof;
use sp1_core::stark::MachineProver;
use sp1_core::stark::ShardCommitment;
use sp1_core::stark::ShardOpenedValues;
use sp1_core::stark::ShardProof;
use sp1_core::stark::StarkVerifyingKey;
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

use crate::cuda_runtime::stream::CudaStream;
use crate::device::memory::cuda_mem_get_info;
use crate::fri::FriOpeningProver;
use crate::fri::FriQueryProver;
use crate::merkle_tree::MmcsProverData;
use crate::stark::DeviceQuotientValues;
use crate::stark::DeviceQuotientValuesGenerator;
use crate::utils::ChipStatistics;
use crate::{
    device::{
        error::CudaError,
        memory::{ToDevice, ToHost},
    },
    fri::TwoAdicFriCommitter,
    matrix::ColMajorMatrixDevice,
    merkle_tree::FieldMerkleTreeGpu,
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH,
};

use super::BabyBearFriConfig;
use super::PermutationTraceGenerator;

use super::natural_domain_for_degree;

const LDE_MEM_RATIO: f64 = 10.0 / 24.0;

/// A CUDA prover for a STARK.
pub struct StarkGpuProver<SC: BabyBearFriConfig, C, A> {
    pub(crate) machine: StarkMachine<SC, A>,
    permutation_trace_generator: PermutationTraceGenerator<SC::Val, SC::Challenge, A>,
    quotient_generator: DeviceQuotientValuesGenerator<SC, A>,
    lde_mem_threshold: usize,
    committer: TwoAdicFriCommitter<SC, C>,
    opening_prover: FriOpeningProver<SC>,
}

pub type GpuMatrix<F> = ColMajorMatrixDevice<F>;

pub type GpuProverData<SC> =
    FieldMerkleTreeGpu<Val<SC>, [Val<SC>; DIGEST_WIDTH], GpuMatrix<Val<SC>>>;

pub type CpuProverData<SC> = PcsProverData<SC>;

pub type CpuMatrix<F> = RowMajorMatrix<F>;

impl<SC, C, A> MachineProver<SC, A> for StarkGpuProver<SC, C, A>
where
    SC: BabyBearFriConfig,
    A: for<'a> Air<P3EvalFolder<'a>>
        + for<'a> Air<ProverConstraintFolder<'a, SC>>
        + MachineAir<BabyBear>,
    A::Record: MachineRecord<Config = SP1CoreOpts> + Sync,
    C: FriQueryProver<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>
        + 'static
        + Send
        + Sync
        + Default,
    C::ProverData: Send + ToHost<HostType = PcsProverData<SC>>,
    PcsProverData<SC>: ToDevice<DeviceType = C::ProverData>,
    Com<SC>: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Proof: Send + Sync,
    SC::FriChallenger: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Commitment: Send + Sync,
    SC::RowMajorProverData: Send + Sync,
{
    type DeviceMatrix = ColMajorMatrixDevice<Val<SC>>;
    type DeviceProverData = C::ProverData;
    type Error = CudaError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        let log_blowup = machine.config().pcs().fri_config().log_blowup;
        let quotient_generator = DeviceQuotientValuesGenerator::new(&machine);
        let (_, total) = cuda_mem_get_info().unwrap();
        let lde_mem_threshold = (LDE_MEM_RATIO * (total as f64)) as usize;
        tracing::info!("LDE memory threshold: {}", lde_mem_threshold);
        Self {
            machine,
            committer: TwoAdicFriCommitter::new(log_blowup),
            permutation_trace_generator: PermutationTraceGenerator::default(),
            opening_prover: FriOpeningProver::default(),
            lde_mem_threshold,
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
        let _span = span.enter();
        let span = tracing::Span::current();
        let _span = span.enter();

        // Copy the traces to device.
        let traces: Vec<_> = traces
            .iter()
            .map(|trace| {
                let stream = CudaStream::create().unwrap();
                trace.to_device_async(&stream).unwrap().to_column_major()
            })
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
    }

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(&self, program: &A::Program) -> (StarkProvingKey<SC>, StarkVerifyingKey<SC>) {
        let mut named_preprocessed_traces = tracing::debug_span!("generate preprocessed traces")
            .in_scope(|| {
                self.machine()
                    .chips()
                    .iter()
                    .map(|chip| {
                        let prep_trace = chip.generate_preprocessed_trace(program);
                        // Assert that the chip width data is correct.
                        let expected_width = prep_trace.as_ref().map(|t| t.width()).unwrap_or(0);
                        assert_eq!(
                            expected_width,
                            chip.preprocessed_width(),
                            "Incorrect number of preprocessed columns for chip {}",
                            chip.name()
                        );

                        (chip.name(), prep_trace)
                    })
                    .filter(|(_, prep_trace)| prep_trace.is_some())
                    .map(|(name, prep_trace)| {
                        let prep_trace = prep_trace.unwrap();
                        (name, prep_trace)
                    })
                    .collect::<Vec<_>>()
            });

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_preprocessed_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        let (chip_information, domains_and_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                (
                    (name.to_owned(), domain, trace.dimensions()),
                    (domain, trace.to_device().unwrap().to_column_major()),
                )
            })
            .unzip();

        // Commit to the batch of traces.
        let (commit, data) = tracing::debug_span!("commit to preprocessed traces")
            .in_scope(|| self.committer.commit(&domains_and_traces));
        let data = data.to_host();

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        // Get the preprocessed traces
        let traces = named_preprocessed_traces
            .into_iter()
            .map(|(_, trace)| trace)
            .collect::<Vec<_>>();

        let pc_start = program.pc_start();

        (
            StarkProvingKey {
                commit: commit.clone(),
                pc_start,
                traces,
                data,
                chip_ordering: chip_ordering.clone(),
            },
            StarkVerifyingKey {
                commit,
                pc_start,
                chip_information,
                chip_ordering,
            },
        )
    }

    fn open(
        &self,
        pk: &StarkProvingKey<SC>,
        data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut <SC as StarkGenericConfig>::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error> {
        let span = tracing::Span::current();
        let _span = span.enter();
        let span = tracing::Span::current();
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

        let recompute_ldes = total_lde_size > self.lde_mem_threshold;

        // Delete the ldes of the main prover data.
        if recompute_ldes {
            tracing::debug!("Recomputing LDEs");
            main_data.clear_matrices();
        }

        // Get the permutation challenges.
        let permutation_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<_>>();
        // Generate permutation traces.
        let permutation_traces =
            tracing::debug_span!("generate permutation traces").in_scope(|| {
                self.generate_permutation_traces(pk, &shard_chips, &traces, &permutation_challenges)
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
        challenger.observe(permutation_commit.clone());

        // Get the cumulative sums from device.
        let cumulative_sums = perm_domains_and_traces
            .iter()
            .map(|(_, trace)| {
                let row_idx = trace.height() - 1;
                let start_col_idx =
                    trace.width() - <SC::Challenge as AbstractExtensionField<SC::Val>>::D;
                SC::Challenge::from_base_fn(|i| {
                    let index = (start_col_idx + i) * trace.height() + row_idx;
                    let val = trace.values[index..index + 1].as_host_vec(trace.stream());
                    val[0]
                })
            })
            .collect::<Vec<_>>();
        // Delete the ldes of the permutation prover data.
        if recompute_ldes {
            perm_prover_data.clear_matrices();
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
        let (quotient_commit, quotient_prover_data) = tracing::debug_span!("commit to quotient")
            .in_scope(|| self.committer.commit(&quotient_domains_and_chunks));
        let num_quotient_chunks = quotient_domains_and_chunks.len();
        drop(quotient_domains_and_chunks);
        // Observe the quotient commitment.
        challenger.observe(quotient_commit.clone());

        // Generate the opening proof and assemble the shard proof.

        // Compute the opening challenge.
        let zeta: SC::Challenge = challenger.sample_ext_element();

        let preprocessed_opening_points = pk
            .traces
            .iter()
            .map(|trace| {
                let domain = natural_domain_for_degree(self.machine.config(), trace.height());
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
                perm_prover_data.push_matrix(perm_lde);
            }
            for (domain, trace) in domains.iter().zip(traces) {
                let main_lde = self.committer.encode(*domain, &trace, true)?;
                main_data.push_matrix(main_lde);
            }
        }

        let pk_data_device = pk.data.to_device().unwrap();

        let (openings, opening_proof) = tracing::debug_span!("compute opening").in_scope(|| {
            self.opening_prover.open(
                &self.committer,
                self.pcs(),
                vec![
                    (pk_data_device, preprocessed_opening_points),
                    (main_data, trace_opening_points.clone()),
                    (perm_prover_data, trace_opening_points),
                    (quotient_prover_data, quotient_opening_points),
                ],
                challenger,
            )
        });

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
            quotient_opened_values.push(slice.map(|mut op| op.pop().unwrap()).collect::<Vec<_>>());
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
            challenger.observe(data.main_commit.clone());
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

impl<SC, C, A> StarkGpuProver<SC, C, A>
where
    SC: BabyBearFriConfig,
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
                    .map(|&index| pk.traces[index].to_device().unwrap().to_column_major());

                self.permutation_trace_generator
                    .generate_flattened_permutation_trace(
                        chip,
                        preprocessed_trace.as_ref(),
                        main_trace,
                        random_elements,
                    )
            })
            .collect::<Result<Vec<_>, CudaError>>()
    }
}

#[cfg(test)]
pub mod tests {
    use sp1_core::{
        runtime::{ExecutionRecord, Program, Runtime},
        stark::RiscvAir,
        utils::{
            run_test,
            tests::{FIBONACCI_ELF, SSZ_WITHDRAWALS_ELF},
            SP1CoreOpts,
        },
    };
    use sp1_recursion_core::stark::config::BabyBearPoseidon2Outer;

    use crate::{
        merkle_tree::FieldMerkleTreeDeviceCommitter,
        poseidon2::{baby_bear::DeviceHasherBabyBear, bn254::DeviceHasherBn254},
        utils::init_tracer,
    };

    use super::*;

    pub fn execute_core(program: Program) -> ExecutionRecord {
        let opts = SP1CoreOpts::default();
        let mut runtime = Runtime::new(program, opts);
        runtime.run().unwrap();
        runtime.record
    }

    #[test]
    fn test_fibonacci_poseidon_2_baby_bear_prove() {
        let program = Program::from(FIBONACCI_ELF);

        init_tracer();
        run_test::<StarkGpuProver<_, FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>, _>>(
            program,
        )
        .unwrap();
    }

    #[test]
    fn test_fibonacci_poseidon2_bn254_prove() {
        use sp1_core::io::SP1Stdin;
        use sp1_core::runtime::SP1Context;

        let program = Program::from(FIBONACCI_ELF);

        type SC = BabyBearPoseidon2Outer;

        type P = StarkGpuProver<
            SC,
            FieldMerkleTreeDeviceCommitter<DeviceHasherBn254>,
            RiscvAir<BabyBear>,
        >;

        init_tracer();

        let config = BabyBearPoseidon2Outer::new();

        // Execute the program.
        let runtime = tracing::debug_span!("runtime.run(...)").in_scope(|| {
            let mut runtime = Runtime::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            runtime
        });

        let machine = RiscvAir::machine(config);
        let prover = P::new(machine);
        let inputs = SP1Stdin::new();
        let (pk, vk) = prover.setup(runtime.program.as_ref());
        let (proof, _, _) = sp1_core::utils::prove_with_context(
            &prover,
            &pk,
            Program::clone(&runtime.program),
            &inputs,
            SP1CoreOpts::default(),
            SP1Context::default(),
        )
        .unwrap();

        let mut challenger = prover.config().challenger();
        prover
            .machine()
            .verify(&vk, &proof, &mut challenger)
            .unwrap();
    }

    #[test]
    #[ignore]
    fn test_ssz_withdrawals_prove() {
        let program = Program::from(SSZ_WITHDRAWALS_ELF);

        init_tracer();
        // Execute the program.
        run_test::<StarkGpuProver<_, FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>, _>>(
            program,
        )
        .unwrap();
    }
}
