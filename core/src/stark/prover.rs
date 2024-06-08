use std::{borrow::Borrow, collections::HashMap};

use p3_air::Air;
use p3_challenger::{CanObserve, FieldChallenger};
use rayon::prelude::*;

use itertools::Itertools;

use p3_baby_bear::BabyBear;
use p3_commit::Pcs;
use p3_commit::PolynomialSpace;
use p3_field::AbstractExtensionField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_core::stark::AirOpenedValues;

use sp1_core::stark::ChipOpenedValues;
use sp1_core::stark::ShardCommitment;
use sp1_core::stark::ShardOpenedValues;
use sp1_core::stark::ShardProof;
use sp1_core::{
    air::MachineAir,
    stark::{
        Com, Dom, MachineRecord, PcsProverData, ProverConstraintFolder, StarkGenericConfig,
        StarkMachine, StarkProvingKey, Val,
    },
    utils::BabyBearPoseidon2,
};

use crate::{
    device::{
        error::CudaError,
        memory::{ToDevice, ToHost},
        CudaSync,
    },
    fri::TwoAdicFriPcs,
    matrix::ColMajorMatrixDevice,
    merkle_tree::FieldMerkleTreeGpu,
    poseidon2::poseidon2_bb31_16_kernels::DIGEST_WIDTH,
    stark::{CpuQuotientValuesGenerator, QuotientValues},
    time::CudaInstant,
};

use super::{BabyBearPoseidon2Config, PermutationTraceGenerator};

use super::natural_domain_for_degree;

use super::CpuTraceGenerator;

pub struct FriGpuProver<SC: StarkGenericConfig, A> {
    machine: StarkMachine<SC, A>,
    gpu_pcs: TwoAdicFriPcs<SC::Val, [SC::Val; DIGEST_WIDTH]>,
}

pub struct FriCpuProver<SC: StarkGenericConfig, A> {
    machine: StarkMachine<SC, A>,
}

pub type GpuMatrix<F> = CudaSync<ColMajorMatrixDevice<F>>;

pub type GpuMainTraceData<SC> = MainTraceData<SC, GpuMatrix<Val<SC>>>;

pub type GpuProverData<SC> =
    ProverData<SC, FieldMerkleTreeGpu<Val<SC>, [Val<SC>; DIGEST_WIDTH], GpuMatrix<Val<SC>>>>;

pub type CpuProverData<SC> = ProverData<SC, PcsProverData<SC>>;

pub type CpuMatrix<F> = RowMajorMatrix<F>;

pub type CpuMainTraceData<SC> = MainTraceData<SC, RowMajorMatrix<Val<SC>>>;

pub type GpuMainData<SC> = MainData<
    SC,
    CudaSync<ColMajorMatrixDevice<Val<SC>>>,
    FieldMerkleTreeGpu<Val<SC>, [Val<SC>; DIGEST_WIDTH], GpuMatrix<Val<SC>>>,
>;

pub type CpuMainData<SC> = MainData<SC, RowMajorMatrix<Val<SC>>, PcsProverData<SC>>;

pub struct MainTraceData<SC: StarkGenericConfig, M> {
    pub index: usize,
    pub traces: Vec<M>,
    pub domains: Vec<Dom<SC>>,
    pub chip_ordering: HashMap<String, usize>,
    pub public_values: Vec<SC::Val>,
}

pub struct MainData<SC: StarkGenericConfig, M, Data> {
    pub trace_data: MainTraceData<SC, M>,
    pub prover_data: ProverData<SC, Data>,
}

pub struct ProverData<SC: StarkGenericConfig, Data> {
    pub commit: Com<SC>,
    pub data: Data,
}

impl<SC, A> FriGpuProver<SC, A>
where
    SC: BabyBearPoseidon2Config,
    A: MachineAir<BabyBear>,
    A::Record: Sync,
{
    pub fn new(machine: StarkMachine<SC, A>) -> Self {
        let log_blowup = machine.config().pcs().fri_config().log_blowup;
        Self {
            machine,
            gpu_pcs: TwoAdicFriPcs::new(log_blowup),
        }
    }

    pub fn shard(&self, record: A::Record) -> Vec<A::Record> {
        self.machine
            .shard(record, &<A::Record as MachineRecord>::Config::default())
    }

    pub fn generate_permutation_traces(
        &self,
        pk: &StarkProvingKey<SC>,
        trace_data: &GpuMainTraceData<SC>,
        random_elements: &[SC::Challenge],
    ) -> Result<Vec<GpuMatrix<SC::Val>>, CudaError> {
        let generator = PermutationTraceGenerator::<SC::Val, SC::Challenge, A>::default();

        let shard_chips = self
            .machine
            .shard_chips_ordered(&trace_data.chip_ordering)
            .collect::<Vec<_>>();

        shard_chips
            .par_iter()
            .zip(trace_data.traces.par_iter())
            .map(|(chip, main_trace)| {
                let preprocessed_trace = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&index| pk.traces[index].to_device().to_column_major());

                let flatenned_trace = generator.generate_flattened_permutation_trace(
                    chip,
                    preprocessed_trace.as_ref(),
                    main_trace,
                    random_elements,
                )?;
                CudaSync::new(flatenned_trace)
            })
            .collect::<Result<Vec<_>, CudaError>>()
    }

    pub fn commit<M>(&self, evaluations: &[(Dom<SC>, M)]) -> GpuProverData<SC>
    where
        M: Send + Sync + Borrow<GpuMatrix<SC::Val>>,
    {
        let (commit, data) = self.gpu_pcs.commit(evaluations);
        GpuProverData { commit, data }
    }

    pub fn commit_main_traces(&self, trace_data: &GpuMainTraceData<SC>) -> GpuProverData<SC> {
        let domains_and_traces = trace_data
            .domains
            .iter()
            .copied()
            .zip(trace_data.traces.iter())
            .collect::<Vec<_>>();

        self.commit(&domains_and_traces)
    }

    pub fn commit_main(&self, shard: &A::Record, index: usize) -> GpuMainData<SC> {
        let trace_generator = CpuTraceGenerator::<SC, A>::default();
        let time = std::time::Instant::now();
        let host_trace_data = trace_generator.generate_main_traces(&self.machine, shard, index);
        println!("Device: time to generate main traces: {:?}", time.elapsed());
        // Copy main traces to the device.
        let time = CudaInstant::now().unwrap();
        let trace_data = host_trace_data.to_device();
        println!(
            "Device: time to copy traces to device: {:?}",
            time.elapsed().unwrap()
        );
        let time = CudaInstant::now().unwrap();
        let prover_data = self.commit_main_traces(&trace_data);
        println!(
            "Device: time to commit traces: {:?}",
            time.elapsed().unwrap()
        );
        GpuMainData {
            trace_data,
            prover_data,
        }
    }

    pub fn prove_shard(
        &self,
        pk: &StarkProvingKey<SC>,
        shard_data: GpuMainData<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<ShardProof<SC>, CudaError>
    where
        A: for<'a> Air<ProverConstraintFolder<'a, SC>>,
    {
        let GpuMainData {
            trace_data: main_trace_data,
            prover_data: main_prover_data,
        } = shard_data;

        let time = CudaInstant::now()?;
        // Copy the main trace prover data to the host and drop the device data.
        let host_main_prover_data = main_prover_data.into_host();
        let elapsed = time.elapsed()?;
        println!(
            "Device: time to transfer main prover data from device: {:?}",
            elapsed
        );

        // Get the permutation challenges.
        let permutation_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<_>>();
        // Generate permutation traces.
        let time = CudaInstant::now()?;
        let permutation_traces =
            self.generate_permutation_traces(pk, &main_trace_data, &permutation_challenges)?;
        let elapsed = time.elapsed()?;
        println!("Device: time to generate permutation traces: {:?}", elapsed);

        // Commit to the permutation traces.
        let time = CudaInstant::now()?;
        let perm_domains_and_traces = main_trace_data
            .domains
            .iter()
            .copied()
            .zip(permutation_traces)
            .collect::<Vec<_>>();
        let perm_prover_data = self.commit(&perm_domains_and_traces);
        let elapsed = time.elapsed()?;
        println!("Device: time to commit permutation traces: {:?}", elapsed);

        // Get the cumulative sums from device.
        let cumulative_sums = perm_domains_and_traces
            .iter()
            .map(|(_, trace)| {
                let row_idx = trace.height() - 1;
                let start_col_idx =
                    trace.width() - <SC::Challenge as AbstractExtensionField<SC::Val>>::D - 1;
                SC::Challenge::from_base_fn(|i| {
                    let index = (start_col_idx + i) * trace.height() + row_idx;
                    let val = trace.values[index..index + 1].to_host();
                    val[0]
                })
            })
            .collect::<Vec<_>>();
        // drop the permutation traces.
        drop(perm_domains_and_traces);

        // Observe the permutation commitment.
        challenger.observe(perm_prover_data.commit);

        // Get a challenge for folding the constraints.
        //
        // *Remark*: this is called `alpha` in [sp1_core].
        let folding_challenge: SC::Challenge = challenger.sample_ext_element();

        // Get the permutation prover data from device.
        let time = CudaInstant::now()?;
        let host_perm_prover_data = perm_prover_data.into_host();
        let elapsed = time.elapsed()?;
        println!(
            "Device: time to transfer permutation prover data from device: {:?}",
            elapsed
        );

        // Compute quotient values.
        let shard_chips = self
            .machine
            .shard_chips_ordered(&main_trace_data.chip_ordering)
            .collect::<Vec<_>>();

        // Compute values
        let time = std::time::Instant::now();
        let quotient_generator = CpuQuotientValuesGenerator::<SC, A>::default();
        let quotient_values = shard_chips
            .iter()
            .enumerate()
            .map(|(i, chip)| {
                let preprocessed_index = pk.chip_ordering.get(&chip.name()).copied();
                quotient_generator.generate_quotient_values(
                    self.machine.config(),
                    chip,
                    main_trace_data.domains[i],
                    (preprocessed_index, &pk.data),
                    (i, &host_main_prover_data),
                    (i, &host_perm_prover_data),
                    &permutation_challenges,
                    folding_challenge,
                    &main_trace_data.public_values,
                    cumulative_sums[i],
                )
            })
            .collect::<Vec<_>>();
        let elapsed = time.elapsed();
        println!("Device: time to compute quotient values: {:?}", elapsed);

        // Commit to the quotient values
        let time = CudaInstant::now()?;
        let quotient_domains_and_chunks = quotient_values
            .into_iter()
            .flat_map(|values| {
                let QuotientValues {
                    quotient_chunks,
                    quotient_chunk_domains,
                } = values;

                let quotient_chunks = quotient_chunks
                    .into_iter()
                    .map(|chunk| CudaSync::new(chunk.to_device().to_column_major()));

                quotient_chunk_domains
                    .into_iter()
                    .zip(quotient_chunks)
                    .map(|(domain, result)| result.map(|c| (domain, c)))
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        let quotient_prover_data = self.commit(&quotient_domains_and_chunks);
        let num_quotient_chunks = quotient_domains_and_chunks.len();
        drop(quotient_domains_and_chunks);
        println!(
            "Device: time to commit quotient values: {:?}",
            time.elapsed()?
        );
        // Transfer the quotient data to the host.
        let time = CudaInstant::now()?;
        let host_quotient_prover_data = quotient_prover_data.into_host();
        println!(
            "Device: time to transfer quotient prover data from device: {:?}",
            time.elapsed()?
        );

        // Observe the quotient commitment.
        challenger.observe(host_quotient_prover_data.commit);

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

        let trace_opening_points = main_trace_data
            .domains
            .iter()
            .map(|domain| vec![zeta, domain.next_point(zeta).unwrap()])
            .collect::<Vec<_>>();

        // Compute quotient openning points, open every chunk at zeta.
        let quotient_opening_points = (0..num_quotient_chunks)
            .map(|_| vec![zeta])
            .collect::<Vec<_>>();

        let (openings, opening_proof) = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::open(
            self.machine.config().pcs(),
            vec![
                (&pk.data, preprocessed_opening_points),
                (&host_main_prover_data.data, trace_opening_points.clone()),
                (&host_perm_prover_data.data, trace_opening_points),
                (&host_quotient_prover_data.data, quotient_opening_points),
            ],
            challenger,
        );

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
                    let log_degree = main_trace_data.domains[i].size().ilog2() as usize;
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
                main_commit: host_main_prover_data.commit,
                permutation_commit: host_perm_prover_data.commit,
                quotient_commit: host_quotient_prover_data.commit,
            },
            opened_values: ShardOpenedValues {
                chips: opened_values,
            },
            opening_proof,
            chip_ordering: main_trace_data.chip_ordering,
            public_values: main_trace_data.public_values,
        })
    }
}

impl<SC, A> FriCpuProver<SC, A>
where
    SC: BabyBearPoseidon2Config,
    A: MachineAir<BabyBear>,
    A::Record: Sync,
{
    pub fn new(machine: StarkMachine<SC, A>) -> Self {
        Self { machine }
    }

    pub fn shard(&self, record: A::Record) -> Vec<A::Record> {
        self.machine
            .shard(record, &<A::Record as MachineRecord>::Config::default())
    }

    pub fn generate_main_traces(&self, shard: &A::Record, index: usize) -> CpuMainTraceData<SC> {
        let generator = CpuTraceGenerator::default();
        generator.generate_main_traces(&self.machine, shard, index)
    }

    pub fn generate_permutation_traces(
        &self,
        pk: &StarkProvingKey<SC>,
        trace_data: &CpuMainTraceData<SC>,
        random_elements: &[SC::Challenge],
    ) -> Vec<CpuMatrix<SC::Val>> {
        let shard_chips = self
            .machine
            .shard_chips_ordered(&trace_data.chip_ordering)
            .collect::<Vec<_>>();

        shard_chips
            .par_iter()
            .zip(trace_data.traces.par_iter())
            .map(|(chip, main_trace)| {
                let preprocessed_trace = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&index| &pk.traces[index]);

                chip.generate_permutation_trace(preprocessed_trace, main_trace, random_elements)
                    .flatten_to_base()
            })
            .collect::<Vec<_>>()
    }

    pub fn natural_domain_for_degree(&self, degree: usize) -> Dom<SC> {
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::natural_domain_for_degree(
            self.machine.config().pcs(),
            degree,
        )
    }

    pub fn commit(
        &self,
        evaluations: Vec<(Dom<SC>, RowMajorMatrix<SC::Val>)>,
    ) -> CpuProverData<SC> {
        let (commit, data) = <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::commit(
            self.machine.config().pcs(),
            evaluations,
        );
        ProverData { commit, data }
    }

    pub fn commit_main_traces(&self, trace_data: &CpuMainTraceData<SC>) -> CpuProverData<SC> {
        let domains_and_traces = trace_data
            .domains
            .iter()
            .copied()
            .zip(trace_data.traces.iter().cloned())
            .collect::<Vec<_>>();

        self.commit(domains_and_traces)
    }

    pub fn commit_main(&self, shard: &A::Record, index: usize) -> CpuMainData<SC> {
        let time = std::time::Instant::now();
        let trace_data = self.generate_main_traces(shard, index);
        println!("Host: time to generate traces: {:?}", time.elapsed());
        let time = std::time::Instant::now();
        let prover_data = self.commit_main_traces(&trace_data);
        println!("Host: time to commit traces: {:?}", time.elapsed());
        CpuMainData {
            trace_data,
            prover_data,
        }
    }
}

impl<SC> ToHost for GpuProverData<SC>
where
    SC: StarkGenericConfig<
        Val = BabyBear,
        Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
        Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
    >,
{
    type HostType = CpuProverData<SC>;

    fn to_host(&self) -> Self::HostType {
        CpuProverData {
            commit: self.commit,
            data: self.data.to_host(),
        }
    }
}

impl<SC> ToDevice for CpuMainTraceData<SC>
where
    SC: StarkGenericConfig<
        Val = BabyBear,
        Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
        Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
    >,
{
    type DeviceType = GpuMainTraceData<SC>;

    fn to_device(&self) -> Self::DeviceType {
        GpuMainTraceData {
            index: self.index,
            traces: self
                .traces
                .iter()
                .map(|t| CudaSync::new(t.to_device().to_column_major()).unwrap())
                .collect(),
            domains: self.domains.clone(),
            chip_ordering: self.chip_ordering.clone(),
            public_values: self.public_values.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use sp1_core::{
        runtime::{ExecutionRecord, Program, Runtime},
        stark::{Challenge, RiscvAir, Verifier},
        utils::{tests::FIBONACCI_ELF, SP1CoreOpts},
    };

    use super::*;

    type F = BabyBear;
    type SC = BabyBearPoseidon2;
    type EF = Challenge<SC>;

    pub const TENDERMINT_BENCHMARK_ELF: &[u8] =
        include_bytes!("../../../tendermint_benchmark/elf/riscv32im-succinct-zkvm-elf");

    fn execute_core(program: Program) -> ExecutionRecord {
        let mut runtime = Runtime::new(program, SP1CoreOpts::default());
        runtime.run().unwrap();
        runtime.record
    }

    #[test]
    fn test_commit_main() {
        let program = Program::from(FIBONACCI_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = FriGpuProver::new(machine);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let cpu_prover = FriCpuProver::new(machine);

        // Execute the program.
        let record = execute_core(program);

        let shards = gpu_prover.shard(record);

        for shard in shards {
            let time = std::time::Instant::now();
            let gpu_main_data = gpu_prover.commit_main(&shard, 1);
            println!("Device commit time: {:?}", time.elapsed());

            let time = std::time::Instant::now();
            let cpu_main_data = cpu_prover.commit_main(&shard, 1);
            println!("Host commit time: {:?}", time.elapsed());

            assert_eq!(
                gpu_main_data.prover_data.commit,
                cpu_main_data.prover_data.commit
            );
        }
    }

    #[test]
    fn test_permutation_generation() {
        let program = Program::from(FIBONACCI_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = FriGpuProver::new(machine);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let cpu_prover = FriCpuProver::new(machine);

        let (pk, _) = gpu_prover.machine.setup(&program);

        // Execute the program.
        let record = execute_core(program);

        let shards = gpu_prover.shard(record);

        let mut rng = thread_rng();
        for shard in shards {
            let time = std::time::Instant::now();
            let gpu_main_data = gpu_prover.commit_main(&shard, 1);
            println!("Device commit time: {:?}", time.elapsed());

            let time = std::time::Instant::now();
            let cpu_main_data = cpu_prover.commit_main(&shard, 1);
            println!("Host commit time: {:?}", time.elapsed());

            assert_eq!(
                gpu_main_data.prover_data.commit,
                cpu_main_data.prover_data.commit
            );

            let random_elements: [EF; 2] = rng.gen();

            // Generate the permutation traces and commit to them on Device.
            let time = std::time::Instant::now();
            let gpu_permutation_traces = gpu_prover
                .generate_permutation_traces(&pk, &gpu_main_data.trace_data, &random_elements)
                .unwrap();
            // Commit to the permutation traces.
            let domains_and_traces = gpu_main_data
                .trace_data
                .domains
                .iter()
                .copied()
                .zip(gpu_permutation_traces.iter())
                .collect::<Vec<_>>();
            gpu_prover.commit(&domains_and_traces);
            let elapsed = time.elapsed();
            println!(
                "Device permutation generation and commit time: {:?}",
                elapsed
            );

            // Generate the permutation traces and commit to them on Host.
            let time = std::time::Instant::now();
            let cpu_permutation_traces = cpu_prover.generate_permutation_traces(
                &pk,
                &cpu_main_data.trace_data,
                &random_elements,
            );
            let elapsed = time.elapsed();
            println!("Host permutation generation time: {:?}", elapsed);
            // Commit to the permutation traces.
            let domains_and_traces = cpu_main_data
                .trace_data
                .domains
                .iter()
                .copied()
                .zip(cpu_permutation_traces)
                .collect::<Vec<_>>();
            cpu_prover.commit(domains_and_traces);
            let elapsed = time.elapsed();
            println!("Host permutation generation and commit time: {:?}", elapsed);
        }
    }

    #[test]
    fn test_prove_shard() {
        let program = Program::from(TENDERMINT_BENCHMARK_ELF);

        // let program = Program::from(FIBONACCI_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = FriGpuProver::new(machine);

        let config = SC::default();
        let machine = RiscvAir::machine(config);

        let (pk, vk) = gpu_prover.machine.setup(&program);

        // Execute the program.
        let record = execute_core(program);

        let shards = gpu_prover.shard(record);

        for shard in shards {
            let time = std::time::Instant::now();
            let gpu_main_data = gpu_prover.commit_main(&shard, 1);
            let commit_time = time.elapsed();
            println!("Device commit time: {:?}", commit_time);

            let main_commit = gpu_main_data.prover_data.commit;

            let mut challenger = gpu_prover.machine.config().challenger();
            challenger.observe(main_commit);
            let time = std::time::Instant::now();
            let proof = gpu_prover
                .prove_shard(&pk, gpu_main_data, &mut challenger)
                .unwrap();
            let prove_shard_time = time.elapsed();
            println!("Device prove_shard time: {:?}", prove_shard_time);

            // Print total time:
            let total_time = commit_time + prove_shard_time;
            println!("Device: Total time: {:?}", total_time);

            // Verify the proof.
            let mut challenger = machine.config().challenger();
            challenger.observe(main_commit);
            let shard_chips = machine
                .shard_chips_ordered(&proof.chip_ordering)
                .collect::<Vec<_>>();
            // machine.verify_shard(&vk, proof, &mut challenger).unwrap();
            Verifier::<SC, _>::verify_shard(
                machine.config(),
                &vk,
                &shard_chips,
                &mut challenger,
                &proof,
            )
            .unwrap();
        }
    }
}
