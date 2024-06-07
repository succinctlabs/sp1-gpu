use std::{borrow::Borrow, cmp::Reverse, collections::HashMap};

use p3_challenger::{CanObserve, FieldChallenger};
use rayon::prelude::*;

use p3_baby_bear::BabyBear;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use p3_commit::Pcs;

use sp1_core::{
    air::MachineAir,
    stark::{
        Com, Dom, MachineRecord, PcsProverData, ShardProof, StarkGenericConfig, StarkMachine,
        StarkProvingKey, Val,
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
    time::CudaInstant,
};

use super::PermutationTraceGenerator;

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
    SC: StarkGenericConfig<
        Val = BabyBear,
        Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
        Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
    >,
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

    pub fn generate_main_traces(&self, shard: &A::Record, index: usize) -> GpuMainTraceData<SC> {
        // Filter the chips based on what is used.
        let shard_chips = self.machine.shard_chips(shard).collect::<Vec<_>>();

        // For each chip, generate the trace, copy to the device, and transpose.

        let mut named_traces = shard_chips
            .par_iter()
            .map(|chip| {
                let host_trace = chip.generate_trace(shard, &mut A::Record::default());
                let host_trace = host_trace.to_device().to_column_major();
                let device_trace = CudaSync::new(host_trace).unwrap();
                (chip.name(), device_trace)
            })
            .collect::<Vec<_>>();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        // Get the chip ordering.
        let chip_ordering = named_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect();

        let (domains, traces): (Vec<_>, Vec<_>) = named_traces
            .into_iter()
            .map(|(_, trace)| (self.natural_domain_for_degree(trace.height()), trace))
            .unzip();

        MainTraceData {
            traces,
            domains,
            chip_ordering,
            index,
            public_values: shard.public_values(),
        }
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

    pub fn natural_domain_for_degree(&self, degree: usize) -> Dom<SC> {
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::natural_domain_for_degree(
            self.machine.config().pcs(),
            degree,
        )
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
        let time = CudaInstant::now().unwrap();
        let trace_data = self.generate_main_traces(shard, index);
        println!(
            "Device: time to generate traces: {:?}",
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

    pub fn compute_quotient_values(
        &self,
        pk: &StarkProvingKey<SC>,
        main_data: &CpuProverData<SC>,
        permutation_data: &CpuProverData<SC>,
    ) {
    }

    pub fn prove_shard(
        &self,
        pk: &StarkProvingKey<SC>,
        shard_data: GpuMainData<SC>,
        challenger: &mut SC::Challenger,
    ) -> Result<(), CudaError> {
        // Get the permutation challenges.
        let permutation_challenges = (0..2)
            .map(|_| challenger.sample_ext_element())
            .collect::<Vec<_>>();
        // Generate permutation traces.
        let time = CudaInstant::now()?;
        let permutation_traces =
            self.generate_permutation_traces(pk, &shard_data.trace_data, &permutation_challenges)?;
        let elapsed = time.elapsed()?;
        println!("Device: time to generate permutation traces: {:?}", elapsed);

        // Commit to the permutation traces.
        let time = CudaInstant::now()?;
        let perm_prover_data = {
            let perm_domains_and_traces = shard_data
                .trace_data
                .domains
                .iter()
                .copied()
                .zip(permutation_traces)
                .collect::<Vec<_>>();
            self.commit(&perm_domains_and_traces)
        };
        let elapsed = time.elapsed()?;
        println!("Device: time to commit permutation traces: {:?}", elapsed);

        // Observe the permutation commitment.
        challenger.observe(perm_prover_data.commit);

        // Get a challenge for folding the constraints.
        let alpha: SC::Challenge = challenger.sample_ext_element();

        // Get the prover data from device to host.

        let GpuMainData {
            trace_data,
            prover_data: main_prover_data,
        } = shard_data;

        let ProverData { commit, data } = main_prover_data;

        let time = CudaInstant::now()?;
        let cpu_main_data = CpuProverData::<SC> {
            commit,
            data: data.to_host(),
        };
        let elapsed = time.elapsed()?;
        println!("Host: time to get main prover data: {:?}", elapsed);

        // Compute the quotient values.

        Ok(())
    }
}

impl<SC, A> FriCpuProver<SC, A>
where
    SC: StarkGenericConfig<
        Val = BabyBear,
        Challenge = <BabyBearPoseidon2 as StarkGenericConfig>::Challenge,
        Challenger = <BabyBearPoseidon2 as StarkGenericConfig>::Challenger,
        Pcs = <BabyBearPoseidon2 as StarkGenericConfig>::Pcs,
    >,
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
        // Filter the chips based on what is used.
        let shard_chips = self.machine.shard_chips(shard).collect::<Vec<_>>();

        // For each chip, generate the trace, copy to the device, and transpose.

        let mut named_traces = shard_chips
            .par_iter()
            .map(|chip| {
                let trace = chip.generate_trace(shard, &mut A::Record::default());
                (chip.name(), trace)
            })
            .collect::<Vec<_>>();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        // Get the chip ordering.
        let chip_ordering = named_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect();

        let (domains, traces): (Vec<_>, Vec<_>) = named_traces
            .into_iter()
            .map(|(_, trace)| (self.natural_domain_for_degree(trace.height()), trace))
            .unzip();

        MainTraceData {
            traces,
            domains,
            chip_ordering,
            index,
            public_values: shard.public_values(),
        }
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

    pub fn prove_shard(
        &self,
        pk: &StarkProvingKey<SC>,
        shard_data: CpuMainData<SC>,
        challenger: &mut SC::Challenger,
    ) {
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use sp1_core::{
        runtime::{ExecutionRecord, Program, Runtime},
        stark::{Challenge, RiscvAir},
        utils::{
            tests::{FIBONACCI_ELF, SSZ_WITHDRAWALS_ELF, TENDERMINT_ELF},
            SP1CoreOpts,
        },
    };

    use super::*;

    type F = BabyBear;
    type SC = BabyBearPoseidon2;
    type EF = Challenge<SC>;

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

        let (pk, vk) = gpu_prover.machine.setup(&program);

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
        let program = Program::from(SSZ_WITHDRAWALS_ELF);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let gpu_prover = FriGpuProver::new(machine);

        let config = SC::default();
        let machine = RiscvAir::machine(config);
        let cpu_prover = FriCpuProver::new(machine);

        let (pk, vk) = gpu_prover.machine.setup(&program);

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

            let mut challenger = gpu_prover.machine.config().challenger();
            let time = std::time::Instant::now();
            let _ = gpu_prover
                .prove_shard(&pk, gpu_main_data, &mut challenger)
                .unwrap();
            println!("Device prove time: {:?}", time.elapsed());

            // let mut challenger = gpu_prover.machine.config().challenger();
            // let time = std::time::Instant::now();
            // let cpu_proof = cpu_prover.prove_shard(&pk, cpu_main_data, &mut challenger);
            // println!("Host prove time: {:?}", time.elapsed());
        }
    }
}
