use hashbrown::HashMap;

use rayon::prelude::*;

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::Mmcs;
use p3_commit::PolynomialSpace;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use sp1_stark::air::InteractionScope;
use sp1_stark::air::MachineAir;
use sp1_stark::air::MachineProgram;
use sp1_stark::AirOpenedValues;
use sp1_stark::Chip;
use sp1_stark::ChipOpenedValues;
use sp1_stark::Com;
use sp1_stark::DebugConstraintBuilder;
use sp1_stark::MachineProof;
use sp1_stark::MachineProver;
use sp1_stark::MachineProvingKey;
use sp1_stark::MachineRecord;
use sp1_stark::PcsProverData;
use sp1_stark::ProverConstraintFolder;
use sp1_stark::SP1CoreOpts;
use sp1_stark::ShardCommitment;
use sp1_stark::ShardMainData;
use sp1_stark::ShardOpenedValues;
use sp1_stark::ShardProof;
use sp1_stark::StarkMachine;
use sp1_stark::StarkProvingKey;
use sp1_stark::StarkVerifyingKey;
use sp1_stark::Val;

use itertools::Itertools;
use tracing::info;

use p3_field::AbstractField;
use std::array;
use std::cmp::Reverse;
use std::marker::PhantomData;

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

const LDE_MEM_RATIO: f64 = 8.0 / 24.0;

/// A CUDA prover for a STARK.
pub struct StarkGpuProver<SC: BabyBearFriConfig, C, A> {
    pub(crate) machine: StarkMachine<SC, A>,
    chip_streams: Vec<CudaStream>,
    permutation_trace_generator: PermutationTraceGenerator<SC::Val, SC::Challenge, A>,
    quotient_generator: DeviceQuotientValuesGenerator<SC, A>,
    lde_mem_threshold: usize,
    committer: TwoAdicFriCommitter<SC, C>,
    opening_prover: FriOpeningProver<SC>,
}

/// A proving key for a STARK.
#[derive(Clone)]
pub struct StarkProvingKeyDevice<SC, C>
where
    SC: BabyBearFriConfig,
    C: FriQueryProver<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>
        + 'static
        + Send
        + Sync
        + Default,
{
    /// The commitment to the preprocessed traces.
    pub commit: Com<SC>,
    /// The start pc of the program.
    pub pc_start: Val<SC>,
    /// The preprocessed traces.
    pub traces: Vec<RowMajorMatrix<Val<SC>>>,
    /// The pcs data for the preprocessed traces.
    pub data: C::ProverData,
    /// The preprocessed chip ordering.
    pub chip_ordering: HashMap<String, usize>,
    pub phantom: PhantomData<C>,
}

impl<SC, C> MachineProvingKey<SC> for StarkProvingKeyDevice<SC, C>
where
    SC: BabyBearFriConfig,
    Com<SC>: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Proof: Send + Sync,
    SC::FriChallenger: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Commitment: Send + Sync,
    SC::RowMajorProverData: Send + Sync,
    C: FriQueryProver<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>
        + 'static
        + Send
        + Sync
        + Default,
    C::ProverData: Send + Sync + ToHost<HostType = PcsProverData<SC>>,
    PcsProverData<SC>: ToDevice<DeviceType = C::ProverData>,
{
    fn commit(&self) -> Com<SC> {
        self.commit.clone()
    }

    fn pc_start(&self) -> Val<SC> {
        self.pc_start
    }

    fn to_host(&self) -> StarkProvingKey<SC> {
        StarkProvingKey {
            commit: self.commit.clone(),
            pc_start: self.pc_start,
            data: self.data.to_host(),
            traces: self.traces.clone(),
            chip_ordering: self.chip_ordering.clone(),
        }
    }

    fn from_host(host: &StarkProvingKey<SC>) -> Self {
        StarkProvingKeyDevice {
            commit: host.commit.clone(),
            pc_start: host.pc_start,
            traces: host.traces.clone(),
            data: host.data.to_device().unwrap(),
            chip_ordering: host.chip_ordering.clone(),
            phantom: PhantomData,
        }
    }

    fn observe_into(&self, challenger: &mut sp1_stark::Challenger<SC>) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        let zero = Val::<SC>::zero();
        for _ in 0..7 {
            challenger.observe(zero);
        }
    }
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
    C::ProverData: Send + Sync + ToHost<HostType = PcsProverData<SC>>,
    PcsProverData<SC>: ToDevice<DeviceType = C::ProverData>,
    Com<SC>: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Proof: Send + Sync,
    SC::FriChallenger: Send,
    <SC::ValMmcs as Mmcs<SC::Val>>::Commitment: Send + Sync,
    SC::RowMajorProverData: Send + Sync,
{
    type DeviceMatrix = ColMajorMatrixDevice<Val<SC>>;
    type DeviceProverData = C::ProverData;
    type DeviceProvingKey = StarkProvingKeyDevice<SC, C>;

    type Error = CudaError;

    fn new(machine: StarkMachine<SC, A>) -> Self {
        let log_blowup = machine.config().pcs().fri_config().log_blowup;
        let quotient_generator = DeviceQuotientValuesGenerator::new(&machine);
        let (_, total) = cuda_mem_get_info().unwrap();
        let lde_mem_threshold = (LDE_MEM_RATIO * (total as f64)) as usize;
        tracing::info!("LDE memory threshold: {}", lde_mem_threshold);
        let chip_streams = machine.chips().iter().map(|_| CudaStream::create().unwrap()).collect();
        Self {
            machine,
            committer: TwoAdicFriCommitter::new(log_blowup),
            permutation_trace_generator: PermutationTraceGenerator::default(),
            opening_prover: FriOpeningProver::default(),
            lde_mem_threshold,
            quotient_generator,
            chip_streams,
        }
    }

    fn machine(&self) -> &StarkMachine<SC, A> {
        &self.machine
    }

    fn generate_traces(
        &self,
        record: &A::Record,
        interaction_scope: InteractionScope,
    ) -> Vec<(String, RowMajorMatrix<Val<SC>>)> {
        let chips = self
            .shard_chips(record)
            .filter(|chip| chip.interaction_randomness() == interaction_scope)
            .collect::<Vec<_>>();

        chips
            .par_iter()
            .map(|chip| {
                let trace = chip.generate_trace(record, &mut A::Record::default());
                (chip.name(), trace)
            })
            .collect::<Vec<_>>()
    }

    fn commit(
        &self,
        shard: &A::Record,
        mut named_traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData> {
        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        // Get the chip ordering.
        let chip_ordering =
            named_traces.iter().enumerate().map(|(i, (name, _))| (name.to_owned(), i)).collect();

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
            .zip(self.chip_streams.iter())
            .map(|(trace, stream)| trace.to_device_async(stream).unwrap().to_column_major())
            .collect();

        // Commit to the traces.
        let domains_and_traces = domains.iter().copied().zip(traces.iter()).collect::<Vec<_>>();
        let (commit, data) =
            tracing::debug_span!("commit").in_scope(|| self.committer.commit(&domains_and_traces));

        tracing::debug_span!("construct main data").in_scope(|| ShardMainData {
            traces,
            main_commit: commit,
            main_data: data,
            chip_ordering,
            public_values: tracing::debug_span!("compute public values")
                .in_scope(|| shard.public_values()),
        })
    }

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>) {
        let mut named_preprocessed_traces = tracing::debug_span!("generate preprocessed traces")
            .in_scope(|| {
                self.machine()
                    .chips()
                    .par_iter()
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

        // Get the chip ordering.
        let chip_ordering = named_preprocessed_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        // Get the preprocessed traces
        let traces =
            named_preprocessed_traces.into_iter().map(|(_, trace)| trace).collect::<Vec<_>>();

        let pc_start = program.pc_start();

        (
            StarkProvingKeyDevice {
                commit: commit.clone(),
                pc_start,
                traces,
                data,
                chip_ordering: chip_ordering.clone(),
                phantom: PhantomData,
            },
            StarkVerifyingKey { commit, pc_start, chip_information, chip_ordering },
        )
    }

    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        global_data: Option<ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>>,
        local_data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut SC::Challenger,
        global_permutation_challenges: &[SC::Challenge],
    ) -> Result<ShardProof<SC>, Self::Error> {
        let span = tracing::Span::current();
        let _span = span.enter();

        let (global_traces, global_main_commit, mut global_main_data, global_chip_ordering) =
            if let Some(global_data) = global_data {
                let ShardMainData {
                    traces: global_traces,
                    main_commit: global_main_commit,
                    main_data: global_main_data,
                    chip_ordering: global_chip_ordering,
                    public_values: _,
                } = global_data;
                (global_traces, global_main_commit, Some(global_main_data), global_chip_ordering)
            } else {
                (vec![], SC::zero_commitment(), None, HashMap::new())
            };

        let ShardMainData {
            traces: local_traces,
            main_commit: local_main_commit,
            main_data: mut local_main_data,
            chip_ordering: local_chip_ordering,
            public_values: local_public_values,
        } = local_data;

        // Merge the chip ordering and traces from the global and local data.
        let (all_chips_ordering, all_chip_scopes, all_shard_data) = self.merge_shard_traces(
            &global_traces,
            &global_chip_ordering,
            &local_traces,
            &local_chip_ordering,
        );
        let all_traces = all_shard_data.iter().map(|data| data.trace).collect::<Vec<_>>();
        let shard_chips = self.machine.shard_chips_ordered(&all_chips_ordering).collect::<Vec<_>>();

        assert!(shard_chips.len() == all_shard_data.len());

        let domains = all_traces
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

            if let Some(global_main_data) = global_main_data.as_mut() {
                global_main_data.clear_matrices();
            }
            local_main_data.clear_matrices();
        }

        // Observe the main commitment.
        challenger.observe(local_main_commit.clone());

        // Get the permutation challenges.
        let local_permutation_challenges =
            (0..2).map(|_| challenger.sample_ext_element()).collect::<Vec<_>>();
        let permutation_challenges = global_permutation_challenges
            .iter()
            .chain(local_permutation_challenges.iter())
            .copied()
            .collect::<Vec<_>>();

        // Generate permutation traces.
        let permutation_span =
            tracing::debug_span!("generate and commit to permutation traces").entered();
        let permutation_traces_and_cumulative_sums = self.generate_permutation_traces(
            pk,
            &shard_chips,
            &all_traces,
            &permutation_challenges,
        )?;
        let (permutation_traces, cumulative_sums): (Vec<_>, Vec<_>) =
            permutation_traces_and_cumulative_sums.into_iter().unzip();

        info!("Shard: [{}]", shard_chips.iter().map(|c| c.name()).collect::<Vec<_>>().join(", "));

        for (i, chip) in shard_chips.iter().enumerate() {
            let width = all_traces[i].width();
            let height = all_traces[i].height();
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
        let perm_domains_and_traces =
            domains.iter().copied().zip(permutation_traces).collect::<Vec<_>>();
        let (permutation_commit, mut perm_prover_data) =
            self.committer.commit(&perm_domains_and_traces);
        permutation_span.exit();

        // Observe the permutation commitment.
        challenger.observe(permutation_commit.clone());

        // Delete the ldes of the permutation prover data.
        if recompute_ldes {
            perm_prover_data.clear_matrices();
        }

        // Get a challenge for folding the constraints.
        //
        // *Remark*: this is called `alpha` in [sp1_core].
        let folding_challenge: SC::Challenge = challenger.sample_ext_element();

        // Compute quotient values.

        let quotient_span =
            tracing::debug_span!("generate and commit to quotient values").entered();

        let permutation_challenges = global_permutation_challenges
            .iter()
            .chain(local_permutation_challenges.iter())
            .copied()
            .collect::<Vec<_>>();

        // Compute values
        let quotient_values = self.quotient_generator.generate_quotient_values(
            &self.committer,
            &shard_chips,
            pk,
            &all_traces,
            &perm_domains_and_traces,
            &permutation_challenges,
            folding_challenge,
            &local_public_values,
            &cumulative_sums,
        )?;

        // Commit to the quotient values
        let quotient_domains_and_chunks = quotient_values
            .into_iter()
            .flat_map(|values| {
                let DeviceQuotientValues { quotient_chunks, quotient_chunk_domains } = values;

                quotient_chunk_domains.into_iter().zip(quotient_chunks)
            })
            .collect::<Vec<_>>();
        let (quotient_commit, quotient_prover_data) =
            self.committer.commit(&quotient_domains_and_chunks);
        let num_quotient_chunks = quotient_domains_and_chunks.len();
        drop(quotient_domains_and_chunks);
        quotient_span.exit();
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
        let quotient_opening_points =
            (0..num_quotient_chunks).map(|_| vec![zeta]).collect::<Vec<_>>();

        // Recompute main and permutation LDE and insert into the prover data.
        if recompute_ldes {
            for (domain, perm_trace) in perm_domains_and_traces {
                let perm_lde = self.committer.encode(domain, &perm_trace, true)?;
                perm_prover_data.push_matrix(perm_lde);
            }
            for (i, (domain, trace)) in domains.iter().zip(all_traces).enumerate() {
                let main_lde = self.committer.encode(*domain, trace, true)?;
                let scope = all_chip_scopes[i];

                if scope == InteractionScope::Global {
                    if let Some(global_main_data) = global_main_data.as_mut() {
                        global_main_data.push_matrix(main_lde);
                    }
                } else {
                    local_main_data.push_matrix(main_lde);
                }
            }
        }

        // Split the trace_opening_points to the global and local chips.
        let mut global_trace_opening_points = Vec::with_capacity(global_chip_ordering.len());
        let mut local_trace_opening_points = Vec::with_capacity(local_chip_ordering.len());
        for (i, trace_opening_point) in trace_opening_points.clone().into_iter().enumerate() {
            let scope = all_chip_scopes[i];
            if scope == InteractionScope::Global {
                global_trace_opening_points.push(trace_opening_point);
            } else {
                local_trace_opening_points.push(trace_opening_point);
            }
        }

        let rounds = if let Some(global_main_data) = global_main_data.as_ref() {
            vec![
                (&pk.data, preprocessed_opening_points),
                (global_main_data, global_trace_opening_points),
                (&local_main_data, local_trace_opening_points),
                (&perm_prover_data, trace_opening_points),
                (&quotient_prover_data, quotient_opening_points),
            ]
        } else {
            vec![
                (&pk.data, preprocessed_opening_points),
                (&local_main_data, local_trace_opening_points),
                (&perm_prover_data, trace_opening_points),
                (&quotient_prover_data, quotient_opening_points),
            ]
        };

        let (openings, opening_proof) = tracing::debug_span!("compute opening")
            .in_scope(|| self.opening_prover.open(&self.committer, self.pcs(), rounds, challenger));

        // Collect the opened values for each chip.
        let (
            preprocessed_values,
            global_main_values,
            local_main_values,
            permutation_values,
            mut quotient_values,
        ) = if global_main_data.is_some() {
            let [preprocessed_values, global_main_values, local_main_values, permutation_values, quotient_values] =
                openings.try_into().unwrap();
            (
                preprocessed_values,
                Some(global_main_values),
                local_main_values,
                permutation_values,
                quotient_values,
            )
        } else {
            let [preprocessed_values, local_main_values, permutation_values, quotient_values] =
                openings.try_into().unwrap();
            (preprocessed_values, None, local_main_values, permutation_values, quotient_values)
        };

        let preprocessed_opened_values = preprocessed_values
            .into_iter()
            .map(|op| {
                let [local, next] = op.try_into().unwrap();
                AirOpenedValues { local, next }
            })
            .collect::<Vec<_>>();

        // Merge the global and local main values.
        let mut main_values =
            Vec::with_capacity(global_chip_ordering.len() + local_chip_ordering.len());
        for chip in shard_chips.iter() {
            let global_order = global_chip_ordering.get(&chip.name());
            let local_order = local_chip_ordering.get(&chip.name());
            match (global_order, local_order) {
                (Some(&global_order), None) => {
                    let global_main_values =
                        global_main_values.as_ref().expect("Global main values should be Some");
                    main_values.push(global_main_values[global_order].clone());
                }
                (None, Some(&local_order)) => {
                    main_values.push(local_main_values[local_order].clone());
                }
                _ => unreachable!(),
            }
        }
        assert!(main_values.len() == shard_chips.len());

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
            .map(|(i, ((((main, permutation), quotient), cumulative_sum), chip))| {
                let preprocessed = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&index| preprocessed_opened_values[index].clone())
                    .unwrap_or(AirOpenedValues { local: vec![], next: vec![] });
                let log_degree = domains[i].size().ilog2() as usize;
                ChipOpenedValues {
                    preprocessed,
                    main,
                    permutation,
                    quotient,
                    global_cumulative_sum: cumulative_sum[0],
                    local_cumulative_sum: cumulative_sum[1],
                    log_degree,
                }
            })
            .collect::<Vec<_>>();

        Ok(ShardProof::<SC> {
            commitment: ShardCommitment {
                global_main_commit,
                local_main_commit,
                permutation_commit,
                quotient_commit,
            },
            opened_values: ShardOpenedValues { chips: opened_values },
            opening_proof,
            chip_ordering: all_chips_ordering,
            chip_scopes: all_chip_scopes,
            public_values: local_public_values,
        })
    }

    /// Prove the execution record is valid.
    ///
    /// Given a proving key `pk` and a matching execution record `record`, this function generates
    /// a STARK proof that the execution record is valid.
    fn prove(
        &self,
        pk: &Self::DeviceProvingKey,
        mut records: Vec<A::Record>,
        challenger: &mut SC::Challenger,
        opts: <A::Record as MachineRecord>::Config,
    ) -> Result<MachineProof<SC>, Self::Error>
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, Val<SC>, SC::Challenge>>,
    {
        let contains_global_bus = self.machine().contains_global_bus();

        self.machine().generate_dependencies(&mut records, &opts, InteractionScope::Global);
        self.machine().generate_dependencies(&mut records, &opts, InteractionScope::Local);

        // Observe the preprocessed commitment.
        challenger.observe(pk.commit.clone());
        challenger.observe(pk.pc_start);
        let zero = Val::<SC>::zero();
        for _ in 0..7 {
            challenger.observe(zero);
        }

        // Generate and commit the traces for each shard.
        let global_data: Vec<_> = records
            .iter()
            .map(|record| {
                if contains_global_bus {
                    let global_named_traces =
                        self.generate_traces(record, InteractionScope::Global);
                    Some(self.commit(record, global_named_traces))
                } else {
                    None
                }
            })
            .collect();

        // Observe the challenges for each segment.
        global_data.iter().zip_eq(records.iter()).for_each(|(global_data, record)| {
            if contains_global_bus {
                challenger.observe(
                    global_data
                        .as_ref()
                        .expect("must have a global commitment")
                        .main_commit
                        .clone(),
                );
            }
            challenger.observe_slice(&record.public_values::<SC::Val>()[0..self.num_pv_elts()]);
        });

        // Obtain the challenges used for the global permutation argument.
        let global_permutation_challenges: [SC::Challenge; 2] = array::from_fn(|_| {
            if contains_global_bus {
                challenger.sample_ext_element()
            } else {
                SC::Challenge::zero()
            }
        });

        let shard_proofs = records
            .iter()
            .zip(global_data.into_iter())
            .map(|(record, global_shard_data)| {
                let traces = self.generate_traces(record, InteractionScope::Local);
                let local_shard_data = self.commit(record, traces);

                let span = tracing::Span::current();
                let _span = span.enter();
                self.open(
                    pk,
                    global_shard_data,
                    local_shard_data,
                    &mut challenger.clone(),
                    &global_permutation_challenges,
                )
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
    C: FriQueryProver<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>
        + 'static
        + Send
        + Sync
        + Default,
{
    pub fn pcs(&self) -> &SC::Pcs {
        self.machine.config().pcs()
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_permutation_traces(
        &self,
        pk: &StarkProvingKeyDevice<SC, C>,
        chips: &[&Chip<SC::Val, A>],
        main_traces: &[&GpuMatrix<SC::Val>],
        random_elements: &[SC::Challenge],
    ) -> Result<Vec<(GpuMatrix<SC::Val>, Vec<SC::Challenge>)>, CudaError> {
        chips
            .iter()
            .zip(main_traces.iter())
            .map(|(chip, main_trace)| {
                let preprocessed_trace = pk.chip_ordering.get(&chip.name()).map(|&index| {
                    pk.traces[index].to_device_async(main_trace.stream()).unwrap().to_column_major()
                });

                self.permutation_trace_generator.generate_flattened_permutation_trace(
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

    use sp1_core_executor::programs::tests::FIBONACCI_ELF;
    use sp1_core_executor::ExecutionRecord;
    use sp1_core_executor::Executor;
    use sp1_core_executor::Program;
    use sp1_core_machine::riscv::RiscvAir;
    use sp1_core_machine::utils::run_test;
    use sp1_recursion_core::stark::config::BabyBearPoseidon2Outer;
    use sp1_stark::StarkGenericConfig;

    use crate::{
        merkle_tree::FieldMerkleTreeDeviceCommitter,
        poseidon2::{baby_bear::DeviceHasherBabyBear, bn254::DeviceHasherBn254},
        utils::init_tracer,
    };

    use super::*;

    pub fn execute_core(program: Program) -> ExecutionRecord {
        let opts = SP1CoreOpts::default();
        let mut runtime = Executor::new(program, opts);
        runtime.run().unwrap();
        runtime.record
    }

    #[test]
    fn test_fibonacci_poseidon_2_baby_bear_prove() {
        let program = Program::from(FIBONACCI_ELF).unwrap();

        init_tracer();
        run_test::<StarkGpuProver<_, FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>, _>>(
            program,
        )
        .unwrap();
    }

    #[test]
    fn test_fibonacci_poseidon2_bn254_prove() {
        use sp1_core_executor::SP1Context;
        use sp1_core_machine::io::SP1Stdin;

        let program = Program::from(FIBONACCI_ELF).unwrap();

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
            let mut runtime = Executor::new(program, SP1CoreOpts::default());
            runtime.run().unwrap();
            runtime
        });

        let machine = RiscvAir::machine(config);
        let prover = P::new(machine);
        let inputs = SP1Stdin::new();
        let (pk, vk) = prover.setup(runtime.program.as_ref());
        let (proof, _, _) = sp1_core_machine::utils::prove_with_context(
            &prover,
            &pk,
            Program::clone(&runtime.program),
            &inputs,
            SP1CoreOpts::default(),
            SP1Context::default(),
        )
        .unwrap();

        let mut challenger = prover.config().challenger();
        prover.machine().verify(&vk, &proof, &mut challenger).unwrap();
    }
}
