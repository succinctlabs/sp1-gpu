use hashbrown::HashMap;

use rayon::prelude::*;

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Mmcs, PolynomialSpace};
use p3_field::{AbstractExtensionField, TwoAdicField};
use p3_fri::TwoAdicFriPcsProof;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_stark::{
    air::{InteractionScope, MachineAir, MachineProgram},
    AirOpenedValues, Chip, ChipOpenedValues, Com, DebugConstraintBuilder, MachineProof,
    MachineProver, MachineProvingKey, MachineRecord, PcsProverData, ProverConstraintFolder,
    SP1CoreOpts, ShardCommitment, ShardMainData, ShardOpenedValues, ShardProof, StarkGenericConfig,
    StarkMachine, StarkProvingKey, StarkVerifyingKey, Val,
};

use itertools::Itertools;
use tracing::info;

use p3_field::AbstractField;
use std::{
    array,
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
};

use crate::{
    cuda_runtime::{event::CudaEvent, stream::CudaStream},
    device::{
        error::CudaError,
        memory::{ToDevice, ToHost},
        DeviceBuffer,
    },
    fri::{FriOpeningProver, FriQueryProver, TwoAdicFriCommitter},
    matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice},
    merkle_tree::{FieldMerkleTreeGpu, MmcsProverData},
    poseidon2::baby_bear::poseidon2_baby_bear_16_kernels::DIGEST_WIDTH,
    stark::{DeviceQuotientValues, DeviceQuotientValuesGenerator},
    univariate::subgroup_normalizer,
    utils::ChipStatistics,
};

use super::{BabyBearFriConfig, PermutationTraceGenerator};

use super::natural_domain_for_degree;

/// A CUDA prover for a STARK.
pub struct StarkGpuProver<SC: BabyBearFriConfig, C, A> {
    pub(crate) machine: StarkMachine<SC, A>,
    main_stream: CudaStream,
    chip_streams: BTreeMap<String, CudaStream>,
    events: StarkEvents,
    permutation_trace_generator: PermutationTraceGenerator<SC::Val, SC::Challenge, A>,
    quotient_generator: DeviceQuotientValuesGenerator<SC, A>,
    committer: TwoAdicFriCommitter<SC, C>,
    opening_prover: FriOpeningProver<SC>,
}

pub struct StarkEvents {
    preprocessed: BTreeMap<String, CudaEvent>,
    global_main: BTreeMap<String, CudaEvent>,
    local_main: BTreeMap<String, CudaEvent>,
    permutation: BTreeMap<String, CudaEvent>,
    quotient: BTreeMap<String, CudaEvent>,
    batching_buffer_initialization: CudaEvent,
    update_openings: BTreeMap<String, CudaEvent>,
}

impl StarkEvents {
    pub fn new<SC: StarkGenericConfig, A: MachineAir<SC::Val>>(
        machine: &StarkMachine<SC, A>,
    ) -> Result<Self, CudaError> {
        let mut preprocessed = BTreeMap::new();
        let mut global_main = BTreeMap::new();
        let mut local_main = BTreeMap::new();
        let mut permutation = BTreeMap::new();
        let mut quotient = BTreeMap::new();
        let batching_buffer_initialization = CudaEvent::new()?;
        let mut update_openings = BTreeMap::new();

        for chip in machine.chips() {
            if chip.preprocessed_width() > 0 {
                preprocessed.insert(chip.name(), CudaEvent::new()?);
            }
            match chip.commit_scope() {
                InteractionScope::Global => {
                    global_main.insert(chip.name(), CudaEvent::new()?);
                }
                InteractionScope::Local => {
                    local_main.insert(chip.name(), CudaEvent::new()?);
                }
            }
            permutation.insert(chip.name(), CudaEvent::new()?);
            quotient.insert(chip.name(), CudaEvent::new()?);
            update_openings.insert(chip.name(), CudaEvent::new()?);
        }

        Ok(Self {
            preprocessed,
            global_main,
            local_main,
            permutation,
            quotient,
            batching_buffer_initialization,
            update_openings,
        })
    }
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
    fn preprocessed_commit(&self) -> Com<SC> {
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
    A: for<'a> Air<ProverConstraintFolder<'a, SC>>
        + for<'a> Air<air::SymbolicProverFolder<'a>>
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
        let chip_streams = machine
            .chips()
            .iter()
            .map(|chip| (chip.name(), CudaStream::create().unwrap()))
            .collect();
        let domain_normalizers = (0..26).map(subgroup_normalizer).collect::<Vec<_>>();
        let events = StarkEvents::new(&machine).unwrap();
        Self {
            machine,
            main_stream: CudaStream::default(),
            committer: TwoAdicFriCommitter::new(log_blowup),
            permutation_trace_generator: PermutationTraceGenerator::default(),
            opening_prover: FriOpeningProver::new(domain_normalizers),
            quotient_generator,
            chip_streams,
            events,
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
            .filter(|chip| chip.commit_scope() == interaction_scope)
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
        named_traces.sort_by_key(|(name, trace)| (Reverse(trace.height()), name.clone()));

        // Get the chip ordering.
        let chip_ordering =
            named_traces.iter().enumerate().map(|(i, (name, _))| (name.to_owned(), i)).collect();

        // Get the domains.
        let config = self.machine.config();
        let domains: Vec<_> = named_traces
            .iter()
            .map(|(_, trace)| natural_domain_for_degree(config, trace.height()))
            .collect();

        let span = tracing::Span::current();
        let _span = span.enter();
        let span = tracing::Span::current();
        let _span = span.enter();

        let commit_span = tracing::debug_span!("copy traces to device and commit").entered();
        let ((traces, ldes), events): ((Vec<_>, Vec<_>), Vec<_>) = named_traces
            .into_par_iter()
            .zip(domains.par_iter())
            .map(|((name, trace), domain)| {
                let stream = self.chip_streams.get(&name).unwrap();
                let event = self
                    .events
                    .global_main
                    .get(&name)
                    .unwrap_or_else(|| self.events.local_main.get(&name).unwrap())
                    .clone();
                let trace = trace.to_device_async(stream).unwrap().to_column_major();
                let lde = self.committer.encode(*domain, &trace, true).unwrap();
                lde.stream().record(&event).unwrap();
                ((trace, lde), event)
            })
            .collect();

        // Get the committer stream to wait for encodings to be done.
        for event in events.iter() {
            self.main_stream.wait_event(event).unwrap();
        }
        // Commit the LDE evaluations.
        let (commit, data) = self.committer.mmcs_commit(ldes, &self.main_stream);

        commit_span.exit();

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
        named_preprocessed_traces
            .sort_by_key(|(name, trace)| (Reverse(trace.height()), name.clone()));

        let (chip_information, domains_and_traces): (Vec<_>, Vec<_>) = named_preprocessed_traces
            .iter()
            .map(|(name, trace)| {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                let event = self.events.preprocessed.get(name).unwrap().clone();
                (
                    (name.to_owned(), domain, trace.dimensions()),
                    (domain, trace.to_device().unwrap().to_column_major(), event),
                )
            })
            .unzip();

        // Commit to the batch of traces.
        let (commit, data) = tracing::debug_span!("commit to preprocessed traces")
            .in_scope(|| self.committer.commit(&domains_and_traces, &self.main_stream));

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
        let proof = {
            let span = tracing::Span::current();
            let _span = span.enter();

            let setup_span = tracing::debug_span!("process shard data").entered();

            let (global_traces, global_main_commit, mut global_main_data, global_chip_ordering) =
                if let Some(global_data) = global_data {
                    let ShardMainData {
                        traces: global_traces,
                        main_commit: global_main_commit,
                        main_data: global_main_data,
                        chip_ordering: global_chip_ordering,
                        public_values: _,
                    } = global_data;
                    (
                        global_traces,
                        global_main_commit,
                        Some(global_main_data),
                        global_chip_ordering,
                    )
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
            let (all_chips_ordering, _, all_shard_data) = self.merge_shard_traces(
                &global_traces,
                &global_chip_ordering,
                &local_traces,
                &local_chip_ordering,
            );
            let all_traces = all_shard_data.iter().map(|data| data.trace).collect::<Vec<_>>();
            let shard_chips =
                self.machine.shard_chips_ordered(&all_chips_ordering).collect::<Vec<_>>();
            let shard_chip_names = shard_chips.iter().map(|chip| chip.name()).collect::<Vec<_>>();
            let shard_chip_stream =
                all_traces.iter().map(|trace| trace.stream().clone()).collect::<Vec<_>>();

            assert!(shard_chips.len() == all_traces.len());

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

            // Observe the main commitment.
            challenger.observe(local_main_commit.clone());

            setup_span.exit();

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

            info!(
                "Shard: [{}]",
                shard_chips.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ")
            );

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
            let perm_domains_and_traces = domains
                .iter()
                .copied()
                .zip_eq(permutation_traces)
                .zip_eq(shard_chips.iter())
                .map(|((domain, permutation_trace), chip)| {
                    (
                        domain,
                        permutation_trace,
                        self.events.permutation.get(&chip.name()).unwrap().clone(),
                    )
                })
                .collect::<Vec<_>>();
            let (permutation_commit, mut perm_prover_data) =
                self.committer.commit(&perm_domains_and_traces, &self.main_stream);
            permutation_span.exit();

            // Observe the permutation commitment.
            challenger.observe(permutation_commit.clone());
            for sums in cumulative_sums.iter() {
                let global_sum = sums[0];
                let local_sum = sums[1];
                CanObserve::<BabyBear>::observe_slice(challenger, global_sum.as_base_slice());
                CanObserve::<BabyBear>::observe_slice(challenger, local_sum.as_base_slice());
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

            // For each chip, get the quotient domains, evaluations on the quotient domain, and compute
            // the quotient values.

            let permutation_challenges_device = permutation_challenges.to_device().unwrap();
            let public_values_device = local_public_values.to_device().unwrap();

            let mut quotient_values = vec![];

            for (i, chip) in shard_chips.iter().enumerate() {
                let log_quotient_degree = chip.log_quotient_degree();
                let trace = &all_traces[i];
                let trace_domain = domains[i];

                let stream = trace.stream();

                let quotient_domain =
                    trace_domain.create_disjoint_domain(trace_domain.size() << log_quotient_degree);

                let cumulative_sums_device =
                    cumulative_sums[i].as_slice().to_device_async(stream).unwrap();

                // Get the evaluations on the quotient domain. If the LDE evalutions can be used, we
                // just bit-reverse them to match the expected quotient kernel.
                let use_lde = chip.log_quotient_degree() == self.committer.log_blowup;

                let chip_quotient_values = if use_lde {
                    let prep_eval = pk
                        .chip_ordering
                        .get(&chip.name())
                        .map(|&index| {
                            pk.traces[index].to_device_async(stream).unwrap().to_column_major()
                        })
                        .map(|prep_trace| {
                            self.committer.get_evaluations_on_domain(
                                trace_domain,
                                quotient_domain,
                                &prep_trace,
                            )
                        })
                        .transpose()
                        .unwrap();
                    let prep_eval = prep_eval.unwrap_or_else(ColMajorMatrixDevice::null);

                    let main_eval = match chip.commit_scope() {
                        InteractionScope::Local => {
                            &mut local_main_data.matrices_mut()[local_chip_ordering[&chip.name()]]
                        }
                        InteractionScope::Global => {
                            &mut global_main_data.as_mut().unwrap().matrices_mut()
                                [global_chip_ordering[&chip.name()]]
                        }
                    };
                    main_eval.bit_reverse_rows().unwrap();

                    let perm_eval = &mut perm_prover_data.matrices_mut()[i];
                    perm_eval.bit_reverse_rows().unwrap();

                    let quotient_values = self.quotient_generator.compute_values(
                        chip,
                        trace_domain,
                        quotient_domain,
                        &prep_eval,
                        main_eval,
                        perm_eval,
                        &public_values_device,
                        &cumulative_sums_device,
                        folding_challenge,
                        &permutation_challenges_device,
                    );

                    // Since we reversed the lde bits, we need to reverse them back.
                    main_eval.bit_reverse_rows().unwrap();
                    perm_eval.bit_reverse_rows().unwrap();

                    quotient_values
                } else {
                    let prep_eval = pk
                        .chip_ordering
                        .get(&chip.name())
                        .map(|&index| {
                            pk.traces[index].to_device_async(stream).unwrap().to_column_major()
                        })
                        .map(|prep_trace| {
                            self.committer.get_evaluations_on_domain(
                                trace_domain,
                                quotient_domain,
                                &prep_trace,
                            )
                        })
                        .transpose()
                        .unwrap();
                    let prep_eval = prep_eval.unwrap_or_else(ColMajorMatrixDevice::null);

                    let main_eval = self
                        .committer
                        .get_evaluations_on_domain(trace_domain, quotient_domain, all_traces[i])
                        .unwrap();
                    let perm_eval = self
                        .committer
                        .get_evaluations_on_domain(
                            trace_domain,
                            quotient_domain,
                            &perm_domains_and_traces[i].1,
                        )
                        .unwrap();

                    self.quotient_generator.compute_values(
                        chip,
                        trace_domain,
                        quotient_domain,
                        &prep_eval,
                        &main_eval,
                        &perm_eval,
                        &public_values_device,
                        &cumulative_sums_device,
                        folding_challenge,
                        &permutation_challenges_device,
                    )
                }
                .unwrap();

                quotient_values.push(chip_quotient_values);
            }

            // Commit to the quotient values
            let quotient_domains_and_chunks = quotient_values
                .into_iter()
                .zip_eq(shard_chip_names.iter())
                .flat_map(|(values, name)| {
                    let DeviceQuotientValues { quotient_chunks, quotient_chunk_domains } = values;
                    let event = self.events.quotient.get(name).unwrap().clone();

                    quotient_chunk_domains
                        .into_iter()
                        .zip(quotient_chunks)
                        .map(move |(domain, chunk)| (domain, chunk, event.clone()))
                })
                .collect::<Vec<_>>();
            let (quotient_commit, quotient_prover_data) =
                self.committer.commit(&quotient_domains_and_chunks, &self.main_stream);
            quotient_span.exit();
            // Observe the quotient commitment.
            challenger.observe(quotient_commit.clone());

            // Generate the opening proof and assemble the shard proof.

            // Compute the opening challenge.
            let zeta: SC::Challenge = challenger.sample_ext_element();

            // Compute the evaluations of the matrices at the point.
            let compute_evaluations_span = tracing::debug_span!("compute FRI opening").entered();

            // Openings for preprocessed traces.
            let mut preprocessed_opens = vec![];
            let mut input_heights = BTreeSet::new();
            for prep_trace in pk.traces.iter() {
                let trace = prep_trace.to_device().unwrap().to_column_major();
                let domain = natural_domain_for_degree(self.config(), trace.height());
                let local_open = self.opening_prover.eval(domain, &trace, zeta);
                let next_open =
                    self.opening_prover.eval(domain, &trace, domain.next_point(zeta).unwrap());
                input_heights.insert(domain.log_n);
                preprocessed_opens.push((domain.log_n, local_open, next_open));
            }

            // Openings for global main traces (if any).
            let mut main_global_openings = vec![];
            for trace in global_traces {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                let local_open = self.opening_prover.eval(domain, &trace, zeta);
                let next_open =
                    self.opening_prover.eval(domain, &trace, domain.next_point(zeta).unwrap());
                input_heights.insert(domain.log_n);
                main_global_openings.push((domain.log_n, local_open, next_open));
            }

            // Openings for local main traces.
            let mut main_local_openings = vec![];
            for trace in local_traces {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                let local_open = self.opening_prover.eval(domain, &trace, zeta);
                let next_open =
                    self.opening_prover.eval(domain, &trace, domain.next_point(zeta).unwrap());
                input_heights.insert(domain.log_n);
                main_local_openings.push((domain.log_n, local_open, next_open));
            }

            let mut perm_openings = vec![];
            // Openings for permutation traces.
            for (domain, trace, _) in perm_domains_and_traces {
                let local_open = self.opening_prover.eval(domain, &trace, zeta);
                let next_open =
                    self.opening_prover.eval(domain, &trace, domain.next_point(zeta).unwrap());
                input_heights.insert(domain.log_n);
                perm_openings.push((domain.log_n, local_open, next_open));
            }
            // Openings for quotient traces
            let mut quot_openings = vec![];
            for (domain, trace, _) in quotient_domains_and_chunks.into_iter() {
                let open = self.opening_prover.eval(domain, &trace, zeta);
                input_heights.insert(domain.log_n);
                quot_openings.push((domain.log_n, open));
            }

            // Create the input for the FRI opening.

            let log_blowup = self.machine.config().pcs().fri_config().log_blowup;

            let mut batched_openings = input_heights
                .into_iter()
                .map(|trace_log_height| {
                    let log_height = trace_log_height + log_blowup;
                    let mut batched_openings = DeviceBuffer::<SC::Challenge>::with_capacity_in(
                        1 << log_height,
                        &self.main_stream,
                    )
                    .unwrap();
                    unsafe {
                        batched_openings.set_max_len();
                        batched_openings.set(0).unwrap();
                    }
                    (log_height, batched_openings)
                })
                .collect::<BTreeMap<_, _>>();

            let event = &self.events.batching_buffer_initialization;
            self.main_stream.record(event).unwrap();
            for stream in shard_chip_stream.iter() {
                stream.wait_event(event).unwrap();
            }

            // Batch the FRI data.

            // Get the batching challenge.
            let alpha: SC::Challenge = challenger.sample_ext_element();

            // Batch the preprocessed traces
            let mut alpha_offsets = batched_openings
                .keys()
                .map(|i| (*i, SC::Challenge::one()))
                .collect::<BTreeMap<_, _>>();
            for (lde, (log_height, local_open, next_open)) in
                pk.data.matrices().iter().zip_eq(preprocessed_opens.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                let g = BabyBear::two_adic_generator(*log_height);
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    next_open,
                    zeta * g,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
            }

            // Batch the main global traces
            if let Some(global_main_data) = global_main_data.as_ref() {
                for (lde, (log_height, local_open, next_open)) in
                    global_main_data.matrices().iter().zip_eq(main_global_openings.iter())
                {
                    let lde_log_height = log_height + log_blowup;
                    self.opening_prover.batch_update(
                        batched_openings.get_mut(&lde_log_height).unwrap(),
                        lde,
                        SC::Val::generator(),
                        local_open,
                        zeta,
                        alpha,
                        alpha_offsets.get_mut(&lde_log_height).unwrap(),
                    );
                    let g = BabyBear::two_adic_generator(*log_height);
                    self.opening_prover.batch_update(
                        batched_openings.get_mut(&lde_log_height).unwrap(),
                        lde,
                        SC::Val::generator(),
                        next_open,
                        zeta * g,
                        alpha,
                        alpha_offsets.get_mut(&lde_log_height).unwrap(),
                    );
                }
            }

            // Batch the main local traces.
            for (lde, (log_height, local_open, next_open)) in
                local_main_data.matrices().iter().zip_eq(main_local_openings.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                let g = BabyBear::two_adic_generator(*log_height);
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    next_open,
                    zeta * g,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
            }

            // Batch the permutation traces.
            for (lde, (log_height, local_open, next_open)) in
                perm_prover_data.matrices().iter().zip_eq(perm_openings.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                let g = BabyBear::two_adic_generator(*log_height);
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    next_open,
                    zeta * g,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
            }

            // Batch the quotient traces.
            for (lde, (log_height, open)) in
                quotient_prover_data.matrices().iter().zip_eq(quot_openings.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::generator(),
                    open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
            }

            // Wait for all batches to update.
            for (stream, name) in shard_chip_stream.iter().zip(shard_chip_names.iter()) {
                let event = self.events.update_openings.get(name).unwrap();
                stream.record(event).unwrap();
                self.main_stream.wait_event(event).unwrap();
            }

            // generate a fri proof.

            let input_leaves = batched_openings
                .into_iter()
                .map(|(i, values)| {
                    let base_values = unsafe { values.flatten_to_base::<SC::Val>() };
                    let leaf_matrix = RowMajorMatrixDevice::new(
                        base_values,
                        2 * <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
                    )
                    .to_column_major();
                    (i, leaf_matrix)
                })
                .collect::<BTreeMap<_, _>>();

            let (fri_proof, query_indices) = self.opening_prover.prove(
                &self.committer,
                self.machine.config().pcs().fri_config(),
                input_leaves,
                challenger,
            );

            let prover_data = if let Some(global_main_data) = global_main_data.as_ref() {
                vec![
                    &pk.data,
                    global_main_data,
                    &local_main_data,
                    &perm_prover_data,
                    &quotient_prover_data,
                ]
            } else {
                vec![&pk.data, &local_main_data, &perm_prover_data, &quotient_prover_data]
            };
            let log_global_max_height_iter =
                prover_data.iter().flat_map(|data| data.matrices().iter().map(|mat| mat.height));

            let log_global_max_height =
                Iterator::max(log_global_max_height_iter).unwrap().ilog2() as usize;

            let query_openings = self.committer.mmcs_committer.query_open_batch(
                &query_indices,
                &prover_data,
                log_global_max_height,
                false,
            );

            let opening_proof = TwoAdicFriPcsProof { fri_proof, query_openings };

            // Get the openings for the chips.
            let preprocessed_opens = preprocessed_opens
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.to_host()))
                .collect::<Vec<_>>();
            let main_global_openings = main_global_openings
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.to_host()))
                .collect::<Vec<_>>();
            let main_local_openings = main_local_openings
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.to_host()))
                .collect::<Vec<_>>();
            let perm_openings = perm_openings
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.to_host()))
                .collect::<Vec<_>>();
            let mut quot_openings =
                quot_openings.into_iter().map(|(_, open)| open.to_host()).collect::<Vec<_>>();

            let mut opened_values = vec![];
            for (i, (chip, perm_opens)) in shard_chips.iter().zip(perm_openings).enumerate() {
                let preprocessed = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&idx| {
                        let (local, next) = preprocessed_opens[idx].clone();
                        AirOpenedValues { local, next }
                    })
                    .unwrap_or(AirOpenedValues { local: vec![], next: vec![] });

                let global_order = global_chip_ordering.get(&chip.name());
                let local_order = local_chip_ordering.get(&chip.name());
                let main = match (global_order, local_order) {
                    (Some(idx), None) => {
                        let (local, next) = main_global_openings[*idx].clone();
                        AirOpenedValues { local, next }
                    }
                    (None, Some(idx)) => {
                        let (local, next) = main_local_openings[*idx].clone();
                        AirOpenedValues { local, next }
                    }
                    _ => unreachable!(),
                };
                let (perm_local, perm_next) = perm_opens;
                let permutation = AirOpenedValues { local: perm_local, next: perm_next };

                let log_degree = domains[i].size().ilog2() as usize;

                let log_quotient_degree = chip.log_quotient_degree();
                let degree = 1 << log_quotient_degree;
                let quotient = quot_openings.drain(0..degree).collect::<Vec<_>>();

                opened_values.push(ChipOpenedValues {
                    preprocessed,
                    main,
                    permutation,
                    quotient,
                    global_cumulative_sum: cumulative_sums[i][0],
                    local_cumulative_sum: cumulative_sums[i][1],
                    log_degree,
                });
            }

            compute_evaluations_span.exit();

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
                public_values: local_public_values,
            })
        };

        let cleanup_span = tracing::debug_span!("cleanup").entered();
        // Synchronize streams to release all resources.
        for stream in self.chip_streams.values() {
            stream.synchronize().unwrap();
        }
        self.main_stream.synchronize().unwrap();
        cleanup_span.exit();

        proof
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

        self.machine().generate_dependencies(&mut records, &opts, None);

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
    A: for<'a> Air<ProverConstraintFolder<'a, SC>> + MachineAir<BabyBear>,
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

    use sp1_core_executor::{programs::tests::FIBONACCI_ELF, ExecutionRecord, Executor, Program};
    use sp1_core_machine::{riscv::RiscvAir, utils::run_test};
    use sp1_recursion_core::stark::BabyBearPoseidon2Outer;
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
            None,
        )
        .unwrap();

        let mut challenger = prover.config().challenger();
        prover.machine().verify(&vk, &proof, &mut challenger).unwrap();
    }
}
