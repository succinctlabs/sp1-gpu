use hashbrown::HashMap;

use p3_air::Air;
use p3_baby_bear::BabyBear;
use p3_challenger::{CanObserve, FieldChallenger};
use p3_commit::{Mmcs, PolynomialSpace};
use p3_field::{extension::BinomialExtensionField, Field, FieldExtensionAlgebra, TwoAdicField};
use p3_fri::TwoAdicFriPcsProof;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{get_symbolic_constraints, SymbolicAirBuilder};
use rayon::prelude::*;
use sp1_stark::air::PublicValues;
use sp1_stark::septic_curve::SepticCurve;
use sp1_stark::septic_digest::SepticDigest;
use sp1_stark::septic_extension::SepticExtension;
use sp1_stark::MachineChip;
use sp1_stark::Word;
use sp1_stark::{
    air::{InteractionScope, MachineAir, MachineProgram},
    count_permutation_constraints, AirOpenedValues, Chip, ChipOpenedValues, Com,
    DebugConstraintBuilder, MachineProof, MachineProver, MachineProvingKey, MachineRecord,
    PcsProverData, ProverConstraintFolder, SP1CoreOpts, ShardCommitment, ShardMainData,
    ShardOpenedValues, ShardProof, StarkGenericConfig, StarkMachine, StarkVerifyingKey, Val,
    PROOF_MAX_NUM_PVS,
};
use std::borrow::BorrowMut;

use itertools::Itertools;
use tracing::info;

use p3_field::FieldAlgebra;
use std::{
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
    tracegen::{DeviceAir, DevicePreprocessedAir},
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
    preprocessed_trace_to_device: BTreeMap<String, CudaEvent>,
    pk_to_device: CudaEvent,
    quotient_common_data_to_device: CudaEvent,
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
        let mut preprocessed_trace_to_device = BTreeMap::new();

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
            preprocessed_trace_to_device.insert(chip.name(), CudaEvent::new()?);
        }

        Ok(Self {
            preprocessed,
            global_main,
            local_main,
            permutation,
            quotient,
            batching_buffer_initialization,
            update_openings,
            preprocessed_trace_to_device,
            pk_to_device: CudaEvent::new()?,
            quotient_common_data_to_device: CudaEvent::new()?,
        })
    }
}

/// A proving key for a STARK.
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
    /// The initial global cumulative sum.
    pub initial_global_cumulative_sum: SepticDigest<Val<SC>>,
    /// The preprocessed traces.
    pub traces: Vec<ColMajorMatrixDevice<Val<SC>>>,
    /// The pcs data for the preprocessed traces.
    pub data: C::ProverData,
    /// The preprocessed chip ordering.
    pub chip_ordering: HashMap<String, usize>,
    /// The preprocessed chip local only information.
    pub local_only: Vec<bool>,
    /// The number of total constraints for each chip.
    pub constraints_map: HashMap<String, usize>,
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

    fn initial_global_cumulative_sum(&self) -> SepticDigest<Val<SC>> {
        self.initial_global_cumulative_sum
    }

    fn observe_into(&self, challenger: &mut sp1_stark::Challenger<SC>) {
        challenger.observe(self.commit.clone());
        challenger.observe(self.pc_start);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(&self.initial_global_cumulative_sum.0.y.0);
        let zero = Val::<SC>::ZERO;
        challenger.observe(zero);
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
        + MachineAir<BabyBear>
        + DeviceAir<BabyBear>
        + DevicePreprocessedAir<BabyBear>
        + for<'a> Air<SymbolicAirBuilder<BabyBear>>,
    // + for<'a> Air<DebugConstraintBuilder<'a, SC::Val, SC::Challenge>>,
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
        unsafe { crate::cuda_runtime::ffi::cuda_setup_mem_pool() };
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
            main_stream: CudaStream::create().unwrap(),
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

    fn pk_to_device(&self, pk: &sp1_stark::StarkProvingKey<SC>) -> Self::DeviceProvingKey {
        let chip_ordering = pk.chip_ordering.clone();
        let mut data = pk.data.to_device_async(&self.main_stream).unwrap();
        self.main_stream.record(&self.events.pk_to_device).unwrap();
        let mut traces = Vec::with_capacity(chip_ordering.len());

        for i in 0..chip_ordering.len() {
            let name =
                chip_ordering.iter().find(|(_, idx)| **idx == i).map(|(name, _)| name).unwrap();
            let stream = self.chip_streams.get(name).unwrap();
            stream.wait_event(&self.events.pk_to_device).unwrap();
            // Update lde stream.
            let lde = &mut data.matrices_mut()[i];
            unsafe {
                lde.values.set_stream(stream.clone());
            }
            let trace = pk.traces[i].to_device_async(stream).unwrap().to_column_major();
            let event = self.events.preprocessed_trace_to_device.get(name).unwrap();
            stream.record(event).unwrap();
            self.main_stream.wait_event(event).unwrap();
            traces.push(trace);
        }

        StarkProvingKeyDevice {
            commit: pk.commit.clone(),
            pc_start: pk.pc_start,
            initial_global_cumulative_sum: pk.initial_global_cumulative_sum,
            traces,
            data,
            chip_ordering,
            local_only: pk.local_only.clone(),
            constraints_map: pk.constraints_map.clone(),
            phantom: PhantomData,
        }
    }

    fn pk_to_host(&self, pk: &Self::DeviceProvingKey) -> sp1_stark::StarkProvingKey<SC> {
        sp1_stark::StarkProvingKey {
            commit: pk.commit.clone(),
            pc_start: pk.pc_start,
            initial_global_cumulative_sum: pk.initial_global_cumulative_sum,
            data: pk.data.to_host(),
            traces: pk.traces.iter().map(|t| t.to_host()).collect(),
            chip_ordering: pk.chip_ordering.clone(),
            local_only: pk.local_only.clone(),
            constraints_map: pk.constraints_map.clone(),
        }
    }

    fn generate_traces(&self, record: &mut A::Record) -> Vec<(String, RowMajorMatrix<Val<SC>>)> {
        let chips = self.shard_chips(record).collect::<Vec<_>>();

        chips
            .par_iter()
            .filter_map(|chip| {
                let trace = chip.air.generate_trace_host(record, &mut A::Record::default())?;
                Some((chip.name(), trace))
            })
            .collect::<Vec<_>>()
    }

    fn commit(
        &self,
        shard: &A::Record,
        named_traces: Vec<(String, RowMajorMatrix<Val<SC>>)>,
    ) -> ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData> {
        /// A possibly finished trace generation job.
        ///
        /// Defined here because otherwise it would be several hundred lines away, and nobody
        /// else needs to worry about it. It's merely a "this or that" type.
        enum TraceGenerationJob<'b, SC, A>
        where
            SC: BabyBearFriConfig,
        {
            /// A finished trace generated on the host, along with its name.
            Host(String, RowMajorMatrix<Val<SC>>),
            /// A chip that needs trace generation on device, along with its height.
            Device(&'b MachineChip<SC, A>, usize),
        }

        impl<'b, SC, A> TraceGenerationJob<'b, SC, A>
        where
            SC: BabyBearFriConfig,
            A: MachineAir<Val<SC>>,
        {
            #[inline]
            fn name(&self) -> String {
                match self {
                    TraceGenerationJob::Host(name, _) => name.clone(),
                    TraceGenerationJob::Device(chip, _) => chip.name(),
                }
            }

            #[inline]
            fn height(&self) -> usize {
                match self {
                    TraceGenerationJob::Host(_, mat) => mat.height(),
                    TraceGenerationJob::Device(_, height) => *height,
                }
            }
        }

        // Get the chips that need to be used for this shard.
        let chips = self.shard_chips(shard).collect::<Vec<_>>();

        // Create the trace jobs.
        let mut trace_jobs: Vec<TraceGenerationJob<'_, SC, A>> = named_traces
            .into_iter()
            .map(|(name, mat)| TraceGenerationJob::Host(name, mat))
            .chain(chips.clone().into_iter().filter_map(|chip| {
                Some(TraceGenerationJob::Device(chip, chip.air.num_rows_device(shard)?))
            }))
            .collect();

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        trace_jobs.sort_by_key(|job| (Reverse(job.height()), job.name()));

        // Get the chip ordering.
        let chip_ordering = trace_jobs.iter().enumerate().map(|(i, job)| (job.name(), i)).collect();

        // Get the domains.
        let config = self.machine.config();
        let domains = trace_jobs
            .iter()
            .map(|job| natural_domain_for_degree(config, job.height()))
            .collect::<Vec<_>>();

        let span = tracing::Span::current();
        let _span = span.enter();

        let chip_streams = trace_jobs
            .iter()
            .map(|job| self.chip_streams.get(&job.name()).unwrap())
            .collect::<Vec<_>>();
        let events = trace_jobs
            .iter()
            .map(|job| {
                self.events
                    .global_main
                    .get(&job.name())
                    .unwrap_or_else(|| self.events.local_main.get(&job.name()).unwrap())
                    .clone()
            })
            .collect::<Vec<_>>();

        let traces: Vec<Self::DeviceMatrix> = tracing::debug_span!("generate trace accel")
            .in_scope(|| {
                let span = tracing::Span::current();
                trace_jobs
                    .par_iter()
                    .zip(chip_streams)
                    .map(|(job, stream)| {
                        let _span = span.enter();
                        match job {
                            TraceGenerationJob::Host(name, mat) => {
                                tracing::debug_span!("copy host trace to device", chip = name)
                                    .in_scope(|| {
                                        mat.to_device_async(stream).unwrap().to_column_major()
                                    })
                            }
                            TraceGenerationJob::Device(chip, _) => {
                                tracing::debug_span!("generate trace on device", chip = chip.name())
                                    .in_scope(|| {
                                        chip.air
                                            .generate_trace_device(
                                                shard,
                                                &mut A::Record::default(),
                                                stream,
                                            )
                                            .unwrap()
                                            .unwrap()
                                    })
                            }
                        }
                    })
                    .collect()
            });

        let global_index = trace_jobs.iter().position(|job| job.name() == "Global");

        let public_values: &mut [Val<SC>] = &mut shard.public_values();
        if let Some(global_index) = global_index {
            let main_trace = &traces[global_index];
            let x = SepticExtension::<BabyBear>::from_base_fn(|i| {
                let index =
                    (main_trace.width() - 14 + i) * main_trace.height() + main_trace.height() - 1;
                let val = main_trace.values[index..index + 1].as_host_vec(main_trace.stream());
                val[0]
            });

            let y = SepticExtension::<BabyBear>::from_base_fn(|i| {
                let index =
                    (main_trace.width() - 7 + i) * main_trace.height() + main_trace.height() - 1;
                let val = main_trace.values[index..index + 1].as_host_vec(main_trace.stream());
                val[0]
            });

            let pv: &mut PublicValues<Word<Val<SC>>, Val<SC>> = public_values.borrow_mut();
            pv.global_cumulative_sum = SepticDigest(SepticCurve { x, y });
        }

        // Commit to the traces.
        let domains_and_traces = domains
            .iter()
            .copied()
            .zip(traces.iter())
            .zip(events)
            .map(|((domain, trace), event)| (domain, trace, event))
            .collect::<Vec<_>>();
        let (commit, data) = tracing::debug_span!("commit")
            .in_scope(|| self.committer.commit(domains_and_traces.as_slice(), &self.main_stream));

        tracing::debug_span!("construct main data").in_scope(|| ShardMainData {
            traces,
            main_commit: commit,
            main_data: data,
            chip_ordering,
            public_values: public_values.to_vec(),
        })
    }

    /// Setup the preprocessed data into a proving and verifying key.
    fn setup(&self, program: &A::Program) -> (Self::DeviceProvingKey, StarkVerifyingKey<SC>) {
        let generate_traces_copy_span =
            tracing::debug_span!("generate preprocessed traces and copy to device").entered();

        let (named_preprocessed_data, num_constraints): (Vec<_>, Vec<_>) = self
            .machine()
            .chips()
            .par_iter()
            .map(|chip| {
                let prep_trace = chip.air.generate_preprocessed_trace_host(program);

                let name = chip.name();
                let stream = self.chip_streams.get(&name).unwrap().clone();

                let trace = match prep_trace {
                    Some(trace) => Some(trace.to_device_async(&stream).unwrap().to_column_major()),
                    None => chip.air.generate_preprocessed_trace_device(program, &stream).unwrap(),
                };

                // Assert that the chip width data is correct.
                let expected_width = trace.as_ref().map(|t| t.width()).unwrap_or(0);
                assert_eq!(
                    expected_width,
                    chip.preprocessed_width(),
                    "Incorrect number of preprocessed columns for chip {}",
                    chip.name()
                );

                // Count the number of constraints.
                let num_main_constraints = get_symbolic_constraints(
                    &chip.air,
                    chip.preprocessed_width(),
                    PROOF_MAX_NUM_PVS,
                )
                .len();

                let num_permutation_constraints = count_permutation_constraints(
                    &chip.sends,
                    &chip.receives,
                    chip.logup_batch_size(),
                    chip.air.commit_scope(),
                );
                (
                    (chip.name(), chip.local_only(), trace),
                    (name, num_main_constraints + num_permutation_constraints),
                )
            })
            .unzip();

        let mut named_preprocessed_data = named_preprocessed_data
            .into_iter()
            .filter(|(_, _, trace)| trace.is_some())
            .map(|(name, local_only, trace)| {
                let trace = trace.unwrap();
                let event = self.events.preprocessed.get(&name).unwrap().clone();
                let domain = natural_domain_for_degree(self.config(), trace.height());
                let dimensions = trace.dimensions();

                (name, domain, event, local_only, trace, dimensions)
            })
            .collect::<Vec<_>>();

        named_preprocessed_data
            .sort_by_key(|(name, domain, _, _, _, _)| (Reverse(domain.size()), name.clone()));

        let ((chip_information, commitment_data), local_only): ((Vec<_>, Vec<_>), Vec<_>) =
            named_preprocessed_data
                .into_iter()
                .map(|(name, domain, event, local_only, trace, dimensions)| {
                    (((name, domain, dimensions), (domain, trace, event)), local_only)
                })
                .collect();

        generate_traces_copy_span.exit();

        // Commit to the batch of traces.
        let commit_span = tracing::debug_span!("commit to preprocessed traces").entered();
        let (commit, data) = self.committer.commit(&commitment_data, &self.main_stream);

        commit_span.exit();

        // // Get the chip ordering.
        let chip_ordering = chip_information
            .iter()
            .enumerate()
            .map(|(i, (name, _, _))| (name.to_owned(), i))
            .collect::<HashMap<_, _>>();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces = commitment_data.into_iter().map(|(_, trace, _)| trace).collect::<Vec<_>>();

        let pc_start = program.pc_start();
        let initial_global_cumulative_sum = program.initial_global_cumulative_sum();

        (
            StarkProvingKeyDevice {
                commit: commit.clone(),
                pc_start,
                initial_global_cumulative_sum,
                traces,
                data,
                chip_ordering: chip_ordering.clone(),
                local_only: local_only.clone(),
                constraints_map: constraints_map.clone(),
                phantom: PhantomData,
            },
            StarkVerifyingKey {
                commit,
                pc_start,
                initial_global_cumulative_sum,
                chip_information,
                chip_ordering,
            },
        )
    }

    fn pk_from_vk(
        &self,
        program: &A::Program,
        vk: &StarkVerifyingKey<SC>,
    ) -> Self::DeviceProvingKey {
        let generate_traces_copy_span =
            tracing::debug_span!("generate preprocessed traces and copy to device").entered();

        let (named_preprocessed_data, num_constraints): (Vec<_>, Vec<_>) = self
            .machine()
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

                let num_main_constraints = get_symbolic_constraints(
                    &chip.air,
                    chip.preprocessed_width(),
                    PROOF_MAX_NUM_PVS,
                )
                .len();
                let num_permutation_constraints = count_permutation_constraints(
                    &chip.sends,
                    &chip.receives,
                    chip.logup_batch_size(),
                    chip.air.commit_scope(),
                );
                (
                    (chip.name(), chip.local_only(), prep_trace),
                    (chip.name(), num_main_constraints + num_permutation_constraints),
                )
            })
            .unzip();

        let mut named_preprocessed_data = named_preprocessed_data
            .into_iter()
            .filter(|(_, _, prep_trace)| prep_trace.is_some())
            .map(|(name, local_only, prep_trace)| {
                let prep_trace = prep_trace.unwrap();
                let event = self.events.preprocessed.get(&name).unwrap().clone();
                let stream = self.chip_streams.get(&name).unwrap().clone();
                let domain = natural_domain_for_degree(self.config(), prep_trace.height());
                let dimensions = prep_trace.dimensions();
                let (tx, rx) = oneshot::channel();
                rayon::spawn(move || {
                    let stream = stream;
                    let trace = prep_trace.to_device_async(&stream).unwrap().to_column_major();
                    tx.send(trace).unwrap();
                });
                (name, domain, event, local_only, rx, dimensions)
            })
            .collect::<Vec<_>>();

        named_preprocessed_data
            .sort_by_key(|(name, domain, _, _, _, _)| (Reverse(domain.size()), name.clone()));

        let (commitment_data, local_only): (Vec<_>, Vec<_>) = named_preprocessed_data
            .into_iter()
            .map(|(_, domain, event, local_only, rx, _)| {
                let trace = rx.recv().unwrap();
                ((domain, trace, event), local_only)
            })
            .collect();

        generate_traces_copy_span.exit();

        // Commit to the batch of traces.
        let commit_span = tracing::debug_span!("commit to preprocessed traces").entered();
        let (commit, data) = self.committer.commit(&commitment_data, &self.main_stream);

        commit_span.exit();

        let constraints_map: HashMap<_, _> = num_constraints.into_iter().collect();

        // Get the preprocessed traces
        let traces = commitment_data.into_iter().map(|(_, trace, _)| trace).collect::<Vec<_>>();

        let StarkVerifyingKey { pc_start, initial_global_cumulative_sum, chip_ordering, .. } = vk;

        StarkProvingKeyDevice {
            commit,
            pc_start: *pc_start,
            initial_global_cumulative_sum: *initial_global_cumulative_sum,
            traces,
            data,
            chip_ordering: chip_ordering.clone(),
            constraints_map: constraints_map.clone(),
            local_only,
            phantom: PhantomData,
        }
    }

    fn open(
        &self,
        pk: &Self::DeviceProvingKey,
        data: ShardMainData<SC, Self::DeviceMatrix, Self::DeviceProverData>,
        challenger: &mut SC::Challenger,
    ) -> Result<ShardProof<SC>, Self::Error> {
        let proof = {
            let span = tracing::Span::current();
            let _span = span.enter();

            let setup_span = tracing::debug_span!("process shard data").entered();

            let ShardMainData { traces, main_commit, mut main_data, chip_ordering, public_values } =
                data;

            let shard_chips = self.machine.shard_chips_ordered(&chip_ordering).collect::<Vec<_>>();
            let shard_chip_names = shard_chips.iter().map(|chip| chip.name()).collect::<Vec<_>>();
            let shard_chip_stream =
                traces.iter().map(|trace| trace.stream().clone()).collect::<Vec<_>>();

            assert!(shard_chips.len() == traces.len());

            let domains = traces
                .iter()
                .map(|trace| {
                    let config = self.machine.config();
                    natural_domain_for_degree(config, trace.height())
                })
                .collect::<Vec<_>>();

            let local_only = shard_chips.iter().map(|chip| chip.local_only()).collect::<Vec<_>>();

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
            challenger.observe_slice(&public_values[0..self.num_pv_elts()]);
            challenger.observe(main_commit.clone());

            setup_span.exit();

            // Get the permutation challenges.
            let local_permutation_challenges =
                (0..2).map(|_| challenger.sample_ext_element()).collect::<Vec<_>>();

            // Generate permutation traces.
            let permutation_span =
                tracing::debug_span!("generate and commit to permutation traces").entered();
            let permutation_traces_and_cumulative_sums = self.generate_permutation_traces(
                pk,
                &shard_chips,
                &traces,
                &local_permutation_challenges,
            )?;
            let (permutation_traces, cumulative_sums): (Vec<_>, Vec<_>) =
                permutation_traces_and_cumulative_sums.into_iter().unzip();

            info!(
                "Shard: [{}]",
                shard_chips.iter().map(|c| c.name()).collect::<Vec<_>>().join(", ")
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
                let local_sum = sums;
                CanObserve::<BabyBear>::observe_slice(challenger, local_sum.as_base_slice());
            }

            // Get a challenge for folding the constraints.
            //
            // *Remark*: this is called `alpha` in [sp1_core].
            let folding_challenge: SC::Challenge = challenger.sample_ext_element();

            // Compute quotient values.

            let quotient_span =
                tracing::debug_span!("generate and commit to quotient values").entered();

            let permutation_challenges = local_permutation_challenges.to_vec();

            // For each chip, get the quotient domains, evaluations on the quotient domain, and compute
            // the quotient values.

            let permutation_challenges_device =
                permutation_challenges.to_device_async(&self.main_stream).unwrap();
            let public_values_device = public_values.to_device_async(&self.main_stream).unwrap();
            self.main_stream.record(&self.events.quotient_common_data_to_device)?;

            let mut quotient_values = vec![];

            for (i, chip) in shard_chips.iter().enumerate() {
                let log_quotient_degree = chip.log_quotient_degree();
                let trace_domain = domains[i];

                let quotient_domain =
                    trace_domain.create_disjoint_domain(trace_domain.size() << log_quotient_degree);

                let local_cumulative_sum = cumulative_sums[i];

                // Get the evaluations on the quotient domain. If the LDE evalutions can be used, we
                // just bit-reverse them to match the expected quotient kernel.
                let use_lde = chip.log_quotient_degree() == self.committer.log_blowup;

                let chip_quotient_values = if use_lde {
                    let prep_eval = pk
                        .chip_ordering
                        .get(&chip.name())
                        .map(|&index| {
                            self.committer.get_evaluations_on_domain(
                                trace_domain,
                                quotient_domain,
                                &pk.traces[index],
                            )
                        })
                        .transpose()
                        .unwrap();
                    let prep_eval = prep_eval.unwrap_or_else(ColMajorMatrixDevice::null);
                    let main_eval = &mut main_data.matrices_mut()[chip_ordering[&chip.name()]];
                    main_eval.bit_reverse_rows().unwrap();

                    let perm_eval = &mut perm_prover_data.matrices_mut()[i];
                    perm_eval.bit_reverse_rows().unwrap();

                    main_eval.stream().wait_event(&self.events.quotient_common_data_to_device)?;

                    let chip_num_constraints = pk.constraints_map.get(&chip.name()).unwrap();
                    let powers_of_folding_challenge =
                        folding_challenge.powers().take(*chip_num_constraints).collect::<Vec<_>>();
                    let mut powers_of_folding_challenge_rev = powers_of_folding_challenge.clone();
                    powers_of_folding_challenge_rev.reverse();
                    let quotient_values = self.quotient_generator.compute_values(
                        chip,
                        trace_domain,
                        quotient_domain,
                        &prep_eval,
                        main_eval,
                        perm_eval,
                        &public_values_device,
                        local_cumulative_sum,
                        &powers_of_folding_challenge_rev,
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
                        .map(|index| {
                            self.committer.get_evaluations_on_domain(
                                trace_domain,
                                quotient_domain,
                                &pk.traces[*index],
                            )
                        })
                        .transpose()
                        .unwrap();
                    let prep_eval = prep_eval.unwrap_or_else(ColMajorMatrixDevice::null);

                    let main_eval = self
                        .committer
                        .get_evaluations_on_domain(trace_domain, quotient_domain, &traces[i])
                        .unwrap();
                    let perm_eval = self
                        .committer
                        .get_evaluations_on_domain(
                            trace_domain,
                            quotient_domain,
                            &perm_domains_and_traces[i].1,
                        )
                        .unwrap();

                    main_eval.stream().wait_event(&self.events.quotient_common_data_to_device)?;

                    let chip_num_constraints = pk.constraints_map.get(&chip.name()).unwrap();
                    // println!("chip_num_constraints: {}", chip_num_constraints);
                    let powers_of_folding_challenge =
                        folding_challenge.powers().take(*chip_num_constraints).collect::<Vec<_>>();
                    let mut powers_of_folding_challenge_rev = powers_of_folding_challenge.clone();
                    powers_of_folding_challenge_rev.reverse();
                    self.quotient_generator.compute_values(
                        chip,
                        trace_domain,
                        quotient_domain,
                        &prep_eval,
                        &main_eval,
                        &perm_eval,
                        &public_values_device,
                        local_cumulative_sum,
                        &powers_of_folding_challenge_rev,
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
            for (trace, local_only) in pk.traces.iter().zip(pk.local_only.iter()) {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                input_heights.insert(domain.log_n);
                let local_open = self.opening_prover.eval(domain, trace, zeta);
                observe_device_buffer::<SC>(challenger, &local_open);
                let next_open = if !local_only {
                    let next_open =
                        self.opening_prover.eval(domain, trace, domain.next_point(zeta).unwrap());
                    observe_device_buffer::<SC>(challenger, &next_open);
                    Some(next_open)
                } else {
                    None
                };
                preprocessed_opens.push((domain.log_n, local_open, next_open));
            }

            // Openings for global main traces (if any).
            let mut main_openings = vec![];
            for (trace, local_only) in traces.iter().zip(local_only.iter()) {
                let domain = natural_domain_for_degree(self.config(), trace.height());
                input_heights.insert(domain.log_n);
                let local_open = self.opening_prover.eval(domain, trace, zeta);
                observe_device_buffer::<SC>(challenger, &local_open);
                let next_open = if !local_only {
                    let next_open =
                        self.opening_prover.eval(domain, trace, domain.next_point(zeta).unwrap());
                    observe_device_buffer::<SC>(challenger, &next_open);
                    Some(next_open)
                } else {
                    None
                };
                main_openings.push((domain.log_n, local_open, next_open));
            }

            let mut perm_openings = vec![];
            // Openings for permutation traces.
            for (domain, trace, _) in perm_domains_and_traces {
                let local_open = self.opening_prover.eval(domain, &trace, zeta);
                observe_device_buffer::<SC>(challenger, &local_open);
                let next_open =
                    self.opening_prover.eval(domain, &trace, domain.next_point(zeta).unwrap());
                observe_device_buffer::<SC>(challenger, &next_open);
                input_heights.insert(domain.log_n);
                perm_openings.push((domain.log_n, local_open, next_open));
            }
            // Openings for quotient traces
            let mut quot_openings = vec![];
            for (domain, trace, _) in quotient_domains_and_chunks.into_iter() {
                let open = self.opening_prover.eval(domain, &trace, zeta);
                observe_device_buffer::<SC>(challenger, &open);
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
                .map(|i| (*i, SC::Challenge::ONE))
                .collect::<BTreeMap<_, _>>();
            for (lde, (log_height, local_open, next_open)) in
                pk.data.matrices().iter().zip_eq(preprocessed_opens.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::GENERATOR,
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                if let Some(next_open) = next_open {
                    let g = BabyBear::two_adic_generator(*log_height);
                    self.opening_prover.batch_update(
                        batched_openings.get_mut(&lde_log_height).unwrap(),
                        lde,
                        SC::Val::GENERATOR,
                        next_open,
                        zeta * g,
                        alpha,
                        alpha_offsets.get_mut(&lde_log_height).unwrap(),
                    );
                }
            }

            // Batch the main traces.
            for (lde, (log_height, local_open, next_open)) in
                main_data.matrices().iter().zip_eq(main_openings.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::GENERATOR,
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                if let Some(next_open) = next_open {
                    let g = BabyBear::two_adic_generator(*log_height);
                    self.opening_prover.batch_update(
                        batched_openings.get_mut(&lde_log_height).unwrap(),
                        lde,
                        SC::Val::GENERATOR,
                        next_open,
                        zeta * g,
                        alpha,
                        alpha_offsets.get_mut(&lde_log_height).unwrap(),
                    );
                }
            }

            // Batch the permutation traces.
            for (lde, (log_height, local_open, next_open)) in
                perm_prover_data.matrices().iter().zip_eq(perm_openings.iter())
            {
                let lde_log_height = log_height + log_blowup;
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::GENERATOR,
                    local_open,
                    zeta,
                    alpha,
                    alpha_offsets.get_mut(&lde_log_height).unwrap(),
                );
                let g = BabyBear::two_adic_generator(*log_height);
                self.opening_prover.batch_update(
                    batched_openings.get_mut(&lde_log_height).unwrap(),
                    lde,
                    SC::Val::GENERATOR,
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
                    SC::Val::GENERATOR,
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
                        2 * <SC::Challenge as FieldExtensionAlgebra<SC::Val>>::D,
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
                &self.main_stream,
            );

            let prover_data = vec![&pk.data, &main_data, &perm_prover_data, &quotient_prover_data];
            let log_global_max_height_iter =
                prover_data.iter().flat_map(|data| data.matrices().iter().map(|mat| mat.height));

            let log_global_max_height =
                Iterator::max(log_global_max_height_iter).unwrap().ilog2() as usize;

            let query_openings = self.committer.mmcs_committer.query_open_batch(
                &query_indices,
                &prover_data,
                log_global_max_height,
                false,
                &self.main_stream,
            );

            let opening_proof = TwoAdicFriPcsProof { fri_proof, query_openings };

            // Get the openings for the chips.
            let preprocessed_opens = preprocessed_opens
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.map(|buf| buf.to_host())))
                .collect::<Vec<_>>();
            let main_openings = main_openings
                .into_iter()
                .map(|(_, local, next)| (local.to_host(), next.map(|buf| buf.to_host())))
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
                        if let Some(next) = next {
                            AirOpenedValues { local, next }
                        } else {
                            let width = local.len();
                            AirOpenedValues { local, next: vec![SC::Challenge::ZERO; width] }
                        }
                    })
                    .unwrap_or(AirOpenedValues { local: vec![], next: vec![] });

                let main = {
                    let (local, next) = main_openings[i].clone();
                    if let Some(next) = next {
                        AirOpenedValues { local, next }
                    } else {
                        let width = local.len();
                        AirOpenedValues { local, next: vec![SC::Challenge::ZERO; width] }
                    }
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
                    local_cumulative_sum: cumulative_sums[i],
                    log_degree,
                });
            }

            compute_evaluations_span.exit();

            Ok(ShardProof::<SC> {
                commitment: ShardCommitment { main_commit, permutation_commit, quotient_commit },
                opened_values: ShardOpenedValues { chips: opened_values },
                opening_proof,
                chip_ordering,
                public_values,
            })
        };

        let cleanup_span = tracing::debug_span!("cleanup").entered();

        self.main_stream.synchronize().unwrap();
        for stream in self.chip_streams.values() {
            stream.synchronize().unwrap();
        }
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
        A: for<'a> Air<DebugConstraintBuilder<'a, SC::Val, SC::Challenge>>,
    {
        self.machine().generate_dependencies(&mut records, &opts, None);

        // Observe the preprocessed commitment.
        pk.observe_into(challenger);

        let shard_proofs = records
            .iter_mut()
            .map(|record| {
                let traces = self.generate_traces(record);
                let shard_data = self.commit(record, traces);

                let span = tracing::Span::current();
                let _span = span.enter();
                self.open(pk, shard_data, &mut challenger.clone())
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
        main_traces: &[GpuMatrix<SC::Val>],
        random_elements: &[SC::Challenge],
    ) -> Result<Vec<(GpuMatrix<SC::Val>, SC::Challenge)>, CudaError> {
        chips
            .iter()
            .zip(main_traces.iter())
            .map(|(chip, main_trace)| {
                let preprocessed_trace =
                    pk.chip_ordering.get(&chip.name()).map(|&index| &pk.traces[index]);

                self.permutation_trace_generator.generate_flattened_permutation_trace(
                    chip,
                    preprocessed_trace,
                    main_trace,
                    random_elements,
                )
            })
            .collect::<Result<Vec<_>, CudaError>>()
    }
}

fn observe_device_buffer<SC: BabyBearFriConfig>(
    challenger: &mut SC::Challenger,
    buffer: &DeviceBuffer<BinomialExtensionField<BabyBear, 4>>,
) {
    let host_buffer = buffer
        .to_host()
        .into_iter()
        .flat_map(|c: BinomialExtensionField<BabyBear, 4>| c.as_base_slice().to_vec())
        .collect::<Vec<BabyBear>>();
    challenger.observe_slice(&host_buffer);
}

// #[cfg(test)]
// pub mod tests {

//     use sp1_core_executor::{programs::tests::FIBONACCI_ELF, ExecutionRecord, Executor, Program};
//     use sp1_core_machine::{riscv::RiscvAir, utils::run_test};
//     use sp1_recursion_core::stark::BabyBearPoseidon2Outer;
//     use sp1_stark::StarkGenericConfig;

//     use crate::{
//         merkle_tree::FieldMerkleTreeDeviceCommitter,
//         poseidon2::{baby_bear::DeviceHasherBabyBear, bn254::DeviceHasherBn254},
//         utils::init_tracer,
//     };

//     use super::*;

//     pub fn execute_core(program: Program) -> ExecutionRecord {
//         let opts = SP1CoreOpts::default();
//         let mut runtime = Executor::new(program, opts);
//         runtime.run().unwrap();
//         runtime.record
//     }

//     #[test]
//     fn test_fibonacci_poseidon_2_baby_bear_prove() {
//         let program = Program::from(FIBONACCI_ELF).unwrap();

//         init_tracer();
//         run_test::<StarkGpuProver<_, FieldMerkleTreeDeviceCommitter<DeviceHasherBabyBear>, _>>(
//             program,
//         )
//         .unwrap();
//     }

//     #[test]
//     fn test_fibonacci_poseidon2_bn254_prove() {
//         use sp1_core_executor::SP1Context;
//         use sp1_core_machine::io::SP1Stdin;

//         let program = Program::from(FIBONACCI_ELF).unwrap();

//         type SC = BabyBearPoseidon2Outer;

//         type P = StarkGpuProver<
//             SC,
//             FieldMerkleTreeDeviceCommitter<DeviceHasherBn254>,
//             RiscvAir<BabyBear>,
//         >;

//         init_tracer();

//         let config = BabyBearPoseidon2Outer::new();

//         // Execute the program.
//         let runtime = tracing::debug_span!("runtime.run(...)").in_scope(|| {
//             let mut runtime = Executor::new(program, SP1CoreOpts::default());
//             runtime.run().unwrap();
//             runtime
//         });

//         let machine = RiscvAir::machine(config);
//         let prover = P::new(machine);
//         let inputs = SP1Stdin::new();
//         let (pk, vk) = prover.setup(runtime.program.as_ref());
//         let (proof, _, _) = sp1_core_machine::utils::prove_with_context(
//             &prover,
//             &pk,
//             Program::clone(&runtime.program),
//             &inputs,
//             SP1CoreOpts::default(),
//             SP1Context::default(),
//             None,
//         )
//         .unwrap();

//         let mut challenger = prover.config().challenger();
//         prover.machine().verify(&vk, &proof, &mut challenger).unwrap();
//     }
// }
