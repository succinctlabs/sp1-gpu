use std::collections::HashMap;

use p3_field::{ExtensionField, PrimeField32};
use sp1_core::{
    air::MachineAir,
    stark::{StarkGenericConfig, StarkMachine},
};

use crate::matrix::{ColMajorMatrixDevice, DeviceMatrix};

pub struct FriGpuProver<SC: StarkGenericConfig, A> {
    machine: StarkMachine<SC, A>,
}

pub struct MainTraceData<SC: StarkGenericConfig> {
    pub index: usize,
    pub traces: Vec<ColMajorMatrixDevice<SC::Val>>,
    pub chip_ordering: HashMap<String, usize>,
    pub public_values: Vec<SC::Val>,
}

// pub type MatrixView<P, T> = <<P as ProverConfig>::Matrix<T> as Matrix<T>>::View;
// pub type MatrixViewMut<P, T> = <<P as ProverConfig>::Matrix<T> as Matrix<T>>::ViewMut;

// pub trait ProverConfig {
//     type Val: PrimeField32;
//     type Challenge: ExtensionField<Self::Val>;

//     type Record: MachineRecord;
//     type Air: MachineAir<Self::Val, Record = Self::Record>;
//     type Runtime;

//     type Matrix<T>: Matrix<T>;

//     type Commitment;
//     type ProverData: Send + Sync;

//     type Domain: PolynomialSpace<Val = Self::Val>;
//     type Pcs: Pcs<
//         Self::Challenge,
//         Challenger<Self::SC>,
//         Commitment = Self::Commitment,
//         Domain = Self::Domain,
//     >;
//     type SC: StarkGenericConfig<Val = Self::Val, Challenge = Self::Challenge, Pcs = Self::Pcs>;

//     fn generate_main_traces(
//         chips: &[&Chip<Self::Val, Self::Air>],
//         shard: &Self::Record,
//         index: usize,
//     ) -> Vec<(String, Self::Matrix<Self::Val>)>;

//     fn generate_permutation_traces(
//         chips: &[&Chip<Self::Val, Self::Air>],
//         traces: &[Self::Matrix<Self::Val>],
//     ) -> Vec<Self::Matrix<Self::Challenge>>;

//     fn compute_quotient_values();

//     fn commit_evaluations(
//         &self,
//         evaluations: Vec<(Self::Domain, MatrixView<Self, Self::Val>)>,
//     ) -> (Self::Commitment, Self::ProverData);
// }

// pub struct MainProverData<C: ProverConfig> {
//     pub main_commit: Com<C::SC>,
//     pub main_data: C::DeviceProverData,
// }

// pub struct MainData<C: ProverConfig> {
//     pub trace_data: MainTraceData<C>,
//     pub prover_data: MainProverData<C>,
// }

// pub struct StarkProver<C: ProverConfig> {
//     machine: StarkMachine<C::SC, C::Air>,
// }

// pub struct BabyBearPoseidon2GpuConfig;

// impl<C: ProverConfig> StarkProver<C> {
//     // pub fn generate_traces(&self, shard: &C::Record, index: usize) -> MainTraceData<C> {
//     //     // Filter the chips based on what is used.
//     //     let shard_chips = self.machine.shard_chips(shard).collect::<Vec<_>>();

//     //     // For each chip, generate the trace, copy to the device, and transpose.
//     //     let parent_span = tracing::debug_span!("generate traces for shard");
//     //     let mut named_traces = parent_span.in_scope(|| {
//     //                 shard_chips
//     //                     .par_iter()
//     //                     .map(|chip| {
//     //                         let chip_name = chip.name();
//     //                         // We need to create an outer span here because, for some reason,
//     //                         // the #[instrument] macro on the chip impl isn't attaching its span to `parent_span`
//     //                         // to avoid the unnecessary span, remove the #[instrument] macro.
//     //                         let trace =
//     //                             tracing::debug_span!(parent: &parent_span, "generate trace for chip", %chip_name)
//     //                                 .in_scope(|| {
//     //                                 let host_trace = chip.generate_trace(shard, &mut C::Record::default());
//     //                                 let device_trace = host_trace.to_device();
//     //                                 device_trace.to_column_major()
//     //                     });
//     //                         (chip_name, trace)
//     //                     })
//     //                     .collect::<Vec<_>>()
//     //             });

//     //     // Order the chips and traces by trace size (biggest first), and get the ordering map.
//     //     named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

//     //     // Get the chip ordering.
//     //     let chip_ordering = named_traces
//     //         .iter()
//     //         .enumerate()
//     //         .map(|(i, (name, _))| (name.to_owned(), i))
//     //         .collect();

//     //     let traces = named_traces
//     //         .into_iter()
//     //         .map(|(_, trace)| trace)
//     //         .collect::<Vec<_>>();

//     //     MainTraceData {
//     //         traces,
//     //         chip_ordering,
//     //         index,
//     //         public_values: shard.public_values(),
//     //     }
//     // }

//     pub fn generate_traces(&self, shard: &C::Record, index: usize) -> MainTraceData<C> {
//         // Filter the chips based on what is used.
//         let shard_chips = self.machine.shard_chips(shard).collect::<Vec<_>>();

//         let mut named_traces = C::generate_main_traces(&shard_chips, shard, index);

//         // Order the chips and traces by trace size (biggest first), and get the ordering map.
//         named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

//         // Get the chip ordering.
//         let chip_ordering = named_traces
//             .iter()
//             .enumerate()
//             .map(|(i, (name, _))| (name.to_owned(), i))
//             .collect();

//         let traces = named_traces
//             .into_iter()
//             .map(|(_, trace)| trace)
//             .collect::<Vec<_>>();

//         MainTraceData {
//             traces,
//             chip_ordering,
//             index,
//             public_values: shard.public_values(),
//         }
//     }

//     pub fn commit_traces(&self, trace_data: &MainTraceData<C>) -> MainProverData<C> {
//         let pcs = self.machine.config().pcs();
//         let domains_and_traces = trace_data
//             .traces
//             .iter()
//             .map(|trace| {
//                 let domain = pcs.natural_domain_for_degree(trace.height());
//                 (domain, trace.view())
//             })
//             .collect::<Vec<_>>();

//         // Commit to the batch of traces.
//         let (main_commit, main_data) = pcs.device_commit(domains_and_traces);

//         MainProverData {
//             main_commit,
//             main_data,
//         }
//     }

//     pub fn commit_main(&self, shard: &C::Record, index: usize) -> MainData<C> {
//         let trace_data = self.generate_traces(shard, index);
//         let prover_data = self.commit_traces(&trace_data);

//         MainData {
//             trace_data,
//             prover_data,
//         }
//     }

//     pub fn prove_shard(
//         &self,
//         pk: &StarkProvingKey<C::SC>,
//         data: MainData<C>,
//         challenger: &mut Challenger<C::SC>,
//     ) {
//         let traces = &data.trace_data.traces;
//         let chips = self
//             .machine
//             .shard_chips_ordered(&data.trace_data.chip_ordering)
//             .collect::<Vec<_>>();

//         let degrees = traces
//             .iter()
//             .map(|trace| trace.height())
//             .collect::<Vec<_>>();

//         let log_degrees = degrees
//             .iter()
//             .map(|degree| log2_strict_usize(*degree))
//             .collect::<Vec<_>>();

//         let log_quotient_degrees = chips
//             .iter()
//             .map(|chip| chip.log_quotient_degree())
//             .collect::<Vec<_>>();

//         let pcs = self.pcs();
//         let trace_domains = degrees
//             .iter()
//             .map(|degree| pcs.natural_domain_for_degree(*degree))
//             .collect::<Vec<_>>();

//         // Obtain the challenges used for the permutation argument.
//         let mut permutation_challenges: Vec<C::Challenge> = Vec::new();
//         for _ in 0..2 {
//             permutation_challenges.push(challenger.sample_ext_element());
//         }
//         let alpha = permutation_challenges[0];
//         let beta = permutation_challenges[1];

//         // Generate the permutation traces.
//         let pk_traces = &pk.traces;
//         let chip_ordering = &pk.chip_ordering;
//         let permutation_traces =
//             tracing::debug_span!("generate permutation traces").in_scope(|| {
//                 chips
//                     .par_iter()
//                     .zip(traces.par_iter())
//                     .map(|(chip, main_trace)| {
//                         let preprocessed_trace = chip_ordering
//                             .get(&chip.name())
//                             .map(|&index| &pk_traces[index]);
//                         let preprocessed_trace =
//                             preprocessed_trace.map(|mat| mat.to_device().to_column_major());

//                         let device_interactions =
//                             HostInteractions::<C::Val>::new(chip.sends(), chip.receives())
//                                 .to_device();

//                         let mut permutation_trace = ColMajorMatrixDevice::with_capacity(
//                             chip.permutation_width(),
//                             main_trace.height(),
//                         );
//                         unsafe {
//                             permutation_trace.set_max_width();
//                             let num_threads_per_block = 256;
//                             let num_blocks = main_trace.height().div_ceil(num_threads_per_block);
//                             device_interactions
//                                 .generate_permutation_trace(
//                                     permutation_trace.view_mut(),
//                                     preprocessed_trace
//                                         .map(|mat| mat.view())
//                                         .unwrap_or(MatrixViewDevice::null(false)),
//                                     main_trace.view(),
//                                     alpha,
//                                     beta,
//                                     chip.logup_batch_size(),
//                                     num_blocks,
//                                     num_threads_per_block,
//                                 )
//                                 .unwrap();
//                         }
//                         permutation_trace
//                     })
//                     .collect::<Vec<_>>()
//             });
//     }

//     fn pcs(&self) -> &C::Pcs {
//         self.machine.config().pcs()
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
// }
