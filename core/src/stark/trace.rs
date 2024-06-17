use std::{cmp::Reverse, marker::PhantomData};

use rayon::prelude::*;

use p3_baby_bear::BabyBear;
use p3_matrix::Matrix;

use sp1_core::{
    air::MachineAir,
    stark::{MachineRecord, StarkMachine},
};

use super::{natural_domain_for_degree, BabyBearPoseidon2Config, CpuMainTraceData, MainTraceData};

#[derive(Clone, Copy, Debug)]
pub struct CpuTraceGenerator<SC, A>(PhantomData<(SC, A)>);

impl<SC, A> CpuTraceGenerator<SC, A>
where
    SC: BabyBearPoseidon2Config,
    A: MachineAir<BabyBear>,
    A::Record: Sync,
{
    pub fn generate_main_traces(
        &self,
        machine: &StarkMachine<SC, A>,
        shard: &A::Record,
        index: usize,
    ) -> CpuMainTraceData<SC> {
        // Filter the chips based on what is used.
        let shard_chips = machine.shard_chips(shard).collect::<Vec<_>>();

        // For each chip, generate the trace, copy to the device, and transpose.

        let parent_span = tracing::debug_span!("generate traces for shard");
        let mut named_traces = parent_span.in_scope(|| {
            shard_chips
                .par_iter()
                .map(|chip| {
                    let chip_name = chip.name();

                    // We need to create an outer span here because, for some reason,
                    // the #[instrument] macro on the chip impl isn't attaching its span to `parent_span`
                    // to avoid the unnecessary span, remove the #[instrument] macro.
                    let trace =
                        tracing::debug_span!(parent: &parent_span, "generate trace for chip", %chip_name)
                            .in_scope(|| chip.generate_trace(shard, &mut A::Record::default()));
                    (chip_name, trace)
                })
                .collect::<Vec<_>>()
        });

        // Order the chips and traces by trace size (biggest first), and get the ordering map.
        named_traces.sort_by_key(|(_, trace)| Reverse(trace.height()));

        // Get the chip ordering.
        let chip_ordering = named_traces
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.to_owned(), i))
            .collect();

        let config = machine.config();
        let (domains, traces): (Vec<_>, Vec<_>) = named_traces
            .into_iter()
            .map(|(_, trace)| (natural_domain_for_degree(config, trace.height()), trace))
            .unzip();

        MainTraceData {
            traces,
            domains,
            chip_ordering,
            index,
            public_values: shard.public_values(),
        }
    }
}

impl<SC, A> Default for CpuTraceGenerator<SC, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}
