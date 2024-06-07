use std::{cmp::Reverse, marker::PhantomData};

use rayon::prelude::*;

use p3_baby_bear::BabyBear;
use p3_matrix::Matrix;

use sp1_core::{
    air::MachineAir,
    stark::{MachineRecord, StarkGenericConfig, StarkMachine},
    utils::BabyBearPoseidon2,
};

use super::{natural_domain_for_degree, CpuMainTraceData, MainTraceData};

#[derive(Clone, Copy, Debug)]
pub struct CpuTraceGenerator<SC, A>(PhantomData<(SC, A)>);

impl<SC, A> CpuTraceGenerator<SC, A>
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
    pub fn generate_main_traces(
        &self,
        machine: &StarkMachine<SC, A>,
        shard: &A::Record,
        index: usize,
    ) -> CpuMainTraceData<SC> {
        // Filter the chips based on what is used.
        let shard_chips = machine.shard_chips(shard).collect::<Vec<_>>();

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
