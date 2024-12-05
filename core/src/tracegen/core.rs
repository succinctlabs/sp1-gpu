use p3_air::BaseAir;
use p3_baby_bear::BabyBear;
use sp1_core_machine::global::GlobalChip;
use sp1_core_machine::memory::MemoryChipType;
use sp1_core_machine::syscall::chip::SyscallShardKind;
use sp1_core_machine::{
    alu::AddSubChip, memory::MemoryGlobalChip, memory::MemoryLocalChip, syscall::chip::SyscallChip,
};
use sp1_stark::septic_curve::SepticCurve;

use crate::device::DeviceBuffer;
use crate::{
    cuda_runtime::stream::CudaStream,
    device::{error::CudaError, memory::ToDevice},
    matrix::ColMajorMatrixDevice,
};

use super::DeviceAir;
use crate::tracegen;

impl DeviceAir<BabyBear> for AddSubChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events =
            input.add_events.iter().chain(input.sub_events.iter()).copied().collect::<Vec<_>>();

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows_device(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <AddSubChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_add_sub_generate_trace(
                trace.view_mut(),
                events.as_ptr(),
                events.len() as u32,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DeviceAir<BabyBear> for MemoryLocalChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events = input.get_local_mem_events().cloned().collect::<Vec<_>>();
        let nb_events = events.len() as u32;

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows_device(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <MemoryLocalChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_memory_local_generate_trace_round_1(
                trace.view_mut(),
                events.as_ptr(),
                nb_events,
                stream.handle(),
            );
        }

        // let mut cumulative_sums = vec![SepticCurve::<BabyBear>::default(); trace.height()]
        //     .to_device_async(stream)
        //     .unwrap();

        // unsafe {
        //     tracegen::ffi::core_memory_local_generate_trace_round_2(
        //         trace.view_mut(),
        //         cumulative_sums.as_mut_ptr(),
        //         stream.handle(),
        //     );
        // }

        // unsafe {
        //     tracegen::ffi::core_memory_local_generate_trace_round_3(
        //         trace.view_mut(),
        //         cumulative_sums.as_ptr(),
        //         nb_events,
        //         stream.handle(),
        //     );
        // }

        Ok(Some(trace))
    }
}

impl DeviceAir<BabyBear> for MemoryGlobalChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let mut events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };
        events.sort_by_key(|event| event.addr);
        let nb_events = events.len() as u32;

        let previous_addr_bits = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr_bits,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr_bits,
        };

        let previous_addr =
            previous_addr_bits.iter().enumerate().fold(0u32, |acc, (i, &bit)| acc + (bit << i));

        let is_receive = match self.kind {
            MemoryChipType::Initialize => false,
            MemoryChipType::Finalize => true,
        };

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows_device(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <MemoryGlobalChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_memory_global_generate_trace_round_1(
                trace.view_mut(),
                events.as_ptr(),
                previous_addr,
                nb_events,
                is_receive,
                stream.handle(),
            );
        }

        let mut cumulative_sums = vec![SepticCurve::<BabyBear>::default(); trace.height()]
            .to_device_async(stream)
            .unwrap();

        unsafe {
            tracegen::ffi::core_memory_global_generate_trace_round_2(
                trace.view_mut(),
                cumulative_sums.as_mut_ptr(),
                stream.handle(),
            );
        }

        unsafe {
            tracegen::ffi::core_memory_global_generate_trace_round_3(
                trace.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DeviceAir<BabyBear> for SyscallChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events = match self.shard_kind() {
            SyscallShardKind::Core => &input.syscall_events,
            SyscallShardKind::Precompile => &input
                .precompile_events
                .all_events()
                .map(|(event, _)| event.to_owned())
                .collect::<Vec<_>>(),
        };
        let nb_events = events.len() as u32;

        let is_receive = match self.shard_kind() {
            SyscallShardKind::Core => false,
            SyscallShardKind::Precompile => true,
        };

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows_device(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <SyscallChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        // Generate the trace.
        unsafe {
            trace.set_max_width();
            tracegen::ffi::core_syscall_generate_trace_round_1(
                trace.view_mut(),
                events.as_ptr(),
                nb_events,
                is_receive,
                stream.handle(),
            );
        }

        let mut cumulative_sums = vec![SepticCurve::<BabyBear>::default(); trace.height()]
            .to_device_async(stream)
            .unwrap();

        unsafe {
            tracegen::ffi::core_syscall_generate_trace_round_2(
                trace.view_mut(),
                cumulative_sums.as_mut_ptr(),
                stream.handle(),
            );
        }

        unsafe {
            tracegen::ffi::core_syscall_generate_trace_round_3(
                trace.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events,
                stream.handle(),
            );
        }

        Ok(Some(trace))
    }
}

impl DeviceAir<BabyBear> for GlobalChip {
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // Get the events for the chip.
        let events = &input.global_interaction_events;
        let nb_events = events.len() as u32;

        // Copy the events to device.
        let events = events.to_device_async(stream)?;

        // Get the number of rows.
        let nb_rows = self.num_rows_device(input).unwrap();

        // Allocate the matrix.
        let mut trace = ColMajorMatrixDevice::<BabyBear>::with_capacity_in(
            <GlobalChip as BaseAir<BabyBear>>::width(self),
            nb_rows,
            stream,
        )?;

        tracing::debug!("nb events: {:?}", nb_events);
        tracing::debug!("nb rows: {:?}", trace.height());

        // Generate the trace.
        tracing::debug_span!("global generate trace round 1").in_scope(|| unsafe {
            trace.set_max_width();
            tracegen::ffi::core_global_generate_trace_round_1(
                trace.view_mut(),
                events.as_ptr(),
                nb_events,
                stream.handle(),
            );
        });

        let mut cumulative_sums =
            DeviceBuffer::<SepticCurve<BabyBear>>::with_capacity_in(trace.height(), stream)?;

        tracing::debug_span!("global generate trace round 2").in_scope(|| unsafe {
            tracegen::ffi::core_global_generate_trace_round_2(
                trace.view_mut(),
                cumulative_sums.as_mut_ptr(),
                stream.handle(),
            );
        });

        tracing::debug_span!("global generate trace round 3").in_scope(|| unsafe {
            tracegen::ffi::core_global_generate_trace_round_3(
                trace.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events,
                stream.handle(),
            );
        });

        Ok(Some(trace))
    }
}

#[cfg(test)]
mod tests {
    use crate::device::memory::ToHost;
    use crate::{
        cuda_runtime::ffi::DEFAULT_STREAM, device::memory::ToDevice, matrix::RowMajorMatrixDevice,
    };
    use p3_baby_bear::BabyBear;
    use p3_field::AbstractField;
    use p3_matrix::dense::RowMajorMatrix;
    use p3_matrix::Matrix;
    use sp1_core_executor::events::{GlobalInteractionEvent, SyscallEvent};
    use sp1_core_executor::{
        events::AluEvent, events::MemoryInitializeFinalizeEvent, events::MemoryLocalEvent,
        ExecutionRecord, Opcode,
    };
    use sp1_core_machine::alu::AddSubChip;
    use sp1_core_machine::global::GlobalChip;
    use sp1_core_machine::memory::{MemoryChipType, MemoryLocalChip};
    use sp1_core_machine::riscv::MemoryGlobalChip;
    use sp1_core_machine::syscall::chip::SyscallChip;
    use sp1_stark::air::MachineAir;
    use sp1_stark::septic_curve::SepticCurve;

    use crate::tracegen;
    use rand::Rng;

    #[test]
    fn test_add_sub_generate_trace() {
        let mut shard = ExecutionRecord::default();
        shard.add_events = [AluEvent::new(0, 0, Opcode::ADD, 14, 8, 6)].repeat(100);

        let chip = AddSubChip;
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_device =
            RowMajorMatrixDevice::new(trace.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.add_events.to_device().unwrap().as_ptr();
        unsafe {
            tracegen::ffi::core_add_sub_generate_trace(
                trace_device.view_mut(),
                events,
                shard.add_events.len() as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();
        assert_eq!(trace, gpu_trace);
    }

    #[test]
    fn test_memory_local_generate_trace() {
        // let mut rng = rand::thread_rng();
        let mut shard = ExecutionRecord::default();
        // Print current working directory to debug file path issue
        println!("Current working directory: {:?}", std::env::current_dir().unwrap());

        let events: Vec<MemoryLocalEvent> =
            bincode::deserialize(&std::fs::read("./memory_local_events_34.bin").unwrap()).unwrap();

        shard.cpu_local_memory_access = events;
        // shard.cpu_local_memory_access = events;
        //         addr: rng.gen_range(0..10000),
        //         initial_mem_access: MemoryRecord {
        //             shard: rng.gen_range(0..10000),
        //             timestamp: rng.gen_range(0..10000),
        //             value: rng.gen_range(0..10000),
        //         },
        //         final_mem_access: MemoryRecord {
        //             shard: rng.gen_range(0..10000),
        //             timestamp: rng.gen_range(0..10000),
        //             value: rng.gen_range(0..10000),
        //         },
        //     })
        //     .collect::<Vec<_>>();

        let chip = MemoryLocalChip::new();
        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_copy = trace.clone();
        trace_copy.values.fill(BabyBear::zero());
        let mut trace_device =
            RowMajorMatrixDevice::new(trace_copy.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.get_local_mem_events().cloned().collect::<Vec<_>>();
        let nb_events = events.len();
        let events = events.to_device().unwrap().as_ptr();
        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_1(
                trace_device.view_mut(),
                events,
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let mut cumulative_sums =
            vec![SepticCurve::<BabyBear>::default(); trace.height()].to_device().unwrap();

        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_2(
                trace_device.view_mut(),
                cumulative_sums.as_mut_ptr(),
                DEFAULT_STREAM,
            );
        }

        unsafe {
            tracegen::ffi::core_memory_local_generate_trace_round_3(
                trace_device.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();
        assert_eq!(trace, gpu_trace);
    }

    #[test]
    fn test_memory_global_generate_trace() {
        let mut rng = rand::thread_rng();

        for (chip_type, is_receive) in
            [(MemoryChipType::Initialize, false), (MemoryChipType::Finalize, true)]
        {
            let mut shard = ExecutionRecord::default();
            let start_addr = 4;
            let bits: [u32; 32] =
                (0..32).map(|i| (start_addr >> i) & 1).collect::<Vec<_>>().try_into().unwrap();
            let mut events = (0..1000)
                .map(|i| MemoryInitializeFinalizeEvent {
                    addr: 5 + 13 * i, // need this to be distinct, and larger than start_addr
                    value: rng.gen_range(0..1000000),
                    shard: rng.gen_range(0..10000),
                    timestamp: rng.gen_range(0..1000000),
                    used: 1,
                })
                .collect::<Vec<_>>();
            events.sort_by_key(|e| e.addr);
            let nb_events = events.len() as u32;

            match chip_type {
                MemoryChipType::Initialize => {
                    shard.global_memory_initialize_events = events.clone();
                    shard.public_values.previous_init_addr_bits = bits;
                }
                MemoryChipType::Finalize => {
                    shard.global_memory_finalize_events = events.clone();
                    shard.public_values.previous_finalize_addr_bits = bits;
                }
            }

            let chip = MemoryGlobalChip::new(chip_type);

            let trace: RowMajorMatrix<BabyBear> =
                chip.generate_trace(&shard, &mut ExecutionRecord::default());

            let mut trace_copy = trace.clone();
            trace_copy.values.fill(BabyBear::zero());
            let mut trace_device =
                RowMajorMatrixDevice::new(trace_copy.values.to_device().unwrap(), trace.width())
                    .to_column_major();

            let events = events.to_device().unwrap().as_ptr();
            unsafe {
                tracegen::ffi::core_memory_global_generate_trace_round_1(
                    trace_device.view_mut(),
                    events,
                    start_addr,
                    nb_events,
                    is_receive,
                    DEFAULT_STREAM,
                );
            }

            let mut cumulative_sums =
                vec![SepticCurve::<BabyBear>::default(); trace.height()].to_device().unwrap();

            unsafe {
                tracegen::ffi::core_memory_global_generate_trace_round_2(
                    trace_device.view_mut(),
                    cumulative_sums.as_mut_ptr(),
                    DEFAULT_STREAM,
                );
            }

            unsafe {
                tracegen::ffi::core_memory_global_generate_trace_round_3(
                    trace_device.view_mut(),
                    cumulative_sums.as_ptr(),
                    nb_events,
                    DEFAULT_STREAM,
                );
            }

            let gpu_trace = trace_device.to_host();
            assert_eq!(trace, gpu_trace);
        }
    }

    #[test]
    fn test_syscall_generate_trace() {
        let mut rng = rand::thread_rng();
        let mut shard = ExecutionRecord::default();
        shard.syscall_events = (0..1000)
            .map(|_| SyscallEvent {
                shard: rng.gen_range(0..10000),
                clk: rng.gen_range(0..1000000),
                lookup_id: sp1_core_executor::events::LookupId(rng.gen_range(0..1000000)),
                syscall_id: rng.gen_range(0..256),
                arg1: rng.gen_range(0..1000000),
                arg2: rng.gen_range(0..1000000),
                nonce: rng.gen_range(0..1000000),
            })
            .collect::<Vec<_>>();

        let chip = SyscallChip::core();

        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_copy = trace.clone();
        trace_copy.values.fill(BabyBear::zero());
        let mut trace_device =
            RowMajorMatrixDevice::new(trace_copy.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let events = shard.syscall_events;
        let nb_events = events.len();
        let events = events.to_device().unwrap().as_ptr();
        unsafe {
            tracegen::ffi::core_syscall_generate_trace_round_1(
                trace_device.view_mut(),
                events,
                nb_events as u32,
                false,
                DEFAULT_STREAM,
            );
        }

        let mut cumulative_sums =
            vec![SepticCurve::<BabyBear>::default(); trace.height()].to_device().unwrap();

        unsafe {
            tracegen::ffi::core_syscall_generate_trace_round_2(
                trace_device.view_mut(),
                cumulative_sums.as_mut_ptr(),
                DEFAULT_STREAM,
            );
        }

        unsafe {
            tracegen::ffi::core_syscall_generate_trace_round_3(
                trace_device.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();
        assert_eq!(trace, gpu_trace);
    }

    #[test]
    fn test_global_generate_trace() {
        let mut rng = rand::thread_rng();
        let mut shard = ExecutionRecord::default();
        shard.global_interaction_events = (0..1000)
            .map(|_| GlobalInteractionEvent {
                message: [rng.gen_range(0..10000); 7],
                is_receive: false,
            })
            .collect::<Vec<_>>();

        let chip = GlobalChip;

        let trace: RowMajorMatrix<BabyBear> =
            chip.generate_trace(&shard, &mut ExecutionRecord::default());

        let mut trace_copy = trace.clone();
        trace_copy.values.fill(BabyBear::zero());
        let mut trace_device =
            RowMajorMatrixDevice::new(trace_copy.values.to_device().unwrap(), trace.width())
                .to_column_major();

        let og_events = shard.global_interaction_events;
        let nb_events = og_events.len();
        let events = og_events.to_device().unwrap().as_ptr();
        unsafe {
            tracegen::ffi::core_global_generate_trace_round_1(
                trace_device.view_mut(),
                events,
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let mut cumulative_sums =
            vec![SepticCurve::<BabyBear>::default(); trace.height()].to_device().unwrap();

        unsafe {
            tracegen::ffi::core_global_generate_trace_round_2(
                trace_device.view_mut(),
                cumulative_sums.as_mut_ptr(),
                DEFAULT_STREAM,
            );
        }

        unsafe {
            tracegen::ffi::core_global_generate_trace_round_3(
                trace_device.view_mut(),
                cumulative_sums.as_ptr(),
                nb_events as u32,
                DEFAULT_STREAM,
            );
        }

        let gpu_trace = trace_device.to_host();

        for j in 0..trace.height() {
            let trace_row_0 = trace.row_slice(j).to_vec();
            let gpu_trace_row_0 = gpu_trace.row_slice(j).to_vec();
            if j < og_events.len() {
                println!("event: {:?}", og_events[j]);
            }
            for i in 0..trace.width() {
                assert_eq!(
                    trace_row_0[i], gpu_trace_row_0[i],
                    "mismatch on index {} and row {}",
                    i, j
                );
            }
        }

        assert_eq!(trace, gpu_trace);
    }
}
