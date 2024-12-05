use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_core::machine::RecursionAir;
use sp1_stark::air::MachineAir;

use crate::{
    cuda_runtime::stream::CudaStream, device::error::CudaError, matrix::ColMajorMatrixDevice,
};

pub mod core;
pub mod ffi;
pub mod recursion;
pub mod recursion_preprocessed;

/// An AIR that can generate the trace on either the host or the device.
pub trait DeviceAir<F: PrimeField32>: MachineAir<F> {
    /// Get the height of the trace that would be generated on device.
    fn num_rows_device(&self, input: &Self::Record) -> Option<usize> {
        self.num_rows(input)
    }

    /// Generate the trace on the host.
    ///
    /// This function returns `None` if the trace  is designed to be generated on device.
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<F>> {
        Some(self.generate_trace(input, output))
    }

    /// Generate the trace on the device.
    ///
    /// This function returns `None` if the trace is designed to be generated on host.
    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<F>>, CudaError>;
}

/// An AIR that can generate the preprocessed trace on either the host or the device.
pub trait DevicePreprocessedAir<F: PrimeField32>: MachineAir<F> {
    /// Generate the preprocessed trace on the host.
    ///
    /// This function returns `None` if the preprocessed trace is designed to be generated on device.
    fn generate_preprocessed_trace_host(
        &self,
        program: &Self::Program,
    ) -> Option<RowMajorMatrix<F>> {
        self.generate_preprocessed_trace(program)
    }

    /// Generate the preprocessed trace on the device.
    ///
    /// This function returns `None` if the preprocessed trace is designed to be generated on host.
    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<F>>, CudaError>;
}

impl DeviceAir<BabyBear> for RiscvAir<BabyBear> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        // We currently only support accelerating the `AddSubChip` and chips with global interaction.
        match self {
            // RiscvAir::Add(_) => None,
            RiscvAir::MemoryLocal(_) => None,
            RiscvAir::MemoryGlobalFinal(_) => None,
            RiscvAir::MemoryGlobalInit(_) => None,
            // RiscvAir::SyscallCore(_) => None,
            // RiscvAir::SyscallPrecompile(_) => None,
            // RiscvAir::Global(_) => None,
            _ => Some(self.generate_trace(input, output)),
        }
    }

    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // We currently only support accelerating the `AddSubChip` and chips with global interaction.
        match self {
            // RiscvAir::Add(chip) => chip.generate_trace_device(input, output, stream),
            RiscvAir::MemoryLocal(chip) => chip.generate_trace_device(input, output, stream),
            RiscvAir::MemoryGlobalFinal(chip) => chip.generate_trace_device(input, output, stream),
            RiscvAir::MemoryGlobalInit(chip) => chip.generate_trace_device(input, output, stream),
            // RiscvAir::SyscallCore(chip) => chip.generate_trace_device(input, output, stream),
            // RiscvAir::SyscallPrecompile(chip) => chip.generate_trace_device(input, output, stream),
            // RiscvAir::Global(chip) => chip.generate_trace_device(input, output, stream),
            _ => Ok(None),
        }
    }
}

impl<const D: usize> DeviceAir<BabyBear> for RecursionAir<BabyBear, D> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        match self {
            RecursionAir::BaseAlu(_) => None,
            RecursionAir::ExtAlu(_) => None,
            RecursionAir::Poseidon2Skinny(_) => None,
            RecursionAir::Poseidon2Wide(_) => None,
            RecursionAir::Select(_) => None,
            RecursionAir::FriFold(_) => None,
            RecursionAir::BatchFRI(_) => None,
            _ => Some(self.generate_trace(input, output)),
        }
    }

    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        match self {
            RecursionAir::BaseAlu(chip) => chip.generate_trace_device(input, output, stream),
            RecursionAir::ExtAlu(chip) => chip.generate_trace_device(input, output, stream),
            RecursionAir::Poseidon2Skinny(chip) => {
                chip.generate_trace_device(input, output, stream)
            }
            RecursionAir::Poseidon2Wide(chip) => chip.generate_trace_device(input, output, stream),
            RecursionAir::Select(chip) => chip.generate_trace_device(input, output, stream),
            RecursionAir::FriFold(chip) => chip.generate_trace_device(input, output, stream),
            RecursionAir::BatchFRI(chip) => chip.generate_trace_device(input, output, stream),
            _ => Ok(None),
        }
    }
}

impl DevicePreprocessedAir<BabyBear> for RiscvAir<BabyBear> {
    fn generate_preprocessed_trace_host(
        &self,
        program: &<sp1_core_machine::riscv::RiscvAir<p3_baby_bear::BabyBear> as MachineAir<
            BabyBear,
        >>::Program,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        self.generate_preprocessed_trace(program)
    }

    fn generate_preprocessed_trace_device(
        &self,
        _program: &<sp1_core_machine::riscv::RiscvAir<p3_baby_bear::BabyBear> as MachineAir<
            BabyBear,
        >>::Program,
        _stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        Ok(None)
    }
}

impl<const D: usize> DevicePreprocessedAir<BabyBear> for RecursionAir<BabyBear, D> {
    fn generate_preprocessed_trace_host(
        &self,
        program: &Self::Program,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        match self {
            RecursionAir::BaseAlu(_) => None,
            RecursionAir::ExtAlu(_) => None,
            RecursionAir::Poseidon2Skinny(_) => None,
            RecursionAir::Poseidon2Wide(_) => None,
            RecursionAir::Select(_) => None,
            _ => self.generate_preprocessed_trace(program),
        }
    }

    fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        match self {
            RecursionAir::BaseAlu(chip) => chip.generate_preprocessed_trace_device(program, stream),
            RecursionAir::ExtAlu(chip) => chip.generate_preprocessed_trace_device(program, stream),
            RecursionAir::Poseidon2Skinny(chip) => {
                chip.generate_preprocessed_trace_device(program, stream)
            }
            RecursionAir::Poseidon2Wide(chip) => {
                chip.generate_preprocessed_trace_device(program, stream)
            }
            RecursionAir::Select(chip) => chip.generate_preprocessed_trace_device(program, stream),
            _ => Ok(None),
        }
    }
}
