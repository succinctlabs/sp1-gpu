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

/// An AIR that can generate the trace on either the host or the device.
pub trait DeviceAir<F: PrimeField32>: MachineAir<F> {
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

    /// Get the height of the trace that would be generated on device.
    fn num_rows(&self, input: &Self::Record) -> Option<usize>;
}

impl DeviceAir<BabyBear> for RiscvAir<BabyBear> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        // We currently only support accelerating the `AddSubChip`.
        match self {
            RiscvAir::Add(_) => None,
            _ => Some(self.generate_trace(input, output)),
        }
    }

    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // We currently only support accelerating the `AddSubChip`.
        match self {
            RiscvAir::Add(chip) => chip.generate_trace_device(input, output, stream),
            _ => Ok(None),
        }
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        // We currently only support accelerating the `AddSubChip`.
        match self {
            RiscvAir::Add(chip) => chip.num_rows(input),
            _ => None,
        }
    }
}

impl<const D: usize> DeviceAir<BabyBear> for RecursionAir<BabyBear, D> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        // We currently do not support accelerating any chips in recursion.
        Some(self.generate_trace(input, output))
    }

    fn generate_trace_device(
        &self,
        _: &Self::Record,
        _: &mut Self::Record,
        _: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        // We currently do not support accelerating any chips in recursion.
        Ok(None)
    }

    fn num_rows(&self, _: &Self::Record) -> Option<usize> {
        // We currently do not support accelerating any chips in recursion.
        None
    }
}
