use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;
use p3_matrix::dense::RowMajorMatrix;
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_core::machine::RecursionAir;
use sp1_stark::air::MachineAir;

use crate::{
    cuda_runtime::stream::CudaStream, device::error::CudaError, matrix::ColMajorMatrixDevice,
};

/// An AIR that can generate the trace on either the host or the device.
pub trait DeviceAir<F: PrimeField32>: MachineAir<F> {
    /// Generate the trace on the host.
    ///
    /// This function returns `None` if the trace  is designed to be generated on device.
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<F>>;

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
    fn height_device(&self, input: &Self::Record) -> Option<usize>;
}

impl<F: PrimeField32> DeviceAir<F> for RiscvAir<F> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<F>> {
        Some(self.generate_trace(input, output))
    }

    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<F>>, CudaError> {
        Ok(None)
    }

    fn height_device(&self, input: &Self::Record) -> Option<usize> {
        None
    }
}

impl<const D: usize> DeviceAir<BabyBear> for RecursionAir<BabyBear, D> {
    fn generate_trace_host(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
    ) -> Option<RowMajorMatrix<BabyBear>> {
        Some(self.generate_trace(input, output))
    }

    fn generate_trace_device(
        &self,
        input: &Self::Record,
        output: &mut Self::Record,
        stream: &CudaStream,
    ) -> Result<Option<ColMajorMatrixDevice<BabyBear>>, CudaError> {
        Ok(None)
    }

    fn height_device(&self, input: &Self::Record) -> Option<usize> {
        None
    }
}
