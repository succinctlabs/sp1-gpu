use crate::device::memory::ToDevice;
use crate::{device::buffer::DeviceBuffer, matrix::MatrixViewDevice};
use p3_baby_bear::BabyBear;
use p3_commit::{LagrangeSelectors, PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{extension::BinomialExtensionField, Field, TwoAdicField};
use sp1_core::air::MachineAir;

use super::ffi;

#[derive(Debug)]
#[repr(C)]
pub struct TwoAdicMultiplicativeCosetDevice<F: TwoAdicField> {
    log_n: usize,
    shift: F,
}

impl ToDevice for TwoAdicMultiplicativeCoset<BabyBear> {
    type DeviceType = TwoAdicMultiplicativeCosetDevice<BabyBear>;

    fn to_device(&self) -> Self::DeviceType {
        Self::DeviceType {
            log_n: self.log_n,
            shift: self.shift,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct LagrangeSelectorsDevice<T: Field> {
    is_first_row: DeviceBuffer<T>,
    is_last_row: DeviceBuffer<T>,
    is_transition: DeviceBuffer<T>,
    inv_zeroifier: DeviceBuffer<T>,
}

impl ToDevice for LagrangeSelectors<Vec<BabyBear>> {
    type DeviceType = LagrangeSelectorsDevice<BabyBear>;

    fn to_device(&self) -> Self::DeviceType {
        Self::DeviceType {
            is_first_row: self.is_first_row.to_device(),
            is_last_row: self.is_last_row.to_device(),
            is_transition: self.is_transition.to_device(),
            inv_zeroifier: self.inv_zeroifier.to_device(),
        }
    }
}

impl LagrangeSelectorsDevice<BabyBear> {
    pub fn to_view(&self) -> LagrangeSelectorsView<BabyBear> {
        LagrangeSelectorsView {
            is_first_row: self.is_first_row.as_ptr(),
            is_last_row: self.is_last_row.as_ptr(),
            is_transition: self.is_transition.as_ptr(),
            inv_zeroifier: self.inv_zeroifier.as_ptr(),
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct LagrangeSelectorsView<'a, T: Field> {
    is_first_row: *const T,
    is_last_row: *const T,
    is_transition: *const T,
    inv_zeroifier: *const T,
    _phantom: std::marker::PhantomData<&'a T>,
}

#[cfg(test)]
mod tests {
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_commit::{Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use sp1_core::runtime::ExecutionRecord;
    use sp1_core::utils::BabyBearPoseidon2;

    use rand::thread_rng;
    use sp1_core::stark::{quotient_values, Domain, StarkGenericConfig};
    use sp1_core::{
        air::MachineAir,
        runtime::Program,
        stark::{permutation_trace_width, ByteChip, Chip},
        utils::{log2_strict_usize, tests::FIBONACCI_ELF},
    };

    use crate::stark::ffi;
    use crate::{
        device::{buffer::DeviceBuffer, memory::ToDevice},
        matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice},
        stark::HostInteractions,
        time::CudaInstant,
    };

    type F = BabyBear;
    const D: usize = 4;
    type EF = BinomialExtensionField<F, D>;
    type SC = BabyBearPoseidon2;

    fn natural_domain_for_degree(degree: usize) -> TwoAdicMultiplicativeCoset<BabyBear> {
        TwoAdicMultiplicativeCoset {
            log_n: log2_strict_usize(degree),
            shift: F::one(),
        }
    }

    #[test]
    pub fn test_quotient_values() {
        let mut rng = thread_rng();
        let air = ByteChip::<F>::default();
        let chip = Chip::new(air);
        let program = Program::from(FIBONACCI_ELF);

        let num_rows = 1 << 16;
        let config = BabyBearPoseidon2::default();
        let pcs = config.pcs();

        let mut main = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());
        let prep = chip.generate_preprocessed_trace(&program).unwrap();

        let mut permutation_challenges = vec![EF::one(), EF::two()];
        let perm = chip.generate_permutation_trace(Some(&prep), &mut main, &permutation_challenges);

        let prep_domain = natural_domain_for_degree(prep.height());
        let degree = main.height();
        let log_degree = log2_strict_usize(degree);
        let log_quotient_degree = chip.log_quotient_degree();
        let trace_domain = natural_domain_for_degree(degree);
        let cumulative_sum = perm.row_slice(main.height() - 1).last().copied().unwrap();

        let (prep_commit, prep_data) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(pcs, vec![(prep_domain, prep)]);
        let (main_commit, main_data) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(pcs, vec![(trace_domain, main)]);
        let (perm_commit, perm_data) =
            <<SC as StarkGenericConfig>::Pcs as Pcs<
                <SC as StarkGenericConfig>::Challenge,
                <SC as StarkGenericConfig>::Challenger,
            >>::commit(pcs, vec![(trace_domain, perm.flatten_to_base())]);

        let quotient_domain =
            trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
        let preprocessed_trace_on_quotient_domain =
            <<SC as StarkGenericConfig>::Pcs as Pcs<
                <SC as StarkGenericConfig>::Challenge,
                <SC as StarkGenericConfig>::Challenger,
            >>::get_evaluations_on_domain(pcs, &prep_data, 0, quotient_domain)
            .to_row_major_matrix();
        let main_trace_on_quotient_domain =
            <<SC as StarkGenericConfig>::Pcs as Pcs<
                <SC as StarkGenericConfig>::Challenge,
                <SC as StarkGenericConfig>::Challenger,
            >>::get_evaluations_on_domain(pcs, &main_data, 0, quotient_domain)
            .to_row_major_matrix();
        let permutation_trace_on_quotient_domain =
            <<SC as StarkGenericConfig>::Pcs as Pcs<
                <SC as StarkGenericConfig>::Challenge,
                <SC as StarkGenericConfig>::Challenger,
            >>::get_evaluations_on_domain(pcs, &perm_data, 0, quotient_domain)
            .to_row_major_matrix();

        let alpha = EF::from_base_slice(&[F::one(), F::one(), F::one(), F::one()]);
        let public_values = vec![];
        let result = quotient_values::<BabyBearPoseidon2, _, _>(
            &chip,
            cumulative_sum,
            trace_domain,
            quotient_domain,
            preprocessed_trace_on_quotient_domain.clone(),
            main_trace_on_quotient_domain.clone(),
            permutation_trace_on_quotient_domain.clone(),
            &permutation_challenges,
            alpha,
            &public_values,
        );
        let selectors_device = trace_domain.selectors_on_coset(quotient_domain).to_device();

        let trace_domain_device = trace_domain.to_device();
        let quotient_domain_device = quotient_domain.to_device();

        let preprocessed_trace_on_quotient_domain_device =
            preprocessed_trace_on_quotient_domain.values.to_device();
        let preprocessed_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
            preprocessed_trace_on_quotient_domain_device,
            preprocessed_trace_on_quotient_domain.width(),
        )
        .to_column_major();
        let main_trace_on_quotient_domain_device = main_trace_on_quotient_domain.values.to_device();
        let main_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
            main_trace_on_quotient_domain_device,
            main_trace_on_quotient_domain.width(),
        )
        .to_column_major();
        let permutation_trace_on_quotient_domain_device =
            permutation_trace_on_quotient_domain.values.to_device();
        let permutation_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
            permutation_trace_on_quotient_domain_device,
            permutation_trace_on_quotient_domain.width(),
        )
        .to_column_major();
        let permutation_challenges_device = permutation_challenges.to_device();
        let public_values_device = public_values.to_device();

        let mut quotient_output = DeviceBuffer::with_capacity(quotient_domain.size());

        unsafe {
            quotient_output.set_len(quotient_domain.size());
            ffi::quotient_values(
                0,
                cumulative_sum,
                trace_domain_device,
                quotient_domain_device,
                preprocessed_trace_on_quotient_domain_device.view(),
                main_trace_on_quotient_domain_device.view(),
                permutation_trace_on_quotient_domain_device.view(),
                permutation_challenges_device.as_ptr(),
                alpha,
                public_values_device.as_ptr(),
                quotient_output.as_mut_ptr(),
                selectors_device.to_view(),
                1,
                1,
            );
        }
    }
}
