use air::operation::Operation;
use p3_baby_bear::BabyBear;
use p3_commit::{LagrangeSelectors, TwoAdicMultiplicativeCoset};
use p3_field::AbstractExtensionField;
use p3_field::{Field, TwoAdicField};

use p3_air::Air;
use p3_commit::{Pcs, PolynomialSpace};

use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use air::P3EvalFolder;

use std::collections::HashMap;
use std::marker::PhantomData;

use sp1_core::stark::{quotient_values, PcsProverData, StarkMachine};
use sp1_core::{
    air::MachineAir,
    stark::{Chip, Dom, PackedChallenge, ProverConstraintFolder, StarkGenericConfig},
};

use crate::device::buffer::DeviceBuffer;
use crate::device::error::CudaError;
use crate::device::memory::ToDevice;
use crate::device::CudaSync;
use crate::matrix::ColMajorMatrixDevice;
use crate::stark::ffi::quotient_gpu;
use crate::time::CudaInstant;

use super::{BabyBearPoseidon2Config, CpuProverData, GpuMatrix};

#[derive(Clone)]
pub struct QuotientValues<SC: StarkGenericConfig> {
    pub quotient_chunks: Vec<RowMajorMatrix<SC::Val>>,
    pub quotient_chunk_domains: Vec<Dom<SC>>,
}

pub struct DeviceQuotientValues<SC: StarkGenericConfig> {
    pub quotient_chunks: Vec<GpuMatrix<SC::Val>>,
    pub quotient_chunk_domains: Vec<Dom<SC>>,
}

#[derive(Clone, Debug)]
pub struct DeviceQuotientValuesGenerator<SC, A> {
    chip_data: HashMap<String, (usize, Vec<Operation>)>,
    _marker: PhantomData<(SC, A)>,
}

#[derive(Clone, Copy, Debug)]
pub struct CpuQuotientValuesGenerator<SC, A>(PhantomData<(SC, A)>);

#[derive(Debug)]
#[repr(C)]
pub struct TwoAdicMultiplicativeCosetDevice<F: TwoAdicField> {
    log_n: usize,
    shift: F,
}

impl<SC, A> DeviceQuotientValuesGenerator<SC, A>
where
    SC: BabyBearPoseidon2Config,
    A: for<'a> Air<P3EvalFolder<'a>> + MachineAir<SC::Val>,
{
    pub fn new(machine: &StarkMachine<SC, A>) -> Self {
        let mut chip_data = HashMap::new();
        for (i, chip) in machine.chips().iter().enumerate() {
            let (operations, _) = air::codegen_cuda_eval(chip);
            chip_data.insert(chip.name().to_owned(), (i, operations));
        }
        Self {
            chip_data,
            _marker: PhantomData,
        }
    }

    pub fn chip_data(&self, chip: &Chip<SC::Val, A>) -> &(usize, Vec<Operation>) {
        self.chip_data.get(&chip.name()).unwrap()
    }

    pub fn get_evaluations_on_subdomain(
        &self,
        mut lde: ColMajorMatrixDevice<SC::Val>,
        domain: Dom<SC>,
        is_bit_reversed: bool,
    ) -> Result<CudaSync<ColMajorMatrixDevice<SC::Val>>, CudaError> {
        assert_eq!(domain.shift, SC::Val::generator());
        assert_eq!(
            lde.height(),
            domain.size(),
            "Currently, only supports the full domain"
        );
        if is_bit_reversed {
            lde.bit_reverse_rows()?;
        }

        CudaSync::new(lde)
    }

    pub fn split_evals(
        &self,
        num_chunks: usize,
        evals: &ColMajorMatrixDevice<SC::Val>,
    ) -> Result<Vec<GpuMatrix<SC::Val>>, CudaError> {
        (0..num_chunks)
            .map(|i| CudaSync::new(evals.vertically_strided(num_chunks, i)?))
            .collect()
    }

    pub fn generate_quotient_values(
        &self,
        chip: &Chip<SC::Val, A>,
        trace_domain: Dom<SC>,
        preprocessed_lde: Option<ColMajorMatrixDevice<SC::Val>>,
        main_lde: ColMajorMatrixDevice<SC::Val>,
        permutation_lde: ColMajorMatrixDevice<SC::Val>,
        permutation_challenges: &[SC::Challenge],
        folding_challenge: SC::Challenge,
        public_values: &[SC::Val],
        cumulative_sum: SC::Challenge,
    ) -> Result<DeviceQuotientValues<SC>, CudaError> {
        let time = CudaInstant::now()?;
        let log_quotient_degree = chip.log_quotient_degree();

        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() << log_quotient_degree);

        let preprocessed_lde = preprocessed_lde.unwrap_or_else(|| {
            let mat = RowMajorMatrix::new_col(vec![SC::Val::zero(); quotient_domain.size()]);
            mat.to_device().to_column_major()
        });
        let elapsed = time.elapsed()?;
        println!("Time to get preprocessed: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let prep_on_quotient_domain =
            self.get_evaluations_on_subdomain(preprocessed_lde, quotient_domain, true)?;
        let elapsed = time.elapsed()?;
        println!("Time to get evaluations on preprocessed: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let main_on_quotient_domain =
            self.get_evaluations_on_subdomain(main_lde, quotient_domain, false)?;
        let elapsed = time.elapsed()?;
        println!("Time to get evaluations on main: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let perm_on_quotient_domain =
            self.get_evaluations_on_subdomain(permutation_lde, quotient_domain, false)?;
        let elapsed = time.elapsed()?;
        println!("Time to get evaluations on permutation: {:?}", elapsed);

        // let time = CudaInstant::now()?;
        // // Compute the quotient values.
        // let (operations, _) = air::codegen_cuda_eval(chip);
        // let elapsed = time.elapsed()?;
        // println!("Time to generate operations: {:?}", elapsed);

        // let time = CudaInstant::now()?;
        // let operations_device = operations.to_device();
        // let elapsed = time.elapsed()?;
        // println!("Time to copy operations: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let mut quotient_flat = ColMajorMatrixDevice::<SC::Val>::with_capacity(
            <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
            quotient_domain.size(),
        );
        let elapsed = time.elapsed()?;
        println!("Time to allocate quotient values: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let permutation_challenges_device = permutation_challenges.to_device();
        let elapsed = time.elapsed()?;
        println!("Time to copy permutation challenges: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let public_values_device = public_values.to_device();
        let elapsed = time.elapsed()?;
        println!("Time to copy public values: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let trace_domain_device = trace_domain.to_device();
        let elapsed = time.elapsed()?;
        println!("Time to copy trace domain: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let quotient_domain_device = quotient_domain.to_device();
        let elapsed = time.elapsed()?;
        println!("Time to copy quotient domain: {:?}", elapsed);
        let time = std::time::Instant::now();
        let (chip_id, operations) = self.chip_data(chip);
        let operations_device = operations.to_device();
        let elapsed = time.elapsed();
        println!("Time to get operations and id: {:?}", elapsed);

        let time = std::time::Instant::now();
        let selectors = trace_domain.selectors_on_coset(quotient_domain);
        let elapsed = time.elapsed();
        println!("Time to get selectors: {:?}", elapsed);
        let time = CudaInstant::now()?;
        let selectors_device = selectors.to_device();
        let elapsed = time.elapsed()?;
        println!("Time to copy selectors: {:?}", elapsed);

        let time = CudaInstant::now()?;
        unsafe {
            quotient_flat.set_max_width();
            quotient_gpu::compute_values(
                *chip_id,
                operations_device.as_ptr(),
                operations.len(),
                cumulative_sum,
                trace_domain_device,
                quotient_domain_device,
                prep_on_quotient_domain.view(),
                main_on_quotient_domain.view(),
                perm_on_quotient_domain.view(),
                permutation_challenges_device.as_ptr(),
                folding_challenge,
                public_values_device.as_ptr(),
                selectors_device.to_view(),
                quotient_flat.view_mut(),
                quotient_domain.size().div_ceil(512),
                512,
            );
        }
        let elapsed = time.elapsed()?;
        println!("Time to compute quotient values: {:?}", elapsed);

        let time = CudaInstant::now()?;
        let quotient_degree = 1 << log_quotient_degree;
        let quotient_chunks = self.split_evals(quotient_degree, &quotient_flat).unwrap();
        let quotient_chunk_domains = quotient_domain.split_domains(quotient_degree);
        let elapsed = time.elapsed()?;
        println!("Time to split quotient values: {:?}", elapsed);

        Ok(DeviceQuotientValues {
            quotient_chunks,
            quotient_chunk_domains,
        })
    }
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

impl<SC, A> CpuQuotientValuesGenerator<SC, A>
where
    SC: BabyBearPoseidon2Config,
    A: for<'a> Air<ProverConstraintFolder<'a, SC>> + MachineAir<SC::Val>,
{
    pub fn get_evaluations_on_domain(
        &self,
        config: &SC,
        prover_data: (usize, &PcsProverData<SC>),
        domain: Dom<SC>,
    ) -> RowMajorMatrix<SC::Val> {
        let (index, data) = prover_data;
        <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::get_evaluations_on_domain(
            config.pcs(),
            data,
            index,
            domain,
        )
        .to_row_major_matrix()
    }

    pub fn generate_quotient_values(
        &self,
        config: &SC,
        chip: &Chip<SC::Val, A>,
        trace_domain: Dom<SC>,
        preprocessed_data: (Option<usize>, &PcsProverData<SC>),
        main_data: (usize, &CpuProverData<SC>),
        permutation_data: (usize, &CpuProverData<SC>),
        permutation_challenges: &[SC::Challenge],
        folding_challenge: SC::Challenge,
        public_values: &[SC::Val],
        cumulative_sum: SC::Challenge,
    ) -> QuotientValues<SC> {
        let log_quotient_degree = chip.log_quotient_degree();

        let quotient_domain =
            trace_domain.create_disjoint_domain(trace_domain.size() << log_quotient_degree);

        // Get the evaluations on the quotient domain.
        let prep_on_quotient_domain = preprocessed_data
            .0
            .map(|index| {
                self.get_evaluations_on_domain(
                    config,
                    (index, preprocessed_data.1),
                    quotient_domain,
                )
            })
            .unwrap_or_else(|| {
                RowMajorMatrix::new_col(vec![SC::Val::zero(); quotient_domain.size()])
            });

        let main_on_quotient_domain =
            self.get_evaluations_on_domain(config, (main_data.0, main_data.1), quotient_domain);
        let perm_on_quotient_domain = self.get_evaluations_on_domain(
            config,
            (permutation_data.0, permutation_data.1),
            quotient_domain,
        );

        let packed_perm_challenges = permutation_challenges
            .iter()
            .map(|c| PackedChallenge::<SC>::from_f(*c))
            .collect::<Vec<_>>();

        // Calculate the quotient values.
        let quotient_values = quotient_values(
            chip,
            cumulative_sum,
            trace_domain,
            quotient_domain,
            prep_on_quotient_domain,
            main_on_quotient_domain,
            perm_on_quotient_domain,
            &packed_perm_challenges,
            folding_challenge,
            public_values,
        );

        // Flatten and split to create the traces.
        let quotient_flat = RowMajorMatrix::new_col(quotient_values).flatten_to_base();
        let quotient_degree = 1 << log_quotient_degree;
        let quotient_chunks = quotient_domain.split_evals(quotient_degree, quotient_flat);
        let quotient_chunk_domains = quotient_domain.split_domains(quotient_degree);

        QuotientValues {
            quotient_chunks,
            quotient_chunk_domains,
        }
    }
}

impl<SC, A> Default for CpuQuotientValuesGenerator<SC, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_commit::{Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{AbstractExtensionField, AbstractField};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use sp1_core::air::SP1_PROOF_NUM_PV_ELTS;
    use sp1_core::utils::BabyBearPoseidon2;

    use rand::thread_rng;
    use sp1_core::stark::{quotient_values, RiscvAir, StarkGenericConfig};
    use sp1_core::{
        air::MachineAir,
        runtime::Program,
        utils::{log2_strict_usize, tests::FIBONACCI_ELF},
    };

    use crate::device::memory::ToHost;
    use crate::matrix::ColMajorMatrixDevice;
    use crate::stark::ffi::quotient_gpu;
    use crate::{device::memory::ToDevice, matrix::RowMajorMatrixDevice};

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

        let config = BabyBearPoseidon2::default();
        let machine = RiscvAir::machine(config);
        let chips = machine.chips();

        for (i, chip) in chips.iter().enumerate() {
            if chip.name() == "Program"
                || chip.name() == "Bn254AddAssign"
                || chip.name() == "MemoryProgram"
                || chip.name() == "Byte"
            {
                continue;
            }
            println!("Chip: {}", chip.name());
            println!("Id: {}", i);
            let program = Program::from(FIBONACCI_ELF);
            let num_rows = 1 << 14;
            let config = BabyBearPoseidon2::default();
            let pcs = config.pcs();

            let main = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());
            let prep = chip.generate_preprocessed_trace(&program);

            let permutation_challenges = vec![EF::one(), EF::two()];
            let perm =
                chip.generate_permutation_trace(prep.as_ref(), &main, &permutation_challenges);

            let degree = main.height();
            let log_degree = log2_strict_usize(degree);
            let log_quotient_degree = chip.log_quotient_degree();
            let trace_domain = natural_domain_for_degree(degree);
            let cumulative_sum = perm.row_slice(main.height() - 1).last().copied().unwrap();

            let (_, main_data) = <<SC as StarkGenericConfig>::Pcs as Pcs<
                <SC as StarkGenericConfig>::Challenge,
                <SC as StarkGenericConfig>::Challenger,
            >>::commit(pcs, vec![(trace_domain, main)]);
            let (_, perm_data) =
                <<SC as StarkGenericConfig>::Pcs as Pcs<
                    <SC as StarkGenericConfig>::Challenge,
                    <SC as StarkGenericConfig>::Challenger,
                >>::commit(pcs, vec![(trace_domain, perm.flatten_to_base())]);

            let quotient_domain =
                trace_domain.create_disjoint_domain(1 << (log_degree + log_quotient_degree));
            let preprocessed_trace_on_quotient_domain = if let Some(prep) = prep {
                let prep_domain = natural_domain_for_degree(prep.height());
                let (_, prep_data) = <<SC as StarkGenericConfig>::Pcs as Pcs<
                    <SC as StarkGenericConfig>::Challenge,
                    <SC as StarkGenericConfig>::Challenger,
                >>::commit(pcs, vec![(prep_domain, prep)]);
                <<SC as StarkGenericConfig>::Pcs as Pcs<
                    <SC as StarkGenericConfig>::Challenge,
                    <SC as StarkGenericConfig>::Challenger,
                >>::get_evaluations_on_domain(pcs, &prep_data, 0, quotient_domain)
                .to_row_major_matrix()
            } else {
                RowMajorMatrix::new_col(vec![BabyBear::zero(); quotient_domain.size() * 4])
            };

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
            let public_values = [F::zero(); SP1_PROOF_NUM_PV_ELTS * 2].to_vec();

            let start = std::time::Instant::now();
            let result = quotient_values::<BabyBearPoseidon2, _, _>(
                chip,
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
            let result_flat = RowMajorMatrix::new_col(result).flatten_to_base::<BabyBear>();
            println!("> CPU Time: {:?} ms", start.elapsed().as_millis());
            let selectors = trace_domain.selectors_on_coset(quotient_domain);
            let selectors_device = selectors.to_device();

            let trace_domain_device = trace_domain.to_device();
            let quotient_domain_device = quotient_domain.to_device();

            let preprocessed_trace_on_quotient_domain_device =
                preprocessed_trace_on_quotient_domain.values.to_device();
            let preprocessed_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
                preprocessed_trace_on_quotient_domain_device,
                preprocessed_trace_on_quotient_domain.width(),
            )
            .to_column_major();
            let main_trace_on_quotient_domain_device =
                main_trace_on_quotient_domain.values.to_device();
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

            let mut quotient_output =
                ColMajorMatrixDevice::with_capacity(D, quotient_domain.size());

            let (operations, expr_ctr) = air::codegen_cuda_eval(chip);
            let operations_device = operations.to_device();
            println!("> Eval Program Len: {}", operations.len());
            println!("> Eval Program Register Count: {}", expr_ctr);

            let start = std::time::Instant::now();
            unsafe {
                quotient_output.set_max_width();
                quotient_gpu::compute_values(
                    i,
                    operations_device.as_ptr(),
                    operations.len(),
                    cumulative_sum,
                    trace_domain_device,
                    quotient_domain_device,
                    preprocessed_trace_on_quotient_domain_device.view(),
                    main_trace_on_quotient_domain_device.view(),
                    permutation_trace_on_quotient_domain_device.view(),
                    permutation_challenges_device.as_ptr(),
                    alpha,
                    public_values_device.as_ptr(),
                    selectors_device.to_view(),
                    quotient_output.view_mut(),
                    num_rows / 512 * 2,
                    512,
                );
            }
            let data = quotient_output.to_host();
            println!("> GPU Time: {:?} ms", start.elapsed().as_millis());

            for (exp, res) in result_flat.values.into_iter().zip_eq(data.values) {
                assert_eq!(exp, res, "failed at index {}", i);
            }

            // for i in 0..result.len() {
            //     assert_eq!(data[i], result[i], "failed at index {}", i);
            // }
        }
    }
}
