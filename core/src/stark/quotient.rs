use sp1_stark::StarkGenericConfig;

use air::operation::Operation;
use p3_baby_bear::BabyBear;
use p3_commit::{LagrangeSelectors, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractExtensionField, Field, TwoAdicField};
use sp1_stark::{
    air::MachineAir, quotient_values, Chip, Dom, PackedChallenge, PcsProverData,
    ProverConstraintFolder, StarkMachine,
};

use p3_air::Air;
use p3_commit::{Pcs, PolynomialSpace};

use crate::fri::FriQueryProver;
use p3_field::AbstractField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use air::P3EvalFolder;

use std::{collections::HashMap, marker::PhantomData};

use crate::{
    device::{error::CudaError, memory::ToDevice, DeviceBuffer},
    fri::TwoAdicFriCommitter,
    matrix::ColMajorMatrixDevice,
    merkle_tree::MmcsCommitter,
    stark::ffi::quotient_gpu,
};

use super::{BabyBearFriConfig, CpuProverData, GpuMatrix, StarkProvingKeyDevice};

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
    eval_programs: HashMap<String, (Vec<Operation>, usize)>,
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
    SC: BabyBearFriConfig,
    A: for<'a> Air<P3EvalFolder<'a>> + MachineAir<SC::Val>,
{
    pub fn new(machine: &StarkMachine<SC, A>) -> Self {
        let mut eval_programs = HashMap::new();
        for chip in machine.chips() {
            let (operations, max) = air::codegen_cuda_eval(chip);
            eval_programs.insert(chip.name().to_owned(), (operations, max));
        }
        Self { eval_programs, _marker: PhantomData }
    }

    pub fn get_eval_program(&self, chip: &Chip<SC::Val, A>) -> &(Vec<Operation>, usize) {
        self.eval_programs.get(&chip.name()).unwrap()
    }

    pub fn split_evals(
        &self,
        num_chunks: usize,
        evals: &ColMajorMatrixDevice<SC::Val>,
    ) -> Result<Vec<GpuMatrix<SC::Val>>, CudaError> {
        (0..num_chunks).map(|i| evals.vertically_strided(num_chunks, i)).collect()
    }

    pub fn compute_values(
        &self,
        chip: &Chip<SC::Val, A>,
        trace_domain: Dom<SC>,
        quotient_domain: Dom<SC>,
        preprocessed_evaluations: &ColMajorMatrixDevice<SC::Val>,
        main_evaluations: &ColMajorMatrixDevice<SC::Val>,
        permutation_evaluations: &ColMajorMatrixDevice<SC::Val>,
        public_values: &DeviceBuffer<SC::Val>,
        cumulative_sums: &DeviceBuffer<SC::Challenge>,
        folding_challenge: SC::Challenge,
        permutation_challenges: &DeviceBuffer<SC::Challenge>,
    ) -> Result<DeviceQuotientValues<SC>, CudaError> {
        let stream = main_evaluations.stream();
        let (operations, memory_size) = self.get_eval_program(chip);
        let operations_device = operations.to_device_async(stream).unwrap();

        let trace_domain_generator =
            <SC::Val as TwoAdicField>::two_adic_generator(trace_domain.log_n);
        let quotient_domain_generator =
            <SC::Val as TwoAdicField>::two_adic_generator(quotient_domain.log_n);
        let quotient_flat = unsafe {
            let mut quotient_flat = ColMajorMatrixDevice::<SC::Val>::with_capacity_in(
                <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
                quotient_domain.size(),
                stream,
            )
            .unwrap();
            quotient_flat.set_max_width();
            quotient_gpu::compute_values(
                operations_device.as_ptr(),
                operations.len(),
                *memory_size,
                cumulative_sums.as_ptr(),
                trace_domain.to_device_async(stream).unwrap(),
                quotient_domain.to_device_async(stream).unwrap(),
                preprocessed_evaluations.view(),
                main_evaluations.view(),
                permutation_evaluations.view(),
                permutation_challenges.as_ptr(),
                folding_challenge,
                public_values.as_ptr(),
                trace_domain_generator,
                quotient_domain_generator,
                quotient_flat.view_mut(),
                stream.handle(),
            );
            quotient_flat
        };

        let quotient_degree = 1 << chip.log_quotient_degree();
        let quotient_chunks = self.split_evals(quotient_degree, &quotient_flat)?;
        let quotient_chunk_domains = quotient_domain.split_domains(quotient_degree);

        Ok(DeviceQuotientValues { quotient_chunks, quotient_chunk_domains })
    }

    #[allow(clippy::type_complexity)]
    pub fn generate_quotient_values<C>(
        &self,
        committer: &TwoAdicFriCommitter<SC, C>,
        chips: &[&Chip<SC::Val, A>],
        pk: &StarkProvingKeyDevice<SC, C>,
        main_traces: &[&ColMajorMatrixDevice<SC::Val>],
        domain_and_permutation_traces: &[(Dom<SC>, ColMajorMatrixDevice<SC::Val>)],
        permutation_challenges: &[SC::Challenge],
        folding_challenge: SC::Challenge,
        public_values: &[SC::Val],
        cumulative_sums: &[Vec<SC::Challenge>],
    ) -> Result<Vec<DeviceQuotientValues<SC>>, CudaError>
    where
        C: MmcsCommitter<SC::Val, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
        C: FriQueryProver<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>
            + 'static
            + Send
            + Sync
            + Default,
    {
        let mut results = Vec::with_capacity(chips.len());

        let permutation_challenges_device = permutation_challenges.to_device().unwrap();
        let public_values_device = public_values.to_device().unwrap();

        let evaluations = chips
            .iter()
            .enumerate()
            .map(|(i, chip)| {
                let (trace_domain, permutation_trace) = &domain_and_permutation_traces[i];
                let trace_domain = *trace_domain;

                let stream = permutation_trace.stream();

                // Get the quotient domain.
                let log_quotient_degree = chip.log_quotient_degree();
                let quotient_domain =
                    trace_domain.create_disjoint_domain(trace_domain.size() << log_quotient_degree);
                // Compute the evaluations of the traces on the quotient domain.
                let preprocessed_on_quotient_domain = pk
                    .chip_ordering
                    .get(&chip.name())
                    .map(|&index| {
                        pk.traces[index].to_device_async(stream).unwrap().to_column_major()
                    })
                    .map(|trace| {
                        committer.get_evaluations_on_domain(trace_domain, quotient_domain, &trace)
                    })
                    .transpose()?;
                let preprocessed_on_quotient_domain =
                    preprocessed_on_quotient_domain.unwrap_or_else(ColMajorMatrixDevice::null);

                let main_on_quotient_domain = committer.get_evaluations_on_domain(
                    trace_domain,
                    quotient_domain,
                    main_traces[i],
                )?;
                let perm_on_quotient_domain = committer.get_evaluations_on_domain(
                    trace_domain,
                    quotient_domain,
                    permutation_trace,
                )?;
                Ok((
                    trace_domain,
                    quotient_domain,
                    preprocessed_on_quotient_domain,
                    main_on_quotient_domain,
                    perm_on_quotient_domain,
                ))
            })
            .collect::<Result<Vec<_>, CudaError>>()?;

        for (i, (chip, evaluations)) in chips.iter().zip(evaluations).enumerate() {
            let log_quotient_degree = chip.log_quotient_degree();
            let (
                trace_domain,
                quotient_domain,
                preprocessed_on_quotient_domain,
                main_on_quotient_domain,
                perm_on_quotient_domain,
            ) = evaluations;

            let stream = main_on_quotient_domain.stream();

            // Move data to device and get generator powers.
            let cumulative_sums_device =
                cumulative_sums[i].as_slice().to_device_async(stream).unwrap();
            let trace_domain_device = trace_domain.to_device_async(stream).unwrap();
            let quotient_domain_device = quotient_domain.to_device_async(stream).unwrap();
            let (operations, memory_size) = self.get_eval_program(chip);
            let operations_device = operations.to_device_async(stream).unwrap();
            let trace_domain_generator =
                <SC::Val as TwoAdicField>::two_adic_generator(trace_domain.log_n);
            let quotient_domain_generator =
                <SC::Val as TwoAdicField>::two_adic_generator(quotient_domain.log_n);

            // Compute quotient values.
            let quotient_flat = unsafe {
                let mut quotient_flat = ColMajorMatrixDevice::<SC::Val>::with_capacity_in(
                    <SC::Challenge as AbstractExtensionField<SC::Val>>::D,
                    quotient_domain.size(),
                    stream,
                )
                .unwrap();
                quotient_flat.set_max_width();
                quotient_gpu::compute_values(
                    operations_device.as_ptr(),
                    operations.len(),
                    *memory_size,
                    cumulative_sums_device.as_ptr(),
                    trace_domain_device,
                    quotient_domain_device,
                    preprocessed_on_quotient_domain.view(),
                    main_on_quotient_domain.view(),
                    perm_on_quotient_domain.view(),
                    permutation_challenges_device.as_ptr(),
                    folding_challenge,
                    public_values_device.as_ptr(),
                    trace_domain_generator,
                    quotient_domain_generator,
                    quotient_flat.view_mut(),
                    stream.handle(),
                );
                quotient_flat
            };

            let quotient_degree = 1 << log_quotient_degree;
            let quotient_chunks = self.split_evals(quotient_degree, &quotient_flat)?;
            let quotient_chunk_domains = quotient_domain.split_domains(quotient_degree);

            results.push(DeviceQuotientValues { quotient_chunks, quotient_chunk_domains });
        }

        Ok(results)
    }
}

impl ToDevice for TwoAdicMultiplicativeCoset<BabyBear> {
    type DeviceType = TwoAdicMultiplicativeCosetDevice<BabyBear>;

    fn to_device_async(
        &self,
        _stream: &crate::cuda_runtime::stream::CudaStream,
    ) -> Result<Self::DeviceType, CudaError> {
        Ok(Self::DeviceType { log_n: self.log_n, shift: self.shift })
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

    fn to_device_async(
        &self,
        stream: &crate::cuda_runtime::stream::CudaStream,
    ) -> Result<Self::DeviceType, CudaError> {
        Ok(Self::DeviceType {
            is_first_row: self.is_first_row.to_device_async(stream)?,
            is_last_row: self.is_last_row.to_device_async(stream)?,
            is_transition: self.is_transition.to_device_async(stream)?,
            inv_zeroifier: self.inv_zeroifier.to_device_async(stream)?,
        })
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
    SC: BabyBearFriConfig,
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
        cumulative_sums: &[SC::Challenge],
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
            cumulative_sums,
            trace_domain,
            quotient_domain,
            Some(prep_on_quotient_domain),
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

        QuotientValues { quotient_chunks, quotient_chunk_domains }
    }
}

impl<SC, A> Default for CpuQuotientValuesGenerator<SC, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

#[cfg(test)]
mod tests {
    use crate::{cuda_runtime::ffi::cuda_device_synchronize, stark::BabyBearPoseidon2};
    use itertools::Itertools;
    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use p3_commit::{Pcs, PolynomialSpace, TwoAdicMultiplicativeCoset};
    use p3_field::{
        extension::BinomialExtensionField, AbstractExtensionField, AbstractField, TwoAdicField,
    };
    use p3_matrix::{dense::RowMajorMatrix, Matrix};
    use sp1_core_executor::{programs::tests::FIBONACCI_ELF, Program};
    use sp1_core_machine::{riscv::RiscvAir, utils::log2_strict_usize};
    use sp1_stark::{
        air::{MachineAir, SP1_PROOF_NUM_PV_ELTS},
        PackedChallenge, StarkGenericConfig,
    };

    use rand::thread_rng;

    use tracing::{debug, info};

    use crate::{
        cuda_runtime::ffi::DEFAULT_STREAM,
        device::memory::{ToDevice, ToHost},
        matrix::{ColMajorMatrixDevice, RowMajorMatrixDevice},
        stark::{ffi::quotient_gpu, quotient::quotient_values},
        utils::init_tracer,
    };

    type F = BabyBear;
    const D: usize = 4;
    type EF = BinomialExtensionField<F, D>;
    type SC = BabyBearPoseidon2;

    fn natural_domain_for_degree(degree: usize) -> TwoAdicMultiplicativeCoset<BabyBear> {
        TwoAdicMultiplicativeCoset { log_n: log2_strict_usize(degree), shift: F::one() }
    }

    #[test]
    pub fn test_quotient_values() {
        let mut rng = thread_rng();
        init_tracer();

        let config = BabyBearPoseidon2::compressed();
        let machine = RiscvAir::machine(config);
        let chips = machine.chips();

        for (i, chip) in chips.iter().enumerate() {
            if chip.name() != "CPU" {
                continue;
            }

            info!("Chip: {}", chip.name());
            info!("Id: {}", i);

            let program = Program::from(FIBONACCI_ELF).unwrap();
            let config = BabyBearPoseidon2::default();
            let pcs = config.pcs();

            let prep = chip.generate_preprocessed_trace(&program);
            let num_rows = if let Some(prep) = prep.as_ref() { prep.height() } else { 1 << 21 };

            let main = RowMajorMatrix::<F>::rand(&mut rng, num_rows, chip.width());

            let permutation_challenges =
                vec![EF::one(), EF::two(), EF::two() + EF::one(), EF::two() + EF::two()];
            let packed_permutation_challenges = permutation_challenges
                .iter()
                .map(|c| PackedChallenge::<SC>::from_f(*c))
                .collect::<Vec<_>>();
            let (perm, global_cumulative_sum, local_cumulative_sum) =
                chip.generate_permutation_trace(prep.as_ref(), &main, &permutation_challenges);

            let degree = main.height();
            let log_degree = log2_strict_usize(degree);
            let log_quotient_degree = chip.log_quotient_degree();
            let trace_domain = natural_domain_for_degree(degree);
            let cumulative_sums = vec![global_cumulative_sum, local_cumulative_sum];

            // Calculate evaluations on quotient domain.

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
                &cumulative_sums,
                trace_domain,
                quotient_domain,
                Some(preprocessed_trace_on_quotient_domain.clone()),
                main_trace_on_quotient_domain.clone(),
                permutation_trace_on_quotient_domain.clone(),
                &packed_permutation_challenges,
                alpha,
                &public_values,
            );
            let result_flat = RowMajorMatrix::new_col(result).flatten_to_base::<BabyBear>();
            info!("> CPU Time: {:?} ms", start.elapsed().as_millis());
            let trace_domain_generator = BabyBear::two_adic_generator(trace_domain.log_n);
            let quotient_domain_generator = BabyBear::two_adic_generator(quotient_domain.log_n);

            let trace_domain_device = trace_domain.to_device().unwrap();
            let quotient_domain_device = quotient_domain.to_device().unwrap();

            let preprocessed_trace_on_quotient_domain_device =
                preprocessed_trace_on_quotient_domain.values.to_device().unwrap();
            let preprocessed_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
                preprocessed_trace_on_quotient_domain_device,
                preprocessed_trace_on_quotient_domain.width(),
            )
            .to_column_major();

            let main_trace_on_quotient_domain_device =
                main_trace_on_quotient_domain.values.to_device().unwrap();
            let main_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
                main_trace_on_quotient_domain_device,
                main_trace_on_quotient_domain.width(),
            )
            .to_column_major();

            let permutation_trace_on_quotient_domain_device =
                permutation_trace_on_quotient_domain.values.to_device().unwrap();
            let permutation_trace_on_quotient_domain_device = RowMajorMatrixDevice::new(
                permutation_trace_on_quotient_domain_device,
                permutation_trace_on_quotient_domain.width(),
            )
            .to_column_major();
            let permutation_challenges_device = permutation_challenges.to_device().unwrap();
            let public_values_device = public_values.to_device().unwrap();
            let cumulative_sums_device = cumulative_sums.to_device().unwrap();

            let mut quotient_output =
                ColMajorMatrixDevice::with_capacity(D, quotient_domain.size()).unwrap();

            let (operations, expr_ctr) = air::codegen_cuda_eval(chip);
            let operations_device = operations.to_device().unwrap();
            info!("> Eval Program Len: {}", operations.len());
            info!("> Eval Program Register Count: {}", expr_ctr);
            // info!("> Eval Program: {:#?}", operations);

            let start = std::time::Instant::now();
            unsafe {
                quotient_output.set_max_width();
                quotient_gpu::compute_values(
                    operations_device.as_ptr(),
                    operations.len(),
                    expr_ctr,
                    cumulative_sums_device.as_ptr(),
                    trace_domain_device,
                    quotient_domain_device,
                    preprocessed_trace_on_quotient_domain_device.view(),
                    main_trace_on_quotient_domain_device.view(),
                    permutation_trace_on_quotient_domain_device.view(),
                    permutation_challenges_device.as_ptr(),
                    alpha,
                    public_values_device.as_ptr(),
                    trace_domain_generator,
                    quotient_domain_generator,
                    quotient_output.view_mut(),
                    DEFAULT_STREAM,
                );
                cuda_device_synchronize();
            }
            info!("> GPU Time: {:?} ms", start.elapsed().as_millis());

            let data = quotient_output.to_host();

            // for (exp, res) in result_flat.values.into_iter().zip_eq(data.values) {
            //     assert_eq!(exp, res, "failed at index {}", i);
            // }
        }
    }
}
