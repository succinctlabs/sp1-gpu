use std::borrow::Borrow;

use p3_baby_bear::BabyBear;
use p3_commit::{PolynomialSpace, TwoAdicMultiplicativeCoset};
use p3_field::{AbstractField, Field};
use sp1_stark::Com;

use crate::{
    cuda_runtime::{event::CudaEvent, stream::CudaStream},
    device::error::CudaError,
    dft::DeviceDft,
    matrix::ColMajorMatrixDevice,
    merkle_tree::MmcsCommitterAsync,
    stark::BabyBearFriConfig,
};

pub struct TwoAdicFriCommitter<SC: BabyBearFriConfig, C> {
    pub dft: DeviceDft<SC::Val>,
    pub mmcs_committer: C,
    pub log_blowup: usize,
}

impl<
        SC: BabyBearFriConfig,
        C: MmcsCommitterAsync<BabyBear, SC::ValMmcs, Matrix = ColMajorMatrixDevice<SC::Val>>,
    > TwoAdicFriCommitter<SC, C>
{
    pub fn new(log_blowup: usize) -> Self
    where
        C: Default,
    {
        Self { dft: DeviceDft::new(), mmcs_committer: C::default(), log_blowup }
    }

    pub fn mmcs_commit(
        &self,
        leaves: Vec<C::Matrix>,
        stream: &CudaStream,
    ) -> (Com<SC>, C::ProverData) {
        self.mmcs_committer.commit(leaves, stream)
    }

    pub const fn log_blowup(&self) -> usize {
        self.log_blowup
    }

    pub fn encode(
        &self,
        domain: TwoAdicMultiplicativeCoset<BabyBear>,
        matrix: &ColMajorMatrixDevice<BabyBear>,
        bit_reversed: bool,
    ) -> Result<ColMajorMatrixDevice<BabyBear>, CudaError> {
        assert_eq!(domain.size(), matrix.height());

        let shift = domain.shift.inverse();
        unsafe {
            let mut lde_mat = matrix.embed_as_blowup(self.log_blowup)?;
            self.dft.coset_lde_batch_device(
                lde_mat.view_mut(),
                self.log_blowup,
                shift,
                bit_reversed,
                lde_mat.stream(),
            )?;

            Ok(lde_mat)
        }
    }

    pub fn encode_batch<M>(
        &self,
        evaluations: &[(TwoAdicMultiplicativeCoset<BabyBear>, M, CudaEvent)],
        bit_reversed: bool,
    ) -> Result<Vec<ColMajorMatrixDevice<BabyBear>>, CudaError>
    where
        M: Borrow<C::Matrix>,
    {
        let lde_evals = evaluations
            .iter()
            .map(|(domain, matrix, event)| {
                let matrix = matrix.borrow();
                let lde = unsafe { matrix.embed_as_blowup(self.log_blowup) }?;
                Ok((*domain, lde, event))
            })
            .collect::<Result<Vec<_>, CudaError>>()?;

        lde_evals
            .into_iter()
            .map(|(domain, mut lde, event)| unsafe {
                let shift = domain.shift.inverse();
                self.dft.coset_lde_batch_device(
                    lde.view_mut(),
                    self.log_blowup,
                    shift,
                    bit_reversed,
                    lde.stream(),
                )?;

                lde.stream().record(event)?;

                Ok(lde)
            })
            .collect()
    }

    pub fn get_evaluations_on_domain(
        &self,
        src_domain: TwoAdicMultiplicativeCoset<BabyBear>,
        dst_domain: TwoAdicMultiplicativeCoset<BabyBear>,
        matrix: &ColMajorMatrixDevice<BabyBear>,
    ) -> Result<ColMajorMatrixDevice<BabyBear>, CudaError> {
        // Domain assertions for the current usage. The code is supposed to work regardless but we
        // keep them here for now since other usages are untested.
        debug_assert!(src_domain.shift.is_one());
        debug_assert_eq!(dst_domain.shift, BabyBear::generator());

        let shift = dst_domain.shift * BabyBear::generator().inverse() * src_domain.shift.inverse();
        let log_blowup = dst_domain.log_n - src_domain.log_n;
        unsafe {
            let mut lde_mat = matrix.embed_as_blowup(log_blowup)?;
            self.dft.coset_lde_batch_device(
                lde_mat.view_mut(),
                log_blowup,
                shift,
                false,
                lde_mat.stream(),
            )?;

            Ok(lde_mat)
        }
    }

    pub fn commit<M>(
        &self,
        evaluations: &[(TwoAdicMultiplicativeCoset<BabyBear>, M, CudaEvent)],
        stream: &CudaStream,
    ) -> (Com<SC>, C::ProverData)
    where
        M: Borrow<C::Matrix>,
    {
        // Encode all the matrices and register the events.
        let lde_evaluations = self.encode_batch(evaluations, true).unwrap();
        // Get the committer stream to wait for encodings to be done.
        for (_, _, event) in evaluations.iter() {
            stream.wait_event(event).unwrap();
        }
        // Commit the LDE evaluations.
        self.mmcs_commit(lde_evaluations, stream)
    }
}

#[cfg(test)]
mod tests {
    use crate::{device::memory::ToDevice, time::CudaInstant};

    use super::*;
    use p3_commit::Pcs;
    use p3_matrix::dense::RowMajorMatrix;
    use rand::thread_rng;

    use p3_field::AbstractField;
    use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

    use crate::merkle_tree::Poseidon2BabyBearCommitter;

    #[test]
    fn test_commit_device() {
        let log_blowup = 1;
        let log_degrees = [16, 10, 8];
        let columns = [100, 200, 300];

        type SC = BabyBearPoseidon2;

        let mut rng = thread_rng();

        let main_stream = CudaStream::default();

        let domains_and_traces = log_degrees
            .iter()
            .zip(columns.iter())
            .map(|(log_degree, cols)| {
                let trace = RowMajorMatrix::<BabyBear>::rand(&mut rng, 1 << log_degree, *cols);

                let domain = TwoAdicMultiplicativeCoset::<BabyBear> {
                    log_n: *log_degree,
                    shift: BabyBear::one(),
                };

                (domain, trace)
            })
            .collect::<Vec<_>>();

        let evaluations = domains_and_traces
            .iter()
            .map(|(domain, trace)| {
                let trace = trace.to_device().unwrap().to_column_major();
                (*domain, trace, CudaEvent::new().unwrap())
            })
            .collect::<Vec<_>>();

        let pcs = TwoAdicFriCommitter::<SC, Poseidon2BabyBearCommitter>::new(log_blowup);
        let time = CudaInstant::now().unwrap();
        let (commit, _) = pcs.commit(&evaluations, &main_stream);
        println!("time: {:?}", time.elapsed().unwrap());

        let sp1_config = SC::default();
        let (expected_commit, _) = <<SC as StarkGenericConfig>::Pcs as Pcs<
            <SC as StarkGenericConfig>::Challenge,
            <SC as StarkGenericConfig>::Challenger,
        >>::commit(sp1_config.pcs(), domains_and_traces);

        assert_eq!(commit, expected_commit);
    }
}
