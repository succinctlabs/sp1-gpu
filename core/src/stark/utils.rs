use p3_commit::Pcs;
use sp1_stark::{Dom, StarkGenericConfig};

pub fn natural_domain_for_degree<SC: StarkGenericConfig>(config: &SC, degree: usize) -> Dom<SC> {
    <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::natural_domain_for_degree(config.pcs(), degree)
}
