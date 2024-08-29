use anyhow::Result;
use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sp1_eval::evaluate_performance;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = gpu_prover_opts();
    evaluate_performance::<GpuProverComponents>(opts).await
}
