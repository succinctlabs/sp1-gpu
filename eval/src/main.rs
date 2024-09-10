use anyhow::Result;
use moongate_prover::{components::GpuProverComponents};
use sp1_eval::evaluate_performance;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    evaluate_performance::<GpuProverComponents>().await
}
