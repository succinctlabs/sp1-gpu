use std::sync::Arc;

use async_trait::async_trait;
use moongate_core::utils::init_tracer;
use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sp1_core_executor::SP1Context;
use sp1_cuda::{CompressRequestPayload, ProveCoreRequestPayload};
use sp1_prover::SP1Prover;
use twirp::{axum, Router};

struct MoongateProverServer {
    prover: SP1Prover<GpuProverComponents>,
}

#[async_trait]
impl sp1_cuda::proto::api::ProverService for MoongateProverServer {
    async fn prove_core(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::ProveCoreRequest,
    ) -> Result<sp1_cuda::proto::api::ProveCoreResponse, twirp::TwirpErrorResponse> {
        let payload: ProveCoreRequestPayload = bincode::deserialize(&req.data).unwrap();
        let result = self
            .prover
            .prove_core(
                &payload.pk,
                &payload.stdin,
                gpu_prover_opts(),
                SP1Context::default(),
            )
            .unwrap();
        let result = bincode::serialize(&result).unwrap();
        Ok(sp1_cuda::proto::api::ProveCoreResponse { result })
    }

    async fn compress(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::CompressRequest,
    ) -> Result<sp1_cuda::proto::api::CompressResponse, twirp::TwirpErrorResponse> {
        let payload: CompressRequestPayload = bincode::deserialize(&req.data).unwrap();
        let result = self
            .prover
            .compress(
                &payload.vk,
                payload.proof,
                payload.deferred_proofs,
                gpu_prover_opts(),
            )
            .unwrap();
        let result = bincode::serialize(&result).unwrap();
        Ok(sp1_cuda::proto::api::CompressResponse { result })
    }
}

#[tokio::main]
pub async fn main() {
    init_tracer();

    let prover = SP1Prover::new();
    let server = MoongateProverServer { prover };
    let server = Arc::new(server);

    let twirp_routes = Router::new().nest(
        sp1_cuda::proto::api::SERVICE_FQN,
        sp1_cuda::proto::api::router(server),
    );

    let app = Router::new()
        .nest("/twirp", twirp_routes)
        .fallback(twirp::server::not_found_handler);

    println!("Listening on localhost:3000");
    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    if let Err(e) = axum::serve(tcp_listener, app).await {
        eprintln!("server error: {}", e);
    }
}
