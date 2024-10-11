use std::{
    any::Any,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{
    header::{self},
    Response, StatusCode,
};
use http_body_util::Full;
use moongate_core::utils::init_tracer;
use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sp1_core_executor::SP1Context;
use sp1_cuda::{
    CompressRequestPayload, ProveCoreRequestPayload, ShrinkRequestPayload, WrapRequestPayload,
};
use sp1_prover::SP1Prover;
use tower_http::catch_panic::CatchPanicLayer;
use twirp::{
    axum::{self},
    internal, Router,
};

struct MoongateProverServer {
    prover: Arc<Mutex<Option<SP1Prover<GpuProverComponents>>>>,
}

impl MoongateProverServer {
    /// Create a new [MoongateProverServer].
    pub fn new() -> Self {
        let server = Self { prover: Arc::new(Mutex::new(None)) };
        server.init();
        server
    }

    /// Initialize the prover lazily.
    pub fn init(&self) {
        tracing::debug!("initializing proving server...");
        let prover_mutex = Arc::clone(&self.prover);
        std::thread::spawn(move || {
            let prover = SP1Prover::new();
            tracing::debug!("prover server initialized");
            let mut prover_lock = prover_mutex.lock().unwrap();
            *prover_lock = Some(prover);
        });
    }
}

#[async_trait]
impl sp1_cuda::proto::api::ProverService for MoongateProverServer {
    async fn ready(
        &self,
        _: twirp::Context,
        _: sp1_cuda::proto::api::ReadyRequest,
    ) -> Result<sp1_cuda::proto::api::ReadyResponse, twirp::TwirpErrorResponse> {
        Ok(sp1_cuda::proto::api::ReadyResponse { ready: self.prover.lock().unwrap().is_some() })
    }

    async fn prove_core(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::ProveCoreRequest,
    ) -> Result<sp1_cuda::proto::api::ProveCoreResponse, twirp::TwirpErrorResponse> {
        let payload: ProveCoreRequestPayload = bincode::deserialize(&req.data)
            .map_err(|e| internal(format!("failed to deserialize {}", e)))?;

        let result = self
            .prover
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| internal("prover not ready".to_string()))?
            .prove_core(&payload.pk, &payload.stdin, gpu_prover_opts(), SP1Context::default())
            .map_err(|e| internal(format!("failed to prove core {}", e)))?;

        let result = bincode::serialize(&result)
            .map_err(|e| internal(format!("failed to serialize {}", e)))?;

        Ok(sp1_cuda::proto::api::ProveCoreResponse { result })
    }

    async fn compress(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::CompressRequest,
    ) -> Result<sp1_cuda::proto::api::CompressResponse, twirp::TwirpErrorResponse> {
        let payload: CompressRequestPayload = bincode::deserialize(&req.data)
            .map_err(|e| internal(format!("failed to deserialize {}", e)))?;

        let result = self
            .prover
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| internal("prover not ready".to_string()))?
            .compress(&payload.vk, payload.proof, payload.deferred_proofs, gpu_prover_opts())
            .map_err(|e| internal(format!("failed to prove compress {}", e)))?;

        let result = bincode::serialize(&result)
            .map_err(|e| internal(format!("failed to serialize {}", e)))?;

        Ok(sp1_cuda::proto::api::CompressResponse { result })
    }

    async fn shrink(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::ShrinkRequest,
    ) -> Result<sp1_cuda::proto::api::ShrinkResponse, twirp::TwirpErrorResponse> {
        let payload: ShrinkRequestPayload = bincode::deserialize(&req.data)
            .map_err(|e| internal(format!("failed to deserialize {}", e)))?;

        let result = self
            .prover
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| internal("prover not ready".to_string()))?
            .shrink(payload.reduced_proof, gpu_prover_opts())
            .map_err(|e| internal(format!("failed to prove shrink {}", e)))?;

        let result = bincode::serialize(&result)
            .map_err(|e| internal(format!("failed to serialize {}", e)))?;

        Ok(sp1_cuda::proto::api::ShrinkResponse { result })
    }

    async fn wrap(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::WrapRequest,
    ) -> Result<sp1_cuda::proto::api::WrapResponse, twirp::TwirpErrorResponse> {
        let payload: WrapRequestPayload = bincode::deserialize(&req.data)
            .map_err(|e| internal(format!("failed to deserialize {}", e)))?;

        let result = self
            .prover
            .lock()
            .unwrap()
            .as_ref()
            .ok_or_else(|| internal("prover not ready".to_string()))?
            .wrap_bn254(payload.reduced_proof, gpu_prover_opts())
            .map_err(|e| internal(format!("failed to prove wrap {}", e)))?;

        let result = bincode::serialize(&result)
            .map_err(|e| internal(format!("failed to serialize {}", e)))?;

        Ok(sp1_cuda::proto::api::WrapResponse { result })
    }
}

fn handle_panic(err: Box<dyn Any + Send + 'static>) -> Response<Full<Bytes>> {
    let details = if let Some(s) = err.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = err.downcast_ref::<&str>() {
        s.to_string()
    } else {
        "Unknown panic message".to_string()
    };

    let body = serde_json::json!({
        "error": {
            "kind": "panic",
            "details": details,
        }
    });
    let body = serde_json::to_string(&body).unwrap();

    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Full::from(body))
        .unwrap()
}

#[tokio::main]
pub async fn main() {
    init_tracer();

    let server = MoongateProverServer::new();
    let server = Arc::new(server);

    let twirp_routes =
        Router::new().nest(sp1_cuda::proto::api::SERVICE_FQN, sp1_cuda::proto::api::router(server));

    let app = Router::new()
        .nest("/twirp", twirp_routes)
        .fallback(twirp::server::not_found_handler)
        .layer(CatchPanicLayer::custom(handle_panic));

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    if let Err(e) = axum::serve(tcp_listener, app).await {
        eprintln!("server error: {}", e);
    }
}
