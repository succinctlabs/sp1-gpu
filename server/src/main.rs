use std::{
    any::Any,
    collections::{hash_map::Entry, HashMap},
    sync::{Arc, Mutex, RwLock},
};

use async_trait::async_trait;
use bytes::Bytes;
use http::{
    header::{self},
    Response, StatusCode,
};
use http_body_util::Full;
use moongate_core::{
    cuda_runtime::stream::CudaStream, device::error::CudaError, utils::init_tracer,
};
use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};
use sha2::{Digest, Sha256};
use sp1_core_executor::{Program, SP1Context};
use sp1_core_machine::io::SP1Stdin;
use sp1_cuda::{
    CompressRequestPayload, ProveCoreRequestPayload, SetupRequestPayload, SetupResponsePayload,
    ShrinkRequestPayload, StatelessProveCoreRequestPayload, WrapRequestPayload,
};
use sp1_prover::{DeviceProvingKey, SP1Prover};
use sp1_stark::MachineProver;
use tower_http::catch_panic::CatchPanicLayer;
use twirp::{
    axum::{self},
    internal, Router,
};

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

struct ProvingContext {
    program: Program,
    pk_d: DeviceProvingKey<GpuProverComponents>,
}

impl ProvingContext {
    pub fn new(program: Program, pk_d: DeviceProvingKey<GpuProverComponents>) -> Self {
        Self { program, pk_d }
    }
}

struct MoongateProverServer {
    prover: Arc<Mutex<Option<SP1Prover<GpuProverComponents>>>>,
    proving_contexts: Arc<RwLock<HashMap<Vec<u8>, Mutex<ProvingContext>>>>,
}

impl MoongateProverServer {
    /// Create a new [MoongateProverServer].
    pub fn try_new() -> Result<Self, CudaError> {
        let server = Self {
            prover: Arc::new(Mutex::new(None)),
            proving_contexts: Arc::new(RwLock::new(HashMap::new())),
        };
        server.init()?;

        Ok(server)
    }

    /// Initialize the prover lazily.
    pub fn init(&self) -> Result<(), CudaError> {
        tracing::debug!("initializing proving server...");
        let prover_mutex = Arc::clone(&self.prover);

        // Check a channel to the GPU can be created and returns an error if not, instead of
        // panicking in `SP1Prover::new()`.
        CudaStream::create()?;

        std::thread::spawn(move || {
            let prover = SP1Prover::new();
            tracing::debug!("prover server initialized");
            let mut prover_lock = prover_mutex.lock().unwrap();
            *prover_lock = Some(prover);
        });

        Ok(())
    }

    fn prove_core_internal(
        &self,
        proving_context: &ProvingContext,
        stdin: &SP1Stdin,
    ) -> Result<sp1_cuda::proto::api::ProveCoreResponse, twirp::TwirpErrorResponse> {
        let result = tracing::info_span!("prove core").in_scope(|| {
            self.prover
                .lock()
                .unwrap()
                .as_ref()
                .ok_or_else(|| internal("prover not ready".to_string()))?
                .prove_core(
                    &proving_context.pk_d,
                    proving_context.program.clone(),
                    stdin,
                    gpu_prover_opts(),
                    SP1Context::default(),
                )
                .map_err(|e| internal(format!("failed to prove core {}", e)))
        })?;

        let result = tracing::info_span!("serialize proof result").in_scope(|| {
            bincode::serialize(&result).map_err(|e| internal(format!("failed to serialize {}", e)))
        })?;

        Ok(sp1_cuda::proto::api::ProveCoreResponse { result })
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

    async fn setup(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::SetupRequest,
    ) -> Result<sp1_cuda::proto::api::SetupResponse, twirp::TwirpErrorResponse> {
        let payload: SetupRequestPayload = bincode::deserialize(&req.data)
            .map_err(|e| internal(format!("failed to deserialize {}", e)))?;

        let (pk, pk_d, program, vk) = tracing::info_span!("setup").in_scope(|| {
            self.prover
                .lock()
                .unwrap()
                .as_ref()
                .ok_or_else(|| internal("prover not ready".to_string()))
                .map(|prover| prover.setup(&payload.elf))
        })?;

        self.proving_contexts
            .write()
            .map_err(|e| internal(format!("{}", e)))?
            .insert(elf_to_hash(&payload.elf), Mutex::new(ProvingContext::new(program, pk_d)));

        let response = SetupResponsePayload { pk, vk };
        let result = bincode::serialize(&response)
            .map_err(|e| internal(format!("failed to serialize {}", e)))?;

        Ok(sp1_cuda::proto::api::SetupResponse { result })
    }

    async fn prove_core(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::ProveCoreRequest,
    ) -> Result<sp1_cuda::proto::api::ProveCoreResponse, twirp::TwirpErrorResponse> {
        let proving_contexts =
            self.proving_contexts.read().map_err(|e| internal(format!("{}", e)))?;

        if proving_contexts.len() > 1 {
            return Err(internal(
                "stateless proving must be used when registering more than one program",
            ));
        }

        let proving_context = proving_contexts
            .iter()
            .next()
            .ok_or_else(|| internal("proving key and program not provided"))?
            .1
            .lock()
            .map_err(|e| internal(format!("{}", e)))?;

        let payload: ProveCoreRequestPayload = tracing::info_span!("deserializing proof request")
            .in_scope(|| {
            bincode::deserialize(&req.data)
                .map_err(|e| internal(format!("failed to deserialize {}", e)))
        })?;

        self.prove_core_internal(&proving_context, &payload.stdin)
    }

    async fn prove_core_stateless(
        &self,
        _: twirp::Context,
        req: sp1_cuda::proto::api::ProveCoreRequest,
    ) -> Result<sp1_cuda::proto::api::ProveCoreResponse, twirp::TwirpErrorResponse> {
        let mut proving_contexts =
            self.proving_contexts.write().map_err(|e| internal(format!("{}", e)))?;

        let payload: StatelessProveCoreRequestPayload =
            tracing::info_span!("deserializing proof request").in_scope(|| {
                bincode::deserialize(&req.data)
                    .map_err(|e| internal(format!("failed to deserialize {}", e)))
            })?;

        let proving_context = match proving_contexts.entry(elf_to_hash(&payload.pk.elf)) {
            Entry::Occupied(occupied) => occupied.into_mut(),
            Entry::Vacant(vacant) => {
                let program = tracing::info_span!("get program").in_scope(|| {
                    self.prover
                        .lock()
                        .unwrap()
                        .as_ref()
                        .ok_or_else(|| internal("prover not ready"))
                        .and_then(|prover| {
                            prover
                                .get_program(&payload.pk.elf)
                                .map_err(|err| internal(format!("{err}")))
                        })
                })?;
                let pk_d = tracing::info_span!("pk to device").in_scope(|| {
                    self.prover
                        .lock()
                        .unwrap()
                        .as_ref()
                        .ok_or_else(|| internal("prover not ready"))
                        .map(|prover| prover.core_prover.pk_to_device(&payload.pk.pk))
                })?;

                vacant.insert(Mutex::new(ProvingContext::new(program, pk_d)))
            }
        };

        let proving_context = proving_context.get_mut().map_err(|e| internal(format!("{}", e)))?;

        self.prove_core_internal(proving_context, &payload.stdin)
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
    println!("panic: {}", details);

    let body = serde_json::json!({
        "error": {
            "kind": "panic",
            "details": details,
        }
    });
    let body = serde_json::to_string(&body).unwrap();

    Response::builder()
        .status(StatusCode::INTERNAL_SERVER_ERROR)
        .header(header::CONTENT_TYPE, details.to_string())
        .body(Full::from(body))
        .unwrap()
}

#[tokio::main]
pub async fn main() -> eyre::Result<()> {
    init_tracer();

    let server = MoongateProverServer::try_new()?;
    let server = Arc::new(server);

    let twirp_routes =
        Router::new().nest(sp1_cuda::proto::api::SERVICE_FQN, sp1_cuda::proto::api::router(server));

    let app = Router::new()
        .nest("/twirp", twirp_routes)
        .fallback(twirp::server::not_found_handler)
        .layer(CatchPanicLayer::custom(handle_panic));

    let tcp_listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(tcp_listener, app).await?;

    Ok(())
}

fn elf_to_hash(elf: &[u8]) -> Vec<u8> {
    Sha256::digest(elf).to_vec()
}
