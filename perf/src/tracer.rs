use std::sync::Once;

use opentelemetry::global;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::{runtime, trace, Resource};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, Registry};

static INIT: Once = Once::new();

fn build_env_filter(base: Option<EnvFilter>) -> EnvFilter {
    base.unwrap_or(EnvFilter::try_from_default_env().unwrap_or(EnvFilter::new("debug")))
        .add_directive("p3_keccak_air=off".parse().unwrap())
        .add_directive("p3_fri=off".parse().unwrap())
        .add_directive("p3_dft=off".parse().unwrap())
        .add_directive("p3_challenger=off".parse().unwrap())
        .add_directive("pprof=error".parse().unwrap())
        .add_directive("Pyroscope=error".parse().unwrap())
        .add_directive("h2=off".parse().unwrap())
        .add_directive("tower=off".parse().unwrap())
}

pub fn init(resource: Resource) {
    INIT.call_once(|| {
        global::set_text_map_propagator(TraceContextPropagator::new());

        let env_filter = build_env_filter(None);
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(opentelemetry_otlp::new_exporter().tonic())
            .with_trace_config(trace::config().with_resource(resource.clone()))
            .install_batch(runtime::Tokio)
            .unwrap();
        let telemetry = tracing_opentelemetry::layer().with_tracer(tracer).with_filter(env_filter);
        let env_filter = build_env_filter(None);
        let fmt_layer = tracing_subscriber::fmt::layer()
            .compact()
            .with_file(false)
            .with_target(true)
            .with_thread_names(false)
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stdout)
            .with_filter(env_filter);
        Registry::default().with(telemetry).with(fmt_layer).init();
    });
}
