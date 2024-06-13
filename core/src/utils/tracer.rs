use std::sync::Once;

use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

/// Initializes the tracing subscriber.
///
/// Set the `RUST_LOG` environment variable to be set to `info` or `debug`.
#[allow(dead_code)]
pub fn init_tracer() {
    INIT.call_once(|| {
        let default_filter = "off";
        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(default_filter))
            .add_directive("p3_keccak_air=off".parse().unwrap())
            .add_directive("p3_fri=off".parse().unwrap())
            .add_directive("p3_dft=off".parse().unwrap())
            .add_directive("p3_challenger=off".parse().unwrap());

        // if the TRACER_TYPE environment variable is set, use it to determine which logger to configure
        // (tracing_forest or tracing_subscriber)
        // otherwise, default to 'flat'
        let logger_type = std::env::var("TRACER_TYPE").unwrap_or_else(|_| "flat".to_string());
        match logger_type.as_str() {
            "flat" => {
                tracing_subscriber::fmt::Subscriber::builder()
                    .compact()
                    .with_file(false)
                    .with_target(false)
                    .with_thread_names(false)
                    .with_env_filter(env_filter)
                    .finish()
                    .init();
            }
            _ => {
                panic!("Invalid logger type: {}", logger_type);
            }
        }
    });
}
