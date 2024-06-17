use tracing::{event, span, Subscriber};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::Layer;

use super::CudaInstant;

pub struct TimeLayer;

impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for TimeLayer {
    fn on_new_span(&self, _attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(id).expect("Failed to get span");

        // Attach start time to the span
        span.extensions_mut().insert(CudaInstant::now().unwrap());
    }

    fn on_close(&self, id: span::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).expect("Failed to get span");
        let metadata = span.metadata();
        let level = metadata.level();

        // Retrieve start time and calculate the duration
        let extensions = span.extensions();
        if let Some(start_time) = extensions.get::<CudaInstant>() {
            let duration = start_time.elapsed().unwrap();
            match *level {
                tracing::Level::TRACE => event!(
                    parent: id,
                    tracing::Level::TRACE,
                    "Closed,  time: {:?}",
                    duration
                ),
                tracing::Level::DEBUG => event!(
                    parent: id,
                    tracing::Level::DEBUG,
                    "Closed,  time: {:?}",
                    duration
                ),
                tracing::Level::INFO => {
                    event!(parent: id, tracing::Level::INFO, "Closed,  time: {:?}", duration)
                }
                tracing::Level::ERROR => {
                    event!(parent: id, tracing::Level::ERROR, "Closed,  time: {:?}", duration)
                }
                _ => event!(parent: id, tracing::Level::WARN, "Closed,  time: {:?}", duration),
            }
        }
    }
}
