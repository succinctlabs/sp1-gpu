use std::{future::Future, sync::Arc};

use moongate_gas::{make_measurement, Stage};
use regex::Regex;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::SP1Prover;

use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};

use moongate_gas::report::write_measurements_to_csv;

use clap::{Parser, ValueEnum};

use eyre::{eyre, Report, Result};

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
use sp1_stark::{SP1CoreOpts, SP1ProverOpts};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, SimplexStream, WriteHalf};
use tokio_stream::{wrappers::LinesStream, StreamExt};
use tokio_util::{bytes::Bytes, io::SyncIoBridge};
use tracing_subscriber::{fmt::writer::OptionalWriter, reload};

use stream_cancel::{StreamExt as _, Tripwire};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    pub s3_dir: String,
}

#[derive(Clone, Debug, ValueEnum)]
enum Program {
    Fibonacci,
    Loop,
    Sha2Chain,
    Tendermint,
    Reth,
    KeyspaceRecord,
    KeyspaceBatcher,
    All,
}

#[derive(Clone, Debug, ValueEnum)]
enum Trace {
    Nvtx,
    Telemetry,
}

const BUCKET: &'static str = "sp1-testing-suite";

fn setup_logger() -> impl Fn(Option<WriteHalf<SimplexStream>>) {
    use std::sync::Mutex;

    use tracing::Level;
    use tracing_subscriber::{filter, fmt, prelude::*, reload};

    let stdout_layer = fmt::Layer::default()
        .with_filter(filter::Targets::new().with_target("sp1_gas", Level::DEBUG));
    let file_layer = fmt::Layer::default()
        .with_ansi(false)
        .with_writer(Mutex::new(OptionalWriter::none()))
        .with_filter(filter::Targets::new().with_target("sp1_core_machine::utils", Level::DEBUG));
    let (file_layer, reload_handle) = reload::Layer::new(file_layer);

    tracing_subscriber::registry().with(file_layer).with(stdout_layer).init();

    move |maybe_writer: Option<WriteHalf<SimplexStream>>| {
        reload_handle
            .modify(|layer| {
                *layer.inner_mut().writer_mut() = Mutex::new(
                    maybe_writer
                        .map(|writer| OptionalWriter::some(SyncIoBridge::new(writer)))
                        .into(),
                );
            })
            .unwrap();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let re = Regex::new(r"proving shard (\d+) took (\d+) ns: (.*)").unwrap();

    let reload_logger = setup_logger();

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = aws_sdk_s3::Client::new(&config);

    let ls = client
        .list_objects()
        .bucket(BUCKET)
        .prefix(format!("{}/", args.s3_dir))
        .delimiter("/")
        .send()
        .await?;
    assert!(!ls.is_truncated().unwrap_or_default(), "s3 directory listing is truncated");

    let programs = ls
        .common_prefixes()
        .iter()
        .filter_map(|o| o.prefix().map(|x| x[..x.len() - 1].to_owned()))
        .collect::<Vec<_>>();

    println!("{programs:?}");

    // let programs = vec!["v4/vector-16434-16718"];

    let get_object = |key: String| async {
        Ok::<_, Report>(
            client
                .get_object()
                .bucket(BUCKET)
                .key(key)
                .response_content_type("application/octet-stream")
                .send()
                .await?
                .body
                .collect()
                .await?
                .into_bytes(),
        )
    };

    let prover = Arc::new(SP1Prover::<GpuProverComponents>::new());
    let opts = gpu_prover_opts();

    // let mut measurements = vec![];
    let mut out = tokio::fs::File::create(format!(
        "out-{}.txt",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    ))
    .await?;
    for key in programs {
        // Ad-hoc retry.
        let mut i = 0;
        let line = loop {
            let r = process_program(
                prover.clone(),
                opts,
                &reload_logger,
                &get_object,
                key.to_owned(),
                re.clone(),
            )
            .await;
            match r {
                Ok(v) => break v,
                Err(e) => {
                    if i < 5 {
                        i += 1;
                        println!("attempt {i} of {key} failed");
                    } else {
                        return Err(e);
                    }
                }
            }
        };
        out.write_all(line.as_bytes()).await?;
    }
    // reload_logger(None);

    // write_measurements_to_csv(&measurements, "measurements.csv")?;

    // tokio::time::sleep(Duration::from_secs(2)).await;

    Ok(())
}

async fn process_program<Fut>(
    prover: Arc<SP1Prover<GpuProverComponents>>,
    opts: SP1ProverOpts,
    reload_logger: impl Fn(Option<WriteHalf<SimplexStream>>),
    get_object: impl Fn(String) -> Fut,
    key: String,
    re: Regex,
) -> Result<String>
where
    Fut: Future<Output = Result<Bytes, Report>>,
{
    let (program, stdin) = tokio::try_join!(
        get_object(format!("{key}/program.bin")),
        get_object(format!("{key}/stdin.bin"))
    )?;

    let stdin = bincode::deserialize::<SP1Stdin>(&stdin)?;

    let (r_log, w_log) = tokio::io::simplex(16384);
    reload_logger(Some(w_log));

    let prover = Arc::clone(&prover);
    let re = &re;

    let (finished_trigger, finished_tripwire) = Tripwire::new();

    let (measurement, matches) = tokio::try_join!(
        async move {
            let res = tokio::task::spawn_blocking(move || {
                let measurement =
                    make_measurement(&prover, &key, &program, Some(stdin), opts, true, Stage::Wrap);

                measurement
            })
            .await
            .map_err(Report::from);
            drop(finished_trigger);
            res
        },
        async move {
            LinesStream::new(BufReader::new(r_log).lines())
                .take_until_if(finished_tripwire)
                .filter_map(|line_res| match line_res {
                    Ok(line) => {
                        // Skip lines that do not match the regex
                        let captures = re.captures(&line)?;
                        let s = captures.extract::<3>().1.map(|x| x.to_owned());
                        Some(Ok(s))
                    }
                    // Forward errors.
                    Err(e) => Some(Err(Report::from(e))),
                })
                .collect::<Result<Vec<_>>>()
                .await
                .map_err(Report::from)
        }
    )?;

    Ok(format!("{:?}", (measurement, matches)))
    // measurements.push(measurement);
}
