use std::{
    future::Future,
    path::Path,
    sync::{Arc, LazyLock},
};

use enum_map::EnumMap;
use itertools::Itertools;
use moongate_gas::{make_measurement, report::Measurement, Stage};
use regex::Regex;
use serde::{Deserialize, Serialize};
use sp1_core_executor::RiscvAirId;
use sp1_core_machine::io::SP1Stdin;
use sp1_prover::SP1Prover;

use moongate_prover::{components::GpuProverComponents, gpu_prover_opts};

use moongate_gas::report::write_measurements_to_csv;

use clap::Parser;

use eyre::{OptionExt as _, Report, Result};

#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;
use sp1_stark::SP1ProverOpts;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, SimplexStream, WriteHalf};
use tokio_stream::{wrappers::LinesStream, StreamExt};
use tokio_util::{bytes::Bytes, io::SyncIoBridge};
use tracing_subscriber::fmt::writer::OptionalWriter;

use stream_cancel::{StreamExt as _, Tripwire};

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    pub s3_dirs: Vec<String>,
    #[arg(long, default_value = "log")]
    pub log_dir: String,
}

const BUCKET: &str = "sp1-testing-suite";

fn setup_logger(central_logfile: impl AsRef<Path>) -> impl Fn(Option<WriteHalf<SimplexStream>>) {
    use std::sync::Mutex;

    use tracing::Level;
    use tracing_subscriber::{filter, fmt, prelude::*, reload};

    let crate_layer = fmt::layer()
        .compact()
        .with_ansi(false)
        .with_writer(std::fs::File::create(central_logfile).unwrap())
        .with_filter(filter::Targets::new().with_target(env!("CARGO_CRATE_NAME"), Level::DEBUG));
    let stdout_layer = fmt::layer();
    let file_layer = fmt::layer()
        .with_ansi(false)
        .with_writer(Mutex::new(OptionalWriter::none()))
        .with_filter(filter::Targets::new().with_target("sp1_core_machine::utils", Level::DEBUG));
    let (file_layer, reload_handle) = reload::Layer::new(file_layer);

    tracing_subscriber::registry().with(crate_layer).with(stdout_layer).with(file_layer).init();

    move |maybe_writer: Option<WriteHalf<SimplexStream>>| {
        reload_handle
            .modify(|layer| {
                *layer.inner_mut().writer_mut() =
                    Mutex::new(maybe_writer.map(|writer| SyncIoBridge::new(writer)).into());
            })
            .unwrap();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Shard {
    pub program: String,
    pub shard_index: usize,
    pub core_proving_time_ns: u64,
    pub heights: Vec<(RiscvAirId, usize)>, // EnumMap<RiscvAirId, usize>,
    pub fitted_shape: Vec<(RiscvAirId, usize)>, // EnumMap<RiscvAirId, usize>,
}

async fn get_programs_in_dir(client: Arc<aws_sdk_s3::Client>, dir: String) -> Result<Vec<String>> {
    let ls = client
        .list_objects()
        .bucket(BUCKET)
        .prefix(format!("{}/", dir))
        .delimiter("/")
        .send()
        .await?;
    assert!(!ls.is_truncated().unwrap_or_default(), "s3 directory listing is truncated");

    let programs = ls
        .common_prefixes()
        .iter()
        .filter_map(|o| o.prefix().map(|x| x[..x.len() - 1].to_owned()))
        .collect::<Vec<_>>();
    Ok(programs)
}

#[tokio::main]
async fn main() -> Result<()> {
    let Args { s3_dirs, log_dir } = Args::parse();

    let log_dir = format!(
        "{log_dir}/{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
    );

    tokio::fs::create_dir_all(&log_dir).await?;

    let reload_logger = setup_logger(format!("{}/moongate_gas.log", &log_dir));

    let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
    let client = Arc::new(aws_sdk_s3::Client::new(&config));

    let programs = if s3_dirs.is_empty() {
        DEFAULT_PROGRAMS.map(|x| format!("v4/{x}")).to_vec()
    } else {
        s3_dirs
            .into_iter()
            .map(|dir| get_programs_in_dir(client.clone(), dir))
            .collect::<tokio::task::JoinSet<_>>()
            .join_all()
            .await
            .into_iter()
            .flatten_ok()
            .collect::<Result<Vec<_>>>()?
    };
    tracing::info!("found programs {programs:?}");

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
    let measurements_filename = format!("{}/programs.log", log_dir);
    let shards_filename = format!("{}/shards.log", log_dir);
    tracing::info!("writing data to {measurements_filename} and {shards_filename}");
    let mut measurements_file = tokio::fs::File::create(measurements_filename).await?;
    let mut shards_file = tokio::fs::File::create(shards_filename).await?;

    for key in programs {
        tracing::info!("running {key}");
        // Ad-hoc retry.
        let mut i = 0;
        let (measurement, shards) = loop {
            let r =
                process_program(prover.clone(), opts, &reload_logger, &get_object, key.to_owned())
                    .await;
            match r {
                Ok(v) => break v,
                Err(e) => {
                    if i < 5 {
                        i += 1;
                        tracing::warn!("attempt {i} of {key} failed");
                    } else {
                        return Err(e);
                    }
                }
            }
        };
        measurements_file.write_all(serde_json::to_string(&measurement)?.as_bytes()).await?;
        for shard in shards {
            shards_file.write_all(serde_json::to_string(&shard)?.as_bytes()).await?;
        }

        tokio::try_join!(measurements_file.flush(), shards_file.flush())?;
        shards_file.flush().await?;
        tracing::info!("wrote data for {key}");
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
) -> Result<(Measurement, Vec<Shard>)>
where
    Fut: Future<Output = Result<Bytes, Report>>,
{
    let (program, stdin) = tokio::try_join!(
        get_object(format!("{}/program.bin", key)),
        get_object(format!("{}/stdin.bin", key))
    )?;

    let stdin = bincode::deserialize::<SP1Stdin>(&stdin)?;

    let (r_log, w_log) = tokio::io::simplex(16384);
    reload_logger(Some(w_log));

    let prover = Arc::clone(&prover);

    let (finished_trigger, finished_tripwire) = Tripwire::new();

    let program_name = key.clone();

    let measurement_task = async move {
        let res = tokio::task::spawn_blocking(move || {
            make_measurement(&prover, &key, &program, Some(stdin), opts, true, Stage::Wrap)
        })
        .await
        .map_err(Report::from);
        reload_logger(None);
        drop(finished_trigger);
        res
    };

    let matches_task = async move {
        LinesStream::new(BufReader::new(r_log).lines())
            .take_until_if(finished_tripwire)
            .filter_map(|line_res| {
                line_res
                    .map(|line| {
                        static LINE_RE: LazyLock<Regex> = LazyLock::new(|| {
                            Regex::new(r"proving shard (\d+) took (\d+) ns: (.*)").unwrap()
                        });
                        // Skip lines that do not match the regex
                        let captures = LINE_RE.captures(&line)?;
                        Some(captures.extract::<3>().1.map(|x| x.to_owned()))
                    })
                    .map_err(Report::from)
                    .transpose()
            })
            .map(|res| {
                type ShardData = Option<(Vec<(RiscvAirId, usize)>, Vec<(RiscvAirId, usize)>)>;
                let [ind_str, prove_str, data_str] = res?;
                let (heights, fitted_shape) =
                    ron::from_str::<ShardData>(&data_str)?.ok_or_eyre("should have shard data")?;
                let shard = Shard {
                    program: program_name.clone(),
                    shard_index: ind_str.parse()?,
                    core_proving_time_ns: prove_str.parse()?,
                    heights,
                    fitted_shape,
                };
                Ok(shard)
            })
            .collect::<Result<Vec<_>>>()
            .await
            .map_err(Report::from)
    };

    let (measurement, matches) = tokio::try_join!(measurement_task, matches_task)?;

    Ok((measurement, matches))
}

const DEFAULT_PROGRAMS: [&str; 71] = [
    "blobstream-11220-11619",
    "blobstream-11297-12243",
    "blobstream-12620-13504",
    "blobstream-12987-13378",
    "blobstream-13375-14264",
    "blobstream-1380-2098",
    "blobstream-14446-14985",
    "blobstream-14480-14869",
    "blobstream-4644-5079",
    "blobstream-6775-7063",
    "blobstream-6945-7480",
    "blobstream-8059-9043",
    "chess",
    "eddsa-verify",
    "fibonacci",
    "fibonacci-1b",
    "fibonacci-200k",
    "fibonacci-200m",
    "fibonacci-20k",
    "fibonacci-20m",
    "fibonacci-2b",
    "fibonacci-2m",
    "fibonacci-400m",
    "fibonacci-40m",
    "fibonacci-4b",
    "fibonacci-4m",
    "helios",
    "json",
    "keccak256-100kb",
    "keccak256-10mb",
    "keccak256-1mb",
    "keccak256-300kb",
    "keccak256-3mb",
    "loop-100k",
    "loop-100m",
    "loop-10m",
    "loop-1m",
    "loop-300m",
    "loop-30m",
    "loop-3m",
    "op-succinct-chain-480-8710143-8710178",
    "regex",
    "rsp",
    "rsp-20526626",
    "rsp-20526627",
    "rsp-20526629",
    "rsp-20526630",
    "rsp-20528708",
    "rsp-20528709",
    "rsp-20528710",
    "rsp-20528711",
    "rsp-20528712",
    "sha256-100kb",
    "sha256-10mb",
    "sha256-1mb",
    "sha256-300kb",
    "sha256-3mb",
    "ssz-withdrawals",
    "tendermint",
    "vector-16434-16718",
    "vector-16495-17244",
    "vector-18964-19165",
    "vector-19220-19874",
    "vector-22266-22712",
    "vector-26830-27571",
    "vector-28340-28600",
    "vector-3695-4546",
    "vector-4730-5428",
    "vector-5809-5925",
    "vector-9201-9801",
    "vector-9844-10412",
];
