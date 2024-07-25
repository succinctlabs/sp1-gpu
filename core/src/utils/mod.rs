mod nvtx;
mod stats;
mod tracer;

#[allow(unused_imports)]
pub use nvtx::*;
use sp1_core::{
    runtime::{ExecutionRecord, Program, Runtime},
    utils::SP1CoreOpts,
};
#[allow(unused_imports)]
pub use tracer::*;

pub use stats::*;

pub fn execute_core(program: Program) -> ExecutionRecord {
    let opts = SP1CoreOpts::default();
    let mut runtime = Runtime::new(program, opts);
    runtime.run().unwrap();
    runtime.record
}
