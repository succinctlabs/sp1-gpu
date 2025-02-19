mod nvtx;
mod stats;
mod tracer;

#[allow(unused_imports)]
pub use nvtx::*;
use sp1_core_executor::{ExecutionRecord, Executor, Program};
use sp1_stark::SP1CoreOpts;
#[allow(unused_imports)]
pub use tracer::*;

pub use stats::*;

// pub fn execute_core(program: Program) -> ExecutionRecord {
//     let opts = SP1CoreOpts::default();
//     let mut runtime = Executor::new(program, opts);
//     runtime.run().unwrap();
//     *(runtime.record)
// }
