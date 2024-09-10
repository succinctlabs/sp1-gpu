#![allow(clippy::uninit_vec)]
#![allow(clippy::too_many_arguments)]

pub mod baby_bear;
pub mod cuda_runtime;
pub mod device;
pub mod dft;
pub mod fri;
pub mod matrix;
pub mod merkle_tree;
pub mod poseidon2;
pub mod stark;
pub mod time;
pub mod utils;

pub use cuda_runtime::{spawn::spawn, sync_device};

#[cfg(debug_assertions)]
compile_error!("moongate must be built in release mode. Please compile with the --release flag.");
