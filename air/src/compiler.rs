use std::path::Path;

pub struct CudaConstraintCompiler {
    base_dir: String,
}

impl CudaConstraintCompiler {
    pub fn new(base_dir: String) -> Self {
        Self { base_dir }
    }
}

//

// What are the options to make an eval function in cuda

// struct CudaConstraintFolder {
//    bb33_t *prep_local,
//    bb33_t *prep_next,
//    bb33_t *main_local,
//    bb33_t *main_next,
//    bb33_t *perm_local,
//    bb33_t *perm_next,
//    Extbb33_t accumulator,
//    Extbb33_t is_first_row,
//    Extbb33_t is_last_row,
//    Extbb33_t is_transition,
//
//

// __device__ eval()
