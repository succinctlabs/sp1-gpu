use crate::CUDA_P3_EVAL_EXPR_EF_CTR;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct SymbolicExprEF(pub u32);

impl SymbolicExprEF {
    pub fn empty() -> Self {
        Self(u32::MAX)
    }

    pub fn alloc() -> Self {
        let mut tmp = CUDA_P3_EVAL_EXPR_EF_CTR.lock().unwrap();
        let id = *tmp;
        *tmp += 1;
        drop(tmp);
        Self(id)
    }
}
