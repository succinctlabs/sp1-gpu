use std::collections::HashMap;

use crate::{operation::Operation, symbolic_folder_expr::SymbolicFolderExpr};

struct RegisterAllocator {
    used: Vec<bool>,
    vreg2phys_map: HashMap<SymbolicFolderExpr, SymbolicFolderExpr>,
    max: usize,
}

impl RegisterAllocator {
    pub fn new() -> Self {
        let mut used = vec![false; 1024];

        // Make %v0 always map to %p0.
        used[0] = true;
        let mut vreg2phys_map = HashMap::new();
        vreg2phys_map.insert(SymbolicFolderExpr(0), SymbolicFolderExpr(0));

        Self { used, vreg2phys_map, max: 0 }
    }

    pub fn vreg2phys(&mut self, vreg: SymbolicFolderExpr) -> SymbolicFolderExpr {
        if self.vreg2phys_map.contains_key(&vreg) {
            return self.vreg2phys_map[&vreg];
        }

        for i in 0..self.used.len() {
            if !self.used[i] {
                self.used[i] = true;
                let phys = SymbolicFolderExpr(i);
                self.vreg2phys_map.insert(vreg, phys);

                if i > self.max {
                    self.max = i;
                }

                return phys;
            }
        }

        unreachable!()
    }

    pub fn free(&mut self, vreg: SymbolicFolderExpr) {
        if self.vreg2phys_map.contains_key(&vreg) {
            let phys = self.vreg2phys_map.remove(&vreg).unwrap();
            self.used[phys.0] = false;
        }
    }
}

pub fn optimize(operations: Vec<Operation>) -> (Vec<Operation>, usize) {
    let mut first_time_vreg_used: HashMap<SymbolicFolderExpr, usize> = HashMap::new();
    let mut last_time_vreg_used: HashMap<SymbolicFolderExpr, usize> = HashMap::new();
    for (i, op) in operations.iter().enumerate() {
        first_time_vreg_used.entry(op.a).or_insert(i);
        last_time_vreg_used.insert(op.a, i);
        last_time_vreg_used.insert(op.b_expr, i);
        last_time_vreg_used.insert(op.c_expr, i);
    }
    last_time_vreg_used.insert(SymbolicFolderExpr(0), usize::MAX);

    let mut optimized_operations = Vec::new();
    let mut allocator = RegisterAllocator::new();
    for (i, op) in operations.iter().enumerate() {
        let phys_a = allocator.vreg2phys(op.a);
        let phys_b = allocator.vreg2phys(op.b_expr);
        let phys_c = allocator.vreg2phys(op.c_expr);

        let mut new_op = *op;
        new_op.a = phys_a;
        new_op.b_expr = phys_b;
        new_op.c_expr = phys_c;
        optimized_operations.push(new_op);

        if last_time_vreg_used.get(&op.a).unwrap() == &i {
            allocator.free(op.a);
        }
        if last_time_vreg_used.get(&op.b_expr).unwrap() == &i {
            allocator.free(op.b_expr);
        }
        if last_time_vreg_used.get(&op.c_expr).unwrap() == &i {
            allocator.free(op.c_expr);
        }
    }

    (optimized_operations, allocator.max)
}
