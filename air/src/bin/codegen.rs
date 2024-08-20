use p3_air::BaseAir;
use sp1_core_machine::riscv::RiscvAir;
use sp1_stark::air::MachineAir;
use sp1_stark::baby_bear_poseidon2::BabyBearPoseidon2;
use std::fs::{self};

const TEMPLATE: &str = include_str!("template.txt");

fn main() {
    let config = BabyBearPoseidon2::default();
    let machine = RiscvAir::machine(config);
    let chips = machine.chips();
    for chip in chips {
        let (_, expr_ctr) = air::codegen_cuda_eval(chip);
        let path = format!("./cuda/air/codegen/{}.cuh", chip.name().to_lowercase());
        let mut template = TEMPLATE.to_string();
        template = template.replace("Template", chip.name().as_str());
        template = template.replace("PREP_WIDTH_VALUE", &chip.preprocessed_width().to_string());
        template = template.replace("MAIN_WIDTH_VALUE", &chip.width().to_string());
        template = template.replace("PERM_WIDTH_VALUE", &chip.permutation_width().to_string());
        template = template.replace("EXPRESSION_COUNT_VALUE", &expr_ctr.to_string());
        fs::write(path, template).unwrap();
    }
}
