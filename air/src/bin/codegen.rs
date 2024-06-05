use sp1_core::air::MachineAir;
use sp1_core::stark::RiscvAir;
use sp1_core::utils::BabyBearPoseidon2;
use std::fs::{self};

const TEMPLATE: &str = include_str!("template.txt");

fn main() {
    let config = BabyBearPoseidon2::default();
    let machine = RiscvAir::machine(config);
    let chips = machine.chips();
    for chip in chips {
        let code = air::codegen_cuda_eval(chip);
        let path = format!("./cuda/air/codegen/{}.cuh", chip.name().to_lowercase());
        let mut template = TEMPLATE.to_string();
        template = template.replace("Template", chip.name().as_str());
        template = template.replace("CODEGEN", &code.join("\n       "));
        fs::write(path, template).unwrap();
    }
}
