// Based on https://github.com/supranational/sppark/blob/main/rust/build.rs

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=../cuda/");
    println!("cargo:rerun-if-changed=../sppark/");
    println!("cargo:rerun-if-env-changed=CXXFLAGS");

    println!("Debug: Watching ../cuda/");
    println!("Debug: Watching ../sppark/");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let base_dir = manifest_dir.join("core");
    println!("basedir: {:?}", base_dir);

    let nvcc = which::which("nvcc");

    if let Ok(nvcc) = nvcc {
        let cuda_version =
            std::process::Command::new(nvcc).arg("--version").output().expect("impossible");
        if !cuda_version.status.success() {
            panic!("{:?}", cuda_version);
        }
        let cuda_version = String::from_utf8(cuda_version.stdout).unwrap();
        let x =
            cuda_version.find("release ").expect("can't find \"release X.Y,\" in --version output")
                + 8;
        let y =
            cuda_version[x..].find(',').expect("can't parse \"release X.Y,\" in --version output");
        let v = cuda_version[x..x + y].parse::<f32>().unwrap();
        if v < 12.0 {
            panic!("Unsupported CUDA version {} < 12.0", v);
        }

        let mut nvcc = cc::Build::new();
        nvcc.cuda(true).flag("-default-stream=per-thread");
        nvcc.include(base_dir);
        nvcc.flag("-Xcompiler").flag("-fopenmp");
        nvcc.flag("-Xptxas").flag("-suppress-stack-size-warning");
        // nvcc.flag("--threads").flag("14");
        nvcc.flag("-lnvToolsExt");
        nvcc.flag("-ldl");

        env::set_var("DEP_SPPARK_ROOT", "../sppark");
        if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
            nvcc.include(include);
            nvcc.define("SPPARK", None);
            nvcc.file("../sppark/rust/src/lib.cpp").file("../sppark/util/all_gpus.cpp");
        }

        nvcc.define("FEATURE_BABY_BEAR", None);

        nvcc.file("bindings/api.cu").compile("moongate_cuda");
    }
}
