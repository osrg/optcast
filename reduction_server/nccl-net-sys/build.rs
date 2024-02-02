/*
 * Copyright (c) 2024, the Optcast Authors. All rights reserved.
 *
 * See LICENSE for license information
 */

use std::env;
use std::path::PathBuf;

fn main() {
    let dst = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_file = dst.join("nccl_net.rs");

    println!("cargo:rerun-if-changed=wrapper.h");
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_default(true)
        .default_enum_style(bindgen::EnumVariation::ModuleConsts)
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file(out_file)
        .expect("Couldn't write bindings!");

    println!("cargo:rustc-link-lib=nccl-net");
}
