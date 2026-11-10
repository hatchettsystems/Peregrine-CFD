//! Builds the C++ shim against an installed OpenCASCADE (7.6+).
//! Linux: `apt install libocct-data-exchange-dev libocct-modeling-algorithms-dev libocct-ocaf-dev`
//! Windows: `vcpkg install opencascade:x64-windows`, then set OCCT_ROOT to
//!   <vcpkg>/installed/x64-windows (expects include/opencascade and lib/).
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=cpp/shim.cpp");
    println!("cargo:rerun-if-env-changed=OCCT_ROOT");
    println!("cargo:rerun-if-env-changed=OCCT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=OCCT_LIB_DIR");
    let root = std::env::var("OCCT_ROOT").ok().map(PathBuf::from);
    let include = std::env::var("OCCT_INCLUDE_DIR").map(PathBuf::from).ok()
        .or_else(|| root.as_ref().map(|r| r.join("include").join("opencascade")))
        .unwrap_or_else(|| PathBuf::from("/usr/include/opencascade"));
    let lib = std::env::var("OCCT_LIB_DIR").map(PathBuf::from).ok().or_else(|| root.as_ref().map(|r| r.join("lib")));
    println!("cargo:rustc-check-cfg=cfg(occt_missing)");
    if !include.join("STEPCAFControl_Reader.hxx").exists() {
        println!("cargo:warning=OpenCASCADE headers not found in {}; STEP import disabled (set OCCT_ROOT or OCCT_INCLUDE_DIR)", include.display());
        println!("cargo:rustc-cfg=occt_missing");
        return;
    }
    cc::Build::new().cpp(true).file("cpp/shim.cpp").include(&include).std("c++17").warnings(false).compile("peregrine_occt_shim");
    if let Some(l) = lib {
        println!("cargo:rustc-link-search=native={}", l.display());
    }
    for name in [
        "TKXDESTEP", "TKXCAF", "TKLCAF", "TKCAF", "TKCDF", "TKSTEP", "TKSTEPAttr", "TKSTEPBase", "TKSTEP209", "TKXSBase",
        "TKMesh", "TKBO", "TKPrim", "TKTopAlgo", "TKGeomAlgo", "TKBRep", "TKGeomBase", "TKG3d", "TKG2d", "TKMath", "TKService", "TKernel",
    ] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}
