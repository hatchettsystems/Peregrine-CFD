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
    if let Some(l) = &lib {
        println!("cargo:rustc-link-search=native={}", l.display());
    }
    let windows = std::env::var("CARGO_CFG_TARGET_OS").map_or(false, |o| o == "windows");
    let mut libs: Vec<String> = vec![];
    if windows {
        // Toolkit names changed between OCCT releases (TKXDESTEP -> TKDESTEP in 7.9), so link every
        // toolkit import library that is installed.
        if let Some(l) = &lib {
            if let Ok(rd) = std::fs::read_dir(l) {
                for e in rd.flatten() {
                    let f = e.file_name().to_string_lossy().into_owned();
                    if f.starts_with("TK") && f.ends_with(".lib") {
                        libs.push(f.trim_end_matches(".lib").to_string());
                    }
                }
            }
        }
        if libs.is_empty() {
            panic!("no TK*.lib found; set OCCT_ROOT to the vcpkg installed/x64-windows directory");
        }
    } else {
        for name in [
            "TKXDESTEP", "TKXCAF", "TKLCAF", "TKCAF", "TKCDF", "TKSTEP", "TKSTEPAttr", "TKSTEPBase", "TKSTEP209", "TKXSBase",
            "TKMesh", "TKBO", "TKPrim", "TKTopAlgo", "TKGeomAlgo", "TKBRep", "TKGeomBase", "TKG3d", "TKG2d", "TKMath", "TKService", "TKernel",
        ] {
            libs.push(name.to_string());
        }
    }
    for name in libs {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}
