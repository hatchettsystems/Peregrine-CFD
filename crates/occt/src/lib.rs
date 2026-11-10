//! STEP import through OpenCASCADE: assembly tree, product names, instance transforms and a
//! triangle mesh per leaf body (units: mm).
use peregrine_core::TriMesh;
use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::path::Path;

#[cfg(not(occt_missing))]
#[repr(C)]
struct PgStep {
    _private: [u8; 0],
}

#[cfg(not(occt_missing))]
extern "C" {
    fn pg_step_read(path: *const c_char, deflection: c_double, err: *mut c_char, err_len: c_int) -> *mut PgStep;
    fn pg_step_body_count(s: *mut PgStep) -> c_int;
    fn pg_step_body_name(s: *mut PgStep, i: c_int) -> *const c_char;
    fn pg_step_body_nverts(s: *mut PgStep, i: c_int) -> c_int;
    fn pg_step_body_verts(s: *mut PgStep, i: c_int) -> *const c_double;
    fn pg_step_body_ntris(s: *mut PgStep, i: c_int) -> c_int;
    fn pg_step_body_tris(s: *mut PgStep, i: c_int) -> *const u32;
    fn pg_step_free(s: *mut PgStep);
    fn pg_step_write_demo(path: *const c_char, err: *mut c_char, err_len: c_int) -> c_int;
}

#[derive(Debug, thiserror::Error)]
pub enum StepError {
    #[error("{0}")]
    Occt(String),
}

/// True when this build was linked against OpenCASCADE.
pub const AVAILABLE: bool = cfg!(not(occt_missing));

pub struct StepBody {
    pub name: String,
    pub mesh: TriMesh,
}

#[cfg(not(occt_missing))]
fn err_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
}

/// Read a STEP file. `deflection` is the tessellation tolerance in mm (0.02 - 0.1 is typical).
#[cfg(occt_missing)]
pub fn read_step(_path: &Path, _deflection: f64) -> Result<Vec<StepBody>, StepError> {
    Err(StepError::Occt("this build has no OpenCASCADE; install it and set OCCT_ROOT, then rebuild (see docs/BUILD.md)".into()))
}
#[cfg(occt_missing)]
pub fn write_demo_step(_path: &Path) -> Result<(), StepError> {
    Err(StepError::Occt("built without OpenCASCADE".into()))
}

#[cfg(not(occt_missing))]
pub fn read_step(path: &Path, deflection: f64) -> Result<Vec<StepBody>, StepError> {
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| StepError::Occt(e.to_string()))?;
    let mut err = vec![0 as c_char; 512];
    let h = unsafe { pg_step_read(c.as_ptr(), deflection, err.as_mut_ptr(), err.len() as c_int) };
    if h.is_null() {
        return Err(StepError::Occt(err_string(&err)));
    }
    let mut out = vec![];
    unsafe {
        for i in 0..pg_step_body_count(h) {
            let name = CStr::from_ptr(pg_step_body_name(h, i)).to_string_lossy().into_owned();
            let nv = pg_step_body_nverts(h, i) as usize;
            let nt = pg_step_body_ntris(h, i) as usize;
            let v = std::slice::from_raw_parts(pg_step_body_verts(h, i), nv * 3);
            let t = std::slice::from_raw_parts(pg_step_body_tris(h, i), nt * 3);
            let raw = TriMesh {
                vertices: v.chunks(3).map(|p| [p[0], p[1], p[2]]).collect(),
                triangles: t.chunks(3).map(|p| [p[0], p[1], p[2]]).collect(),
            };
            out.push(StepBody { name, mesh: raw.welded(1e-5) });
        }
        pg_step_free(h);
    }
    Ok(out)
}

/// Write a small test assembly (see shim.cpp). For tests and demos.
#[cfg(not(occt_missing))]
pub fn write_demo_step(path: &Path) -> Result<(), StepError> {
    let c = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| StepError::Occt(e.to_string()))?;
    let mut err = vec![0 as c_char; 512];
    let rc = unsafe { pg_step_write_demo(c.as_ptr(), err.as_mut_ptr(), err.len() as c_int) };
    if rc != 0 { Err(StepError::Occt(err_string(&err))) } else { Ok(()) }
}
