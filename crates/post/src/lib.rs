//! Result export (legacy VTK, reports) and flow trajectories.
pub mod streamlines;
use peregrine_solver::{PortReport, Simulation};
use peregrine_voxel::Grid;
use std::io::Write;
use std::path::Path;

/// Write cell type, speed (m/s), velocity (m/s) and gauge pressure (Pa) as binary legacy VTK.
pub fn write_vtk(sim: &mut Simulation, grid: &Grid, path: &Path) -> std::io::Result<()> {
    let [nx, ny, nz] = grid.dims;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "# vtk DataFile Version 3.0\nPeregrine CFD\nBINARY\nDATASET STRUCTURED_POINTS")?;
    writeln!(w, "DIMENSIONS {} {} {}", nx + 1, ny + 1, nz + 1)?;
    writeln!(w, "ORIGIN {} {} {}", grid.origin[0], grid.origin[1], grid.origin[2])?;
    writeln!(w, "SPACING {} {} {}", grid.dx, grid.dx, grid.dx)?;
    writeln!(w, "CELL_DATA {}", nx * ny * nz)?;
    let u = sim.units;
    let mac: Vec<[f32; 4]> = sim.mac().to_vec();
    let cells = sim.cells();
    let fluid = |i: usize| cells[i] != 1;
    let put_scalar = |w: &mut dyn Write, name: &str, f: &dyn Fn(usize) -> f32| -> std::io::Result<()> {
        writeln!(w, "SCALARS {name} float 1\nLOOKUP_TABLE default")?;
        for i in 0..cells.len() {
            w.write_all(&f(i).to_be_bytes())?;
        }
        writeln!(w)
    };
    writeln!(w, "SCALARS cell_type float 1\nLOOKUP_TABLE default")?;
    for &c in cells {
        w.write_all(&(c as f32).to_be_bytes())?;
    }
    writeln!(w)?;
    put_scalar(&mut w, "speed", &|i| if fluid(i) { let m = mac[i]; u.vel_to_phys((m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() as f64) as f32 } else { 0.0 })?;
    put_scalar(&mut w, "pressure", &|i| if fluid(i) { u.pressure(mac[i][0] as f64) as f32 } else { 0.0 })?;
    writeln!(w, "VECTORS velocity float")?;
    for i in 0..cells.len() {
        let m = mac[i];
        for a in 1..4 {
            let v = if fluid(i) { u.vel_to_phys(m[a] as f64) as f32 } else { 0.0 };
            w.write_all(&v.to_be_bytes())?;
        }
    }
    writeln!(w)?;
    w.flush()
}

pub fn format_report(reports: &[PortReport]) -> String {
    let mut s = String::from("port            type    flow (L/min)   pressure (Pa)   cells\n");
    for r in reports {
        s += &format!("{:<15} {:<7} {:>12.3} {:>15.2} {:>7}\n", r.name, if r.inlet { "inlet" } else { "outlet" }, r.flow_l_min, r.pressure_pa, r.cells);
    }
    let net: f64 = reports.iter().map(|r| r.flow_l_min).sum();
    s += &format!("net inflow (should be ~0 at steady state): {net:.3} L/min\n");
    s
}
