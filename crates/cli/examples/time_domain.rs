//! Times the CPU-side set-up of a run: cargo run --release -p peregrine-cli --example time_domain -- project.toml [cell_mm]
use peregrine_geom::*;
use peregrine_project::*;
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&a[1])).unwrap();
    if let Some(dx) = a.get(2).and_then(|v| v.parse().ok()) { project.solver.cell_size_mm = dx; }
    let t = Instant::now();
    let mut bodies = load_bodies(project.geometry_file.as_ref().unwrap(), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    println!("load geometry: {:.1} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let d = build_project_domain(&project, &bodies).unwrap();
    println!("build domain: {:.1} s  ({} cells, dx {})", t.elapsed().as_secs_f64(), d.grid.len(), d.grid.dx);
    let solid = d.cells.iter().filter(|c| **c == 1).count();
    println!("solid cells {:.0}%", 100.0 * solid as f64 / d.cells.len() as f64);
    let t = Instant::now();
    let _w = (leak_warning(&project, &d), sealed_regions(&d));
    println!("warnings + sealed regions: {:.1} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let sim = peregrine_solver::Simulation::new(&project, &d, peregrine_solver::Backend::Cpu);
    println!("simulation setup (CPU engine): {:.1} s ok={}", t.elapsed().as_secs_f64(), sim.is_ok());
}
