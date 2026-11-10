use anyhow::{bail, Context, Result};
use peregrine_core::{shapes, stl};
use peregrine_geom::{build_project_domain, duct_port_candidates, load_bodies};
use peregrine_project::{Port, PortKind, Project};
use peregrine_solver::{Backend, Simulation};
use std::path::{Path, PathBuf};

const USAGE: &str = "peregrine <command>
  demo <dir>                       write a demo duct + nozzle block (STL) and a project file
  ports <project.toml>             list port candidates (planar openings) of the duct bodies
  set-port <project.toml> <#> <name> <kind>
                                   define a port from candidate #; kind = velocity=<m/s> | flow=<L/min> | outlet=<Pa>
  run <project.toml> [--steps N] [--out DIR] [--cpu|--gpu]
                                   voxelize, solve, print the port report, write VTK";

fn main() {
    if let Err(e) = real_main() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn real_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("demo") => demo(Path::new(args.get(1).context("demo <dir>")?)),
        Some("ports") => ports(Path::new(args.get(1).context("ports <project>")?)),
        Some("set-port") => {
            if args.len() < 5 { bail!("{USAGE}"); }
            set_port(Path::new(&args[1]), args[2].parse()?, &args[3], &args[4])
        }
        Some("run") => {
            let proj = PathBuf::from(args.get(1).context("run <project>")?);
            let mut steps = None;
            let mut out = PathBuf::from("out");
            let mut backend = Backend::Auto;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--steps" => { steps = Some(args.get(i + 1).context("--steps N")?.parse()?); i += 2; }
                    "--cpu" => { backend = Backend::Cpu; i += 1; }
                    "--gpu" => { backend = Backend::Gpu; i += 1; }
                    "--out" => { out = PathBuf::from(args.get(i + 1).context("--out DIR")?); i += 2; }
                    other => bail!("unknown option {other}"),
                }
            }
            run(&proj, steps, &out, backend)
        }
        _ => {
            println!("{USAGE}");
            Ok(())
        }
    }
}

fn load(project_path: &Path) -> Result<(Project, Vec<peregrine_geom::Body>)> {
    let project = Project::load(project_path)?;
    let dir = project_path.parent().unwrap_or(Path::new("."));
    let geo = project.geometry_file.clone().context("project has no geometry_file")?;
    let bodies = load_bodies(&dir.join(geo), &project)?;
    Ok((project, bodies))
}

fn demo(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    // 40 mm straight duct, 10x10 mm outside, 6x6 mm channel, plus a "nozzle" block under the outlet.
    let duct = shapes::hollow_tube(10.0, 6.0, 40.0);
    stl::write_binary_stl(&duct, &dir.join("duct.stl"))?;
    let nozzle = shapes::solid_box([46.0, -5.0, -5.0], [56.0, 5.0, 5.0]);
    stl::write_binary_stl(&nozzle, &dir.join("nozzle.stl"))?;
    let mut p = Project::new("demo");
    p.geometry_file = Some(".".into());
    p.solver.cell_size_mm = 0.5;
    p.solver.reference_speed = 12.0;
    p.solver.max_steps = 4000;
    p.domain.padding_mm = 4.0;
    p.save(&dir.join("project.toml"))?;
    println!("wrote demo to {}\nnext:\n  peregrine ports {0}/project.toml", dir.display());
    Ok(())
}

fn ports(project_path: &Path) -> Result<()> {
    let (_, bodies) = load(project_path)?;
    for b in &bodies {
        println!("body {:<16} role {:?}  ({} triangles)", b.name, b.role, b.mesh.triangles.len());
    }
    let c = duct_port_candidates(&bodies, 0.5);
    if c.is_empty() {
        println!("no port candidates found (a duct needs planar faces with openings)");
    }
    for (i, (body, cand)) in c.iter().enumerate() {
        let ctr = cand.cap.centroid();
        println!("#{i}: body {body}  area {:.2} mm2  centre ({:.1}, {:.1}, {:.1})  outward normal ({:.2}, {:.2}, {:.2})",
            cand.area, ctr[0], ctr[1], ctr[2], cand.cap.normal[0], cand.cap.normal[1], cand.cap.normal[2]);
    }
    Ok(())
}

fn set_port(project_path: &Path, index: usize, name: &str, kind: &str) -> Result<()> {
    let (mut project, bodies) = load(project_path)?;
    let c = duct_port_candidates(&bodies, 0.5);
    let (body, cand) = c.get(index).with_context(|| format!("no candidate #{index}"))?;
    let (k, v) = kind.split_once('=').context("kind must look like velocity=5")?;
    let v: f64 = v.parse()?;
    let kind = match k {
        "velocity" => PortKind::VelocityInlet { speed: v },
        "flow" => PortKind::FlowInlet { l_per_min: v },
        "outlet" => PortKind::PressureOutlet { pressure: v },
        _ => bail!("unknown port kind {k}"),
    };
    project.ports.retain(|p| p.name != name);
    project.ports.push(Port { name: name.into(), body: body.clone(), kind, cap: cand.cap.clone() });
    project.save(project_path)?;
    println!("port '{name}' set from candidate #{index}");
    Ok(())
}

fn run(project_path: &Path, steps: Option<usize>, out: &Path, backend: Backend) -> Result<()> {
    let (project, bodies) = load(project_path)?;
    let domain = build_project_domain(&project, &bodies)?;
    let g = &domain.grid;
    let (gpu, cpu) = g.memory_estimate();
    println!("grid {}x{}x{} = {} cells at {} mm  (GPU ~{} MB, CPU ~{} MB)", g.dims[0], g.dims[1], g.dims[2], g.len(), g.dx, gpu >> 20, cpu >> 20);
    if domain.odd_columns > 0 {
        println!("warning: {} ray columns crossed the surface an odd number of times; a body may not be watertight", domain.odd_columns);
    }
    if domain.sealed_cells_removed > 0 {
        println!("note: {} enclosed fluid cells not connected to any port were closed off", domain.sealed_cells_removed);
    }
    let mut sim = Simulation::new(&project, &domain, backend)?;
    for w in sim.units.warnings() { println!("warning: {w}"); }
    println!("engine: {}", sim.engine_name());
    println!("dt = {:.3e} s, tau = {:.3}", sim.units.dt, sim.units.tau_used());
    let total = steps.unwrap_or(project.solver.max_steps);
    let chunk = 500;
    let mut done = 0;
    while done < total {
        let n = chunk.min(total - done);
        sim.advance(n);
        done += n;
        if !sim.is_finite() { bail!("solution diverged at step {done}; use a coarser cell or lower reference speed"); }
        let net: f64 = sim.report().iter().map(|r| r.flow_l_min).sum();
        println!("step {done:>6}/{total}  net flow imbalance {net:+.3} L/min  max speed {:.2} m/s", { let ms = sim.max_speed() as f64; sim.units.vel_to_phys(ms) });
    }
    print!("{}", peregrine_post::format_report(&sim.report()));
    let infl = sim.units.viscosity_inflation();
    if infl > 3.0 {
        println!("CAUTION: effective viscosity is {infl:.0}x physical at this cell size. Flow patterns are qualitative; pressure drops are overestimated. Refine the grid before trusting numbers.");
    }
    std::fs::create_dir_all(out)?;
    let vtk = out.join("result.vtk");
    peregrine_post::write_vtk(&mut sim, g, &vtk)?;
    println!("wrote {}", vtk.display());
    Ok(())
}
