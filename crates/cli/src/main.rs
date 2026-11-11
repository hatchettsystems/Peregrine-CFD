use anyhow::{bail, Context, Result};
use peregrine_core::{shapes, stl};
use peregrine_geom::{build_project_domain, duct_port_candidates, load_bodies};
use peregrine_project::presets;
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use std::path::{Path, PathBuf};

/// Round holes up to this diameter (mm) are treated as screw holes and not offered as ports.
const SCREW_MAX_D: f64 = 6.0;

const USAGE: &str = "peregrine <command>
  demo <dir>                       write a demo duct + nozzle block (STL) and a project file
  ports <project.toml>             list port candidates (planar openings) of the duct bodies
  set-port <project.toml> <#> <name> <kind>
                                   define a port from candidate #; kind = velocity=<m/s> | flow=<L/min> | outlet=<Pa> | source=<name>
  fans                             list the built-in fan library
  set-source <project.toml> <name> <spec> [area|weights|plenum]
                                   define a source; spec = flow=<L/min> | velocity=<m/s> | fan=<library name or part> | fan=q:dp,q:dp,...
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
        Some("fans") => {
            for p in presets::fan_presets() {
                println!("{:<52} {:>7.1} L/min  {:>7.1} Pa  {}", p.name, p.qmax_lpm, p.pmax_pa, if p.verified { "" } else { "(ESTIMATE)" });
            }
            Ok(())
        }
        Some("set-source") => {
            if args.len() < 4 { bail!("{USAGE}"); }
            set_source(Path::new(&args[1]), &args[2], &args[3], args.get(4).map(String::as_str))
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
    let c = duct_port_candidates(&bodies, 0.5, Some(SCREW_MAX_D));
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
    let c = duct_port_candidates(&bodies, 0.5, Some(SCREW_MAX_D));
    let (body, cand) = c.get(index).with_context(|| format!("no candidate #{index}"))?;
    let (k, v) = kind.split_once('=').context("kind must look like velocity=5")?;
    let kind = match k {
        "source" => PortKind::SourceInlet { source: v.to_string(), weight: 1.0 },
        "velocity" => PortKind::VelocityInlet { speed: v.parse()? },
        "flow" => PortKind::FlowInlet { l_per_min: v.parse()? },
        "outlet" => PortKind::PressureOutlet { pressure: v.parse()? },
        _ => bail!("unknown port kind {k}"),
    };
    project.ports.retain(|p| p.name != name);
    project.ports.push(Port { name: name.into(), body: body.clone(), kind, cap: cand.cap.clone() });
    project.save(project_path)?;
    println!("port '{name}' set from candidate #{index}");
    Ok(())
}

fn set_source(project_path: &Path, name: &str, spec: &str, split: Option<&str>) -> Result<()> {
    let mut project = Project::load(project_path)?;
    let (k, v) = spec.split_once('=').context("spec must look like flow=10")?;
    let mut preset = None;
    let kind = match k {
        "flow" => SourceKind::Flow { l_per_min: v.parse()? },
        "velocity" => SourceKind::Velocity { speed: v.parse()? },
        "fan" => {
            if let Some(p) = presets::fan_presets().into_iter().find(|p| p.name.to_lowercase().contains(&v.to_lowercase())) {
                preset = Some(p.name.to_string());
                SourceKind::Fan { curve: p.curve() }
            } else {
                let curve: Result<Vec<(f64, f64)>> = v.split(',').map(|pt| {
                    let (q, dp) = pt.split_once(':').context("fan curve points look like q:dp")?;
                    Ok((q.parse()?, dp.parse()?))
                }).collect();
                SourceKind::Fan { curve: curve? }
            }
        }
        _ => bail!("unknown source kind {k}"),
    };
    let split = match split {
        None | Some("area") => SplitMode::ByArea,
        Some("weights") => SplitMode::Weights,
        Some("plenum") => SplitMode::Plenum,
        Some(o) => bail!("unknown split {o} (area|weights|plenum)"),
    };
    project.sources.retain(|s| s.name != name);
    project.sources.push(Source { name: name.to_string(), kind, split, preset, streamlines: 40 });
    project.save(project_path)?;
    println!("source '{name}' set; assign inlets with: set-port <project> <#> <port> source={name}");
    Ok(())
}

fn run(project_path: &Path, steps: Option<usize>, out: &Path, backend: Backend) -> Result<()> {
    let (project, bodies) = load(project_path)?;
    let problems = project.validate();
    if !problems.is_empty() { bail!("project is not ready: {}", problems.join("; ")); }
    for w in presets::preset_warnings(&project) { println!("warning: {w}"); }
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
    let total = steps.unwrap_or(project.solver.max_steps);
    let mut ref_speed = project.solver.reference_speed;
    let mut sim;
    let mut attempt = 0;
    'attempts: loop {
        sim = Simulation::new_with_ref(&project, &domain, backend, ref_speed)?;
        println!("engine: {}", sim.engine_name());
        if sim.ref_speed > ref_speed * 1.01 {
            println!("note: reference speed raised to {:.1} m/s (estimated peak flow speed is higher than the project setting)", sim.ref_speed);
        }
        for w in sim.units.warnings() { println!("warning: {w}"); }
        println!("dt = {:.3e} s, tau = {:.3}", sim.units.dt, sim.units.tau_used());
        let chunk = 500;
        let mut done = 0;
        while done < total {
            let n = chunk.min(total - done);
            sim.advance(n);
            done += n;
            if sim.unstable() {
                attempt += 1;
                if attempt > 2 {
                    bail!("solution became unstable at step {done} even with reference speed {:.0} m/s. Check the geometry for tiny gaps, reduce the flow, or refine the cell size.", sim.ref_speed);
                }
                ref_speed = sim.ref_speed * 1.6;
                println!("unstable at step {done}: retrying with reference speed {ref_speed:.1} m/s (attempt {attempt})");
                continue 'attempts;
            }
            let net: f64 = sim.report().iter().map(|r| r.flow_l_min).sum();
            println!("step {done:>6}/{total}  net flow imbalance {net:+.3} L/min  max speed {:.2} m/s", { let ms = sim.max_speed() as f64; sim.units.vel_to_phys(ms) });
        }
        break;
    }
    print!("{}", peregrine_post::format_report(&sim.report()));
    for g in sim.source_reports() {
        println!("source {:<12} total {:>8.2} L/min  pressure {:>8.1} Pa", g.name, g.total_flow_l_min, g.pressure_pa);
    }
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
