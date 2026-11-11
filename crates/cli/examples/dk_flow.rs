use peregrine_geom::*;
use peregrine_project::*;
use peregrine_solver::{Backend, Simulation};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let dx: f64 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let steps: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(3000);
    let ins: Vec<usize> = args.get(4).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let outs: Vec<usize> = args.get(5).map(|s| s.split(',').filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
    let mut project = Project::new("dk");
    let mut bodies = load_bodies(std::path::Path::new(path), &project).unwrap();
    let parts: Vec<String> = ["Structural Duct", "duct v4"].iter().map(|s| s.to_string()).collect();
    project.define_duct("duct", &parts);
    apply_roles(&project, &mut bodies, false);
    let c = duct_port_candidates(&bodies, &project.ducts, 0.5, None, true);
    if ins.is_empty() {
        for (i, (b, c)) in c.iter().enumerate() {
            let o = c.cap.centroid();
            println!("#{i:<3} {b:<8} area {:>7.1} surf {} vis {:.2} clr {:.2} ctr ({:>6.1},{:>6.1},{:>6.1}) n ({:>5.2},{:>5.2},{:>5.2}) holes {}", c.area, c.cap.is_surface() as u8, c.visibility, c.clearance, o[0], o[1], o[2], c.cap.normal[0], c.cap.normal[1], c.cap.normal[2], c.cap.holes.len());
        }
        return;
    }
    project.solver.cell_size_mm = dx;
    project.solver.max_steps = steps;
    project.solver.min_tau = 0.505;
    project.domain.padding_mm = 6.0;
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Fan { curve: presets::find_preset("WS7040 blower 24V").unwrap().curve() }, split: SplitMode::ByArea, preset: None, streamlines: 0 });
    for &i in &ins { project.ports.push(Port { name: format!("in{i}"), body: "duct".into(), kind: PortKind::SourceInlet { source: "fan".into(), weight: 1.0 }, cap: c[i].1.cap.clone() }); }
    for &i in &outs { project.ports.push(Port { name: format!("out{i}"), body: "duct".into(), kind: PortKind::PressureOutlet { pressure: 0.0 }, cap: c[i].1.cap.clone() }); }
    let domain = build_project_domain(&project, &bodies).unwrap();
    eprintln!("grid {:?}", domain.grid.dims);
    let mut sim = Simulation::new(&project, &domain, Backend::Cpu).unwrap();
    for k in 1..=steps / 500 {
        sim.advance(500);
        eprintln!("step {}", k * 500);
    }
    let rep = sim.report();
    for r in &rep { println!("{:8} {:>8.2} L/min  p {:>8.2} Pa  speed {:>5.2}", r.name, r.flow_l_min, r.pressure_pa, r.mean_speed); }
    for s in sim.source_reports() { println!("source {} total {:.1} L/min pressure {:.1} Pa", s.name, s.total_flow_l_min, s.pressure_pa); }
}
