//! Headless run of the "restriction sample" exactly as the app builds it. args: cell_mm steps
use peregrine_core::shapes;
use peregrine_geom::*;
use peregrine_project::*;
use peregrine_solver::{Backend, Simulation};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dx: f64 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let steps: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(5000);
    let mut project = Project::new("restriction sample");
    project.solver.cell_size_mm = dx;
    project.solver.reference_speed = 0.0;
    project.domain.padding_mm = 4.0;
    let main = shapes::hollow_tube(14.0, 10.0, 30.0);
    let mut tip = shapes::hollow_tube(14.0, 4.0, 10.0);
    tip.translate([30.0 - std::env::var("OVL").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(0.0), 0.0, 0.0]);
    let mut bodies = vec![
        Body { name: "duct_main (10 mm bore)".into(), role: BodyRole::Duct, mesh: main },
        Body { name: "exit_tip (4 mm bore)".into(), role: BodyRole::Duct, mesh: tip },
    ];
    let parts: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
    project.define_duct("duct", &parts);
    apply_roles(&project, &mut bodies, false);
    let c = duct_port_candidates(&bodies, &project.ducts, 0.5, None, true);
    for (i, (_, c)) in c.iter().enumerate() { println!("#{i} area {:.1} ctr {:?} n {:?}", c.area, c.cap.centroid(), c.cap.normal); }
    let ends = |sign: f64| c.iter().enumerate().filter(|(_, (_, c))| c.cap.normal[0] * sign > 0.5).min_by(|a, b| (sign * -a.1 .1.cap.origin[0]).total_cmp(&(sign * -b.1 .1.cap.origin[0]))).map(|(i, _)| i).unwrap();
    let (i_in, i_out) = (ends(-1.0), ends(1.0));
    println!("inlet #{i_in} outlet #{i_out}");
    let curve = vec![(0.0, 300.0), (5.0, 280.0), (10.0, 230.0), (15.0, 150.0), (20.0, 0.0)];
    project.sources.push(Source { name: "blower".into(), kind: SourceKind::Fan { curve }, split: SplitMode::ByArea, preset: None, streamlines: 0 });
    project.ports.push(Port { name: "in".into(), body: "duct".into(), kind: PortKind::SourceInlet { source: "blower".into(), weight: 1.0 }, cap: c[i_in].1.cap.clone() });
    project.ports.push(Port { name: "out".into(), body: "duct".into(), kind: PortKind::PressureOutlet { pressure: 0.0 }, cap: c[i_out].1.cap.clone() });
    if let Ok(v) = std::env::var("COLLAR") { project.domain.inlet_collar_mm = v.parse().unwrap(); }
    let d = build_project_domain(&project, &bodies).unwrap();
    println!("grid {:?} sealed {} thin_restored {} repaired {}", d.grid.dims, d.sealed_cells_removed, d.thin_restored, d.repaired.len());
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    for k in 1..=steps / 1000 {
        sim.advance(1000);
        let r = sim.report();
        for sr in sim.source_reports() { println!("   source {} total {:.2} L/min p {:.1} Pa commanded {:.2}", sr.name, sr.total_flow_l_min, sr.pressure_pa, sr.commanded); }
        println!("step {}: in {:.2} out {:.2} L/min p_in {:.1} Pa", k * 1000, r[0].flow_l_min, r[1].flow_l_min, r[0].pressure_pa);
    }
    let mut line = String::new();
    for _ in 0..24 { sim.advance(25); let r = sim.report(); line += &format!("{:.0} ", r[0].pressure_pa); }
    println!("p_in every 25 steps: {line}");
    let u = sim.units.clone();
    let dims = d.grid.dims;
    let cells = sim.cells().to_vec();
    let mac = sim.mac().to_vec();
    let mut x = 0;
    while x < dims[0] {
        let (mut flux, mut pm, mut n) = (0.0, 0.0, 0);
        for z in 0..dims[2] { for y in 0..dims[1] {
            let i = x + dims[0] * (y + dims[1] * z);
            let (yy, zz) = (d.grid.origin[1] + (y as f64 + 0.5) * d.grid.dx, d.grid.origin[2] + (z as f64 + 0.5) * d.grid.dx);
            if cells[i] != 1 && yy.abs() < 4.9 && zz.abs() < 4.9 { flux += u.vel_to_phys(mac[i][1] as f64) * d.grid.dx * d.grid.dx * 0.06; pm += u.pressure(mac[i][0] as f64); n += 1; }
        } }
        println!("x {:6.1}: net +x flux {:7.2} L/min  mean p {:7.1} Pa  fluid cells {}", d.grid.origin[0] + (x as f64 + 0.5) * d.grid.dx, flux, pm / n.max(1) as f64, n);
        x += 6;
    }
}

