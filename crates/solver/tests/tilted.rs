use peregrine_core::{find_port_candidates, shapes, Aabb};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use peregrine_voxel::{build_domain, PortSpec};

/// A duct whose openings are tilted 35 degrees to the grid axes must behave like an axis-aligned one.
fn run(angle_deg: f64) -> (f64, f64, bool) {
    let mut tube = shapes::hollow_tube(8.0, 5.0, 24.0);
    tube.rotate_z(angle_deg.to_radians());
    let mut project = Project::new("t");
    project.solver.cell_size_mm = 0.5;
    project.solver.reference_speed = 6.0;
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Velocity { speed: 3.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
    for (i, c) in find_port_candidates(&tube, 1.0).into_iter().enumerate() {
        let inlet = c.cap.normal[0] < 0.0;
        let kind = if inlet { PortKind::SourceInlet { source: "fan".into(), weight: 1.0 } } else { PortKind::PressureOutlet { pressure: 0.0 } };
        project.ports.push(Port { name: format!("p{i}"), body: "duct".into(), kind, cap: c.cap });
    }
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet() }).collect();
    let mut b = Aabb::empty();
    b.union(&tube.bounds());
    let d = build_domain(&[&tube], &specs, &b.expanded(4.0), 0.5);
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    sim.advance(2600);
    let ok = sim.is_finite() && sim.max_speed() < 0.3;
    let rep = sim.report();
    let q_in: f64 = rep.iter().filter(|r| r.inlet).map(|r| r.flow_l_min).sum();
    let q_out: f64 = rep.iter().filter(|r| !r.inlet).map(|r| -r.flow_l_min).sum();
    eprintln!("angle {angle_deg}: in {q_in:.2} out {q_out:.2} L/min (expected {:.2}), ok={ok}", 3.0 * 25.0e-6 * 60000.0);
    (q_in, q_out, ok)
}

#[test]
fn tilted_ports_match_aligned_ports() {
    let (a_in, a_out, a_ok) = run(0.0);
    let (t_in, t_out, t_ok) = run(35.0);
    assert!(a_ok && t_ok, "stable: aligned {a_ok}, tilted {t_ok}");
    let expect = 3.0 * 25.0e-6 * 60000.0; // 3 m/s through 5x5 mm = 4.5 L/min
    for (name, v) in [("aligned in", a_in), ("aligned out", a_out), ("tilted in", t_in), ("tilted out", t_out)] {
        assert!((v - expect).abs() / expect < 0.15, "{name}: {v} vs {expect}");
    }
}
