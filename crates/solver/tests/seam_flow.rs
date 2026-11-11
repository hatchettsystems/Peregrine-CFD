use peregrine_core::{find_port_candidates, shapes, Aabb};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use peregrine_voxel::{build_domain_ext, PortSpec};

/// Fraction of the inlet flow that reaches the outlet of a two-part duct with a 0.4 mm seam.
fn outlet_fraction(seal_tol: f64) -> f64 {
    let a = shapes::hollow_tube(8.0, 4.0, 15.0);
    let mut b = shapes::hollow_tube(8.0, 4.0, 15.0);
    b.translate([15.4, 0.0, 0.0]);
    let mut project = Project::new("t");
    project.solver.cell_size_mm = 0.5;
    project.solver.reference_speed = 6.0;
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Velocity { speed: 3.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
    for (m, inlet) in [(&a, true), (&b, false)] {
        for c in find_port_candidates(m, 1.0) {
            if (inlet && c.cap.normal[0] < 0.0) || (!inlet && c.cap.normal[0] > 0.0) {
                let kind = if inlet { PortKind::SourceInlet { source: "fan".into(), weight: 1.0 } } else { PortKind::PressureOutlet { pressure: 0.0 } };
                project.ports.push(Port { name: if inlet { "in".into() } else { "out".into() }, body: "duct".into(), kind, cap: c.cap });
            }
        }
    }
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet(), collar_mm: 0.0 }).collect();
    let mut bb = Aabb::empty();
    bb.union(&a.bounds());
    bb.union(&b.bounds());
    let d = build_domain_ext(&[&a, &b], &[vec![0, 1]], seal_tol, &specs, &bb.expanded(4.0), 0.5);
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    sim.advance(3200);
    let rep = sim.report();
    let q_in = rep.iter().find(|r| r.name == "in").unwrap().flow_l_min;
    let q_out = -rep.iter().find(|r| r.name == "out").unwrap().flow_l_min;
    eprintln!("seal tolerance {seal_tol}: in {q_in:.2}, out {q_out:.2} L/min -> {:.0}% reaches the outlet", 100.0 * q_out / q_in);
    q_out / q_in
}

#[test]
fn sealed_seam_keeps_the_flow_in_the_duct() {
    let sealed = outlet_fraction(0.5);
    let open = outlet_fraction(0.0);
    assert!(sealed > 0.85, "sealed seam: only {:.0}% reaches the outlet", 100.0 * sealed);
    assert!(sealed >= open - 0.02, "sealing must not make things worse ({sealed} vs {open})");
}
