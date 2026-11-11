use peregrine_core::{find_port_candidates, shapes, Aabb, PortCap, TriMesh};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{split_flows, Backend, Simulation};
use peregrine_voxel::{build_domain, PortSpec};

#[test]
fn split_math() {
    let f = split_flows(&[16.0, 64.0], &[1.0, 1.0], SplitMode::ByArea, 10.0);
    assert!((f[0] - 2.0).abs() < 1e-12 && (f[1] - 8.0).abs() < 1e-12);
    let f = split_flows(&[16.0, 64.0], &[3.0, 1.0], SplitMode::Weights, 8.0);
    assert!((f[0] - 6.0).abs() < 1e-12 && (f[1] - 2.0).abs() < 1e-12);
    assert!((split_flows(&[1.0, 2.0, 3.0], &[1.0; 3], SplitMode::ByArea, 5.0).iter().sum::<f64>() - 5.0).abs() < 1e-12);
}

/// Two parallel ducts (4 mm and 8 mm square channels) fed by one source, each with its own outlet.
fn setup(split: SplitMode) -> (Project, peregrine_voxel::Domain) {
    let mut a = shapes::hollow_tube(8.0, 4.0, 20.0);
    a.translate([0.0, -6.0, 0.0]);
    let mut b = shapes::hollow_tube(12.0, 8.0, 20.0);
    b.translate([0.0, 6.0, 0.0]);
    let mut project = Project::new("t");
    project.solver.cell_size_mm = 0.5;
    project.solver.reference_speed = 4.0;
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Flow { l_per_min: 10.0 }, split, preset: None, streamlines: 40 });
    let mut caps: Vec<(String, PortCap)> = vec![];
    for (tag, m) in [("A", &a), ("B", &b)] {
        for c in find_port_candidates(m, 1.0) {
            if c.cap.normal[0] < 0.0 {
                caps.push((format!("in{tag}"), c.cap));
            } else {
                caps.push((format!("out{tag}"), c.cap));
            }
        }
    }
    for (name, cap) in caps {
        let kind = if name.starts_with("in") { PortKind::SourceInlet { source: "fan".into(), weight: 1.0 } } else { PortKind::PressureOutlet { pressure: 0.0 } };
        project.ports.push(Port { name, body: "duct".into(), kind, cap });
    }
    assert!(project.validate().is_empty(), "{:?}", project.validate());
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet(), collar_mm: 0.0 }).collect();
    let mut bounds = Aabb::empty();
    for m in [&a, &b] { bounds.union(&m.bounds()); }
    let meshes: Vec<&TriMesh> = vec![&a, &b];
    let d = build_domain(&meshes, &specs, &bounds.expanded(3.0), 0.5);
    assert_eq!(d.sealed_cells_removed, 0);
    (project, d)
}

fn flow(rep: &[peregrine_solver::PortReport], name: &str) -> f64 {
    rep.iter().find(|r| r.name == name).unwrap().flow_l_min
}

#[test]
fn flow_source_split_by_cross_section() {
    let (project, domain) = setup(SplitMode::ByArea);
    let mut sim = Simulation::new(&project, &domain, Backend::Cpu).unwrap();
    sim.advance(2500);
    assert!(sim.is_finite());
    let rep = sim.report();
    let (a, b) = (flow(&rep, "inA"), flow(&rep, "inB"));
    eprintln!("by area: A {a:.2}  B {b:.2}  total {:.2}", a + b);
    // areas 16 : 64 mm2 -> 20% / 80% of 10 L/min
    assert!((a + b - 10.0).abs() < 1.5, "total {}", a + b);
    assert!((a / (a + b) - 0.2).abs() < 0.04, "A share {}", a / (a + b));
    let srep = sim.source_reports();
    assert_eq!(srep.len(), 1);
    assert!((srep[0].total_flow_l_min - (a + b)).abs() < 1e-6);
}

#[test]
fn flow_source_plenum_splits_by_resistance() {
    let (project, domain) = setup(SplitMode::Plenum);
    let mut sim = Simulation::new(&project, &domain, Backend::Cpu).unwrap();
    sim.advance(9000);
    assert!(sim.is_finite());
    let rep = sim.report();
    let (a, b) = (flow(&rep, "inA"), flow(&rep, "inB"));
    eprintln!("plenum: A {a:.2}  B {b:.2}  total {:.2}", a + b);
    assert!((a + b - 10.0).abs() < 2.0, "total {}", a + b);
    // the wide duct has far lower resistance, so it takes much more than its 80% area share
    assert!(b / (a + b) > 0.85, "B share {}", b / (a + b));
}

