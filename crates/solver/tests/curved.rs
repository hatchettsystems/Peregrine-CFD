use peregrine_core::{cap_from_loop, find_port_candidates, shapes, smooth_region_loops, vec3::*, Aabb};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use peregrine_voxel::{build_domain, PortSpec};

/// Round duct with a flat inlet and a saddle-shaped (non-planar) outlet. The air must come out of the curved opening.
#[test]
fn curved_outlet_carries_the_flow() {
    let tube = shapes::scooped_round_tube(6.0, 4.0, 30.0, 1.5, 64).welded(1e-6);
    let seed = (0..tube.triangles.len())
        .find(|&i| {
            let [a, b, c] = tube.tri(i);
            let ctr = scale(add(add(a, b), c), 1.0 / 3.0);
            let r = (ctr[1] * ctr[1] + ctr[2] * ctr[2]).sqrt();
            ctr[0] > 27.0 && r > 4.3 && r < 5.7
        })
        .unwrap();
    let loops = smooth_region_loops(&tube, seed, 25.0);
    let inner = loops.iter().min_by(|a, b| a.perimeter.partial_cmp(&b.perimeter).unwrap()).unwrap();
    let outlet_cap = cap_from_loop(&inner.points, &tube).unwrap();
    let inlet_cap = find_port_candidates(&tube, 1.0).remove(0).cap;

    let mut project = Project::new("t");
    project.solver.cell_size_mm = 0.5;
    project.solver.reference_speed = 6.0;
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Velocity { speed: 3.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
    project.ports.push(Port { name: "in".into(), body: "duct".into(), kind: PortKind::SourceInlet { source: "fan".into(), weight: 1.0 }, cap: inlet_cap });
    project.ports.push(Port { name: "out".into(), body: "duct".into(), kind: PortKind::PressureOutlet { pressure: 0.0 }, cap: outlet_cap });
    assert!(project.validate().is_empty(), "{:?}", project.validate());
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet() }).collect();
    let mut b = Aabb::empty();
    b.union(&tube.bounds());
    let mut bounds = b.expanded(4.0);
    bounds.max[0] += 10.0;
    let d = build_domain(&[&tube], &specs, &bounds, 0.5);
    assert!(d.port_cell_counts.iter().all(|&n| n > 50), "{:?}", d.port_cell_counts);
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    sim.advance(3200);
    assert!(sim.is_finite() && sim.max_speed() < 0.3);
    let rep = sim.report();
    let q_in = rep.iter().find(|r| r.name == "in").unwrap().flow_l_min;
    let q_out = -rep.iter().find(|r| r.name == "out").unwrap().flow_l_min;
    let expect = 3.0 * std::f64::consts::PI * 16.0e-6 * 60000.0; // 3 m/s through a 4 mm radius bore
    eprintln!("curved outlet: in {q_in:.2} out {q_out:.2} L/min (expected {expect:.2})");
    assert!((q_in - expect).abs() / expect < 0.2, "inlet {q_in} vs {expect}");
    assert!((q_out - q_in).abs() / q_in < 0.25, "outlet {q_out} vs inlet {q_in}");
}
