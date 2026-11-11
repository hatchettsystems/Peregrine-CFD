//! Flow split between a small port that faces the inlet and a large port off to the side.
use peregrine_core::{shapes, Aabb, PortCap, TriMesh};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use peregrine_voxel::{build_domain, PortSpec};

const L: f64 = 30.0;
const W: f64 = 13.0;
const T: f64 = 2.0;

type Hole = Option<([f64; 2], [f64; 2])>; // centre (u, v) and half sizes

/// A plenum wall on `axis` at the `side` (+1 / -1) end, with an optional rectangular hole.
/// Returns the wall pieces and the cap of the hole (on the outer face).
fn wall(axis: usize, side: f64, hole: Hole) -> (Vec<TriMesh>, Option<PortCap>) {
    let lo = [-T, -W - T, -W - T];
    let hi = [L + T, W + T, W + T];
    let inner = if side > 0.0 { [L, W, W][axis] } else { [0.0, -W, -W][axis] };
    let (a0, a1) = if side > 0.0 { (inner, inner + T) } else { (inner - T, inner) };
    let (u, v) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
    let mk = |ur: [f64; 2], vr: [f64; 2]| {
        let mut mn = [0.0; 3];
        let mut mx = [0.0; 3];
        mn[axis] = a0; mx[axis] = a1;
        mn[u] = ur[0]; mx[u] = ur[1];
        mn[v] = vr[0]; mx[v] = vr[1];
        shapes::solid_box(mn, mx)
    };
    let Some((c, h)) = hole else { return (vec![mk([lo[u], hi[u]], [lo[v], hi[v]])], None) };
    let parts = vec![
        mk([lo[u], c[0] - h[0]], [lo[v], hi[v]]),
        mk([c[0] + h[0], hi[u]], [lo[v], hi[v]]),
        mk([c[0] - h[0], c[0] + h[0]], [lo[v], c[1] - h[1]]),
        mk([c[0] - h[0], c[0] + h[0]], [c[1] + h[1], hi[v]]),
    ];
    let plane = if side > 0.0 { a1 } else { a0 };
    let corner = |du: f64, dv: f64| { let mut p = [0.0; 3]; p[axis] = plane; p[u] = c[0] + du * h[0]; p[v] = c[1] + dv * h[1]; p };
    let mut n = [0.0; 3];
    n[axis] = side;
    let mut o = [0.0; 3];
    o[axis] = plane; o[u] = c[0]; o[v] = c[1];
    let cap = PortCap { origin: o, normal: n, polygon: vec![corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)], triangles: vec![], holes: vec![] };
    (parts, Some(cap))
}

struct Out { small: f64, large: f64, inflow: f64, p_small: f64, p_large: f64 }

fn run(baffle: bool, dx: f64, steps: usize, q_lmin: f64) -> Out { run_kind(baffle, dx, steps, SourceKind::Flow { l_per_min: q_lmin }) }

fn run_kind(baffle: bool, dx: f64, steps: usize, kind: SourceKind) -> Out {
    let mut solids: Vec<TriMesh> = vec![];
    let (w, inlet) = wall(0, -1.0, Some(([0.0, 0.0], [3.0, 3.0])));
    solids.extend(w);
    let (w, small) = wall(0, 1.0, Some(([0.0, 0.0], [2.0, 2.0])));
    solids.extend(w);
    let (w, large) = wall(1, 1.0, Some(([15.0, 0.0], [6.0, 6.0])));
    solids.extend(w);
    solids.extend(wall(1, -1.0, None).0);
    solids.extend(wall(2, 1.0, None).0);
    solids.extend(wall(2, -1.0, None).0);
    if baffle { solids.push(shapes::solid_box([9.0, -5.0, -5.0], [11.0, 5.0, 5.0])); }
    let mut project = Project::new("branch");
    project.solver.cell_size_mm = dx;
    project.solver.reference_speed = 0.0;
    project.sources.push(Source { name: "fan".into(), kind, split: SplitMode::ByArea, preset: None, streamlines: 0 });
    let mk = |name: &str, kind: PortKind, cap: PortCap| Port { name: name.into(), body: "p".into(), kind, cap };
    project.ports.push(mk("in", PortKind::SourceInlet { source: "fan".into(), weight: 1.0 }, inlet.unwrap()));
    project.ports.push(mk("small", PortKind::PressureOutlet { pressure: 0.0 }, small.unwrap()));
    project.ports.push(mk("large", PortKind::PressureOutlet { pressure: 0.0 }, large.unwrap()));
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet() }).collect();
    let mut bounds = Aabb::empty();
    for m in &solids { bounds.union(&m.bounds()); }
    let refs: Vec<&TriMesh> = solids.iter().collect();
    let d = build_domain(&refs, &specs, &bounds.expanded(4.0), dx);
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    sim.advance(steps);
    assert!(sim.is_finite());
    let rep = sim.report();
    let get = |n: &str| rep.iter().find(|r| r.name == n).unwrap().clone();
    let (i, s, l) = (get("in"), get("small"), get("large"));
    Out { small: -s.flow_l_min, large: -l.flow_l_min, inflow: i.flow_l_min, p_small: s.pressure_pa, p_large: l.pressure_pa }
}

#[test]
#[ignore = "slow exploration (minutes); run with --ignored"]
fn explore_small_inline_vs_large_side() {
    for (label, baffle) in [("direct jet", false), ("baffled", true)] {
        for dx in [0.5] {
            let o = run(baffle, dx, 6000, 10.0);
            eprintln!("{label:11} dx {dx}: in {:.2} | small(16mm2) {:.2} L/min p {:.2} Pa | large(144mm2) {:.2} L/min p {:.2} Pa | sum out {:.2}", o.inflow, o.small, o.p_small, o.large, o.p_large, o.small + o.large);
        }
    }
}

#[test]
#[ignore = "slow exploration (minutes); run with --ignored"]
fn explore_fan_source() {
    for (label, baffle) in [("direct jet", false), ("baffled", true)] {
        let o = run_kind(baffle, 0.5, 6000, SourceKind::Fan { curve: vec![(0.0, 300.0), (10.0, 220.0), (20.0, 120.0), (30.0, 0.0)] });
        eprintln!("FAN {label:11}: in {:.2} | small(16mm2) {:.2} L/min p {:.2} Pa | large(144mm2) {:.2} L/min p {:.2} Pa | sum out {:.2}", o.inflow, o.small, o.p_small, o.large, o.p_large, o.small + o.large);
    }
}
