//! A duct with a smaller exit than inlet chokes the source: the pressure inside rises (the fan is working
//! against a restriction), the exit speed rises, and the volumetric flow falls.
use peregrine_core::{shapes, Aabb, PortCap, TriMesh};
use peregrine_project::{Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_solver::{Backend, Simulation};
use peregrine_voxel::{build_domain, PortSpec};

const L: f64 = 30.0;
const W: f64 = 7.0; // half width of the channel
const T: f64 = 2.0;
const STEPS: usize = 5000;

fn cap(x: f64, h: f64, nx: f64) -> PortCap {
    PortCap { origin: [x, 0.0, 0.0], normal: [nx, 0.0, 0.0], polygon: vec![[x, -h, -h], [x, h, -h], [x, h, h], [x, -h, h]], triangles: vec![], holes: vec![] }
}

/// Square channel with a 12x12 mm inlet at x=0 and an exit hole of half-size `h_out` at x=L.
fn run(h_out: f64) -> (f64, f64, f64, f64, f64) {
    let h_in = 6.0;
    let end = |x0: f64, x1: f64, h: f64| -> Vec<TriMesh> {
        let (lo, hi) = (-W - T, W + T);
        vec![
            shapes::solid_box([x0, lo, lo], [x1, -h, hi]),
            shapes::solid_box([x0, h, lo], [x1, hi, hi]),
            shapes::solid_box([x0, -h, lo], [x1, h, -h]),
            shapes::solid_box([x0, -h, h], [x1, h, hi]),
        ]
    };
    let mut solids = end(-T, 0.0, h_in);
    solids.extend(end(L, L + T, h_out));
    for (a, b) in [([0.0, W, -W - T], [L, W + T, W + T]), ([0.0, -W - T, -W - T], [L, -W, W + T]), ([0.0, -W, W], [L, W, W + T]), ([0.0, -W, -W - T], [L, W, -W])] {
        solids.push(shapes::solid_box(a, b));
    }
    let mut project = Project::new("restriction");
    project.solver.cell_size_mm = 0.5;
    project.solver.reference_speed = 0.0;
    // a small blower: 20 L/min free air, 300 Pa shut-off
    let curve = vec![(0.0, 300.0), (5.0, 280.0), (10.0, 230.0), (15.0, 150.0), (20.0, 0.0)];
    project.sources.push(Source { name: "fan".into(), kind: SourceKind::Fan { curve }, split: SplitMode::ByArea, preset: None, streamlines: 0 });
    let mk = |name: &str, kind: PortKind, cap: PortCap| Port { name: name.into(), body: "p".into(), kind, cap };
    project.ports.push(mk("in", PortKind::SourceInlet { source: "fan".into(), weight: 1.0 }, cap(0.0, h_in, -1.0)));
    project.ports.push(mk("out", PortKind::PressureOutlet { pressure: 0.0 }, cap(L + T, h_out, 1.0)));
    let specs: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet(), collar_mm: 0.0 }).collect();
    let mut bounds = Aabb::empty();
    for m in &solids { bounds.union(&m.bounds()); }
    let refs: Vec<&TriMesh> = solids.iter().collect();
    let d = build_domain(&refs, &specs, &bounds.expanded(4.0), 0.5);
    let mut sim = Simulation::new(&project, &d, Backend::Cpu).unwrap();
    for chunk in 0..STEPS / 2500 {
        sim.advance(2500);
        let r = sim.report();
        let (o, i) = (r.iter().find(|r| r.name == "out").unwrap(), r.iter().find(|r| r.name == "in").unwrap());
        eprintln!("  [exit half {h_out}] step {}: in {:.2} out {:.2} L/min, p_in {:.1} Pa", (chunk + 1) * 2500, i.flow_l_min, -o.flow_l_min, i.pressure_pa);
    }
    assert!(sim.is_finite());
    let rep = sim.report();
    let o = rep.iter().find(|r| r.name == "out").unwrap().clone();
    let i = rep.iter().find(|r| r.name == "in").unwrap().clone();
    // pressure on the channel axis, a quarter of the way along (well upstream of the exit)
    let u = sim.units.clone();
    let dims = d.grid.dims;
    let cells = sim.cells().to_vec();
    let mac = sim.mac().to_vec();
    let at = |x: f64| {
        let c = [((x - d.grid.origin[0]) / d.grid.dx) as usize, dims[1] / 2, dims[2] / 2];
        let idx = c[0] + dims[0] * (c[1] + dims[1] * c[2]);
        assert_ne!(cells[idx], 1);
        u.pressure(mac[idx][0] as f64)
    };
    (-o.flow_l_min, o.mean_speed, at(L * 0.25), at(L * 0.9), i.pressure_pa)
}

#[test]
fn smaller_exit_chokes_the_source() {
    let mut prev: Option<(f64, f64, f64)> = None;
    for h_out in [4.0, 2.0] {
        let (q, v, p_up, p_near_exit, _p_in) = run(h_out);
        eprintln!("exit {:>4.0} mm2: flow {:.2} L/min, exit speed {:.2} m/s, pressure upstream {:.1} Pa, just before exit {:.1} Pa", (2.0 * h_out) * (2.0 * h_out), q, v, p_up, p_near_exit);
        if let Some((q0, v0, p0)) = prev {
            assert!(q < q0, "flow must fall as the exit shrinks");
            assert!(v > v0, "exit speed must rise as the exit shrinks");
            assert!(p_up > p0, "pressure inside must rise as the exit shrinks");
        }
        prev = Some((q, v, p_up));
    }
}
