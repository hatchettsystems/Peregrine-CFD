use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let mut bodies = load_bodies(std::path::Path::new(&args[2]), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    project.solver.cell_size_mm = args[3].parse().unwrap();
    let cands = duct_port_candidates(&bodies, &project.ducts, 0.05, None, false);
    let mut n = 0;
    for (b, c) in &cands {
        if c.circularity > 0.75 && c.equiv_diameter <= 6.0 {
            let ctr = c.cap.centroid();
            if project.ports.iter().any(|p| (p.cap.centroid()[0] - ctr[0]).abs() < 0.2 && (p.cap.centroid()[1] - ctr[1]).abs() < 0.2 && (p.cap.centroid()[2] - ctr[2]).abs() < 0.2) { continue; }
            if project.blockers.iter().any(|k| (k.cap.centroid()[0] - ctr[0]).abs() < 0.2 && (k.cap.centroid()[1] - ctr[1]).abs() < 0.2 && (k.cap.centroid()[2] - ctr[2]).abs() < 0.2) { continue; }
            project.blockers.push(Blocker { name: format!("block_auto_{n}"), body: b.clone(), cap: c.cap.clone() });
            n += 1;
        }
    }
    println!("auto-blocked {n} round holes <= 6 mm (total blockers {})", project.blockers.len());
    if let Some(out) = args.get(4) {
        project.domain.padding_mm = 3.0;
        project.domain.outlet_clearance_mm = 12.0;
        project.domain.auto_seal_leaks = false;
        project.solver.max_steps = 3000;
        project.save(std::path::Path::new(out)).unwrap();
        println!("saved {out}");
        return;
    }
    let d = build_project_domain(&project, &bodies).unwrap();
    println!("auto-repaired {} gaps", d.repaired.len());
    let mut cl: std::collections::BTreeMap<(i32, i32, i32), usize> = Default::default();
    for p in &d.repaired { *cl.entry(((p[0] / 8.0).floor() as i32 * 8, (p[1] / 8.0).floor() as i32 * 8, (p[2] / 8.0).floor() as i32 * 8)).or_default() += 1; }
    let mut v: Vec<_> = cl.into_iter().collect();
    v.sort_by(|a, b| b.1.cmp(&a.1));
    for (k, n) in v.iter().take(14) { println!("  repairs in 8mm cell x{} y{} z{}: {n}", k.0, k.1, k.2); }
    println!("leak: {:?}", leak_warning(&project, &d).map(|m| m[..m.find(':').unwrap_or(m.len()).min(160)].to_string()));
}
