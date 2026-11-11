use peregrine_core::{find_curved_openings_with, CurvedParams, TriMesh};
use peregrine_geom::load_bodies;
use peregrine_project::Project;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let bodies = load_bodies(std::path::Path::new(&p), &Project::new("x")).unwrap();
    let mut mesh = TriMesh::default();
    for b in &bodies { mesh.append(&b.mesh); }
    for (abs, rel, conf) in [(0.3, 0.02, 0.2), (0.1, 0.005, 0.2), (0.03, 0.001, 0.2), (0.03, 0.001, 0.35), (0.03, 0.001, 0.5)] {
        let t = std::time::Instant::now();
        let prm = CurvedParams { min_off_plane: abs, min_off_plane_rel: rel, min_confidence: conf, ..Default::default() };
        let c = find_curved_openings_with(&mesh, &prm);
        println!("off-plane>={abs} rel>={rel} conf>={conf}: {} openings ({:.1}s)", c.len(), t.elapsed().as_secs_f64());
        if conf == 0.2 && abs == 0.3 {
            for (i, o) in c.iter().enumerate().take(30) {
                let ctr = o.cap.centroid();
                println!("   #{i}: area {:>7.1}  perim {:>6.1}  off-plane {:>5.2}  conf {:.2}  centre ({:.1},{:.1},{:.1}) n=({:.2},{:.2},{:.2})", o.cap.area(), o.perimeter, o.planarity_dev, o.confidence, ctr[0], ctr[1], ctr[2], o.cap.normal[0], o.cap.normal[1], o.cap.normal[2]);
            }
        }
    }
}
