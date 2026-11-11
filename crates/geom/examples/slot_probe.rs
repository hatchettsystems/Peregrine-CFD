use peregrine_core::{find_curved_openings_with, CurvedParams, TriMesh};
use peregrine_geom::load_bodies;
use peregrine_project::Project;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let bodies = load_bodies(std::path::Path::new(&p), &Project::new("x")).unwrap();
    let mut mesh = TriMesh::default();
    for b in &bodies { mesh.append(&b.mesh); }
    let prm = CurvedParams { include_flat: true, min_confidence: 0.0, ..Default::default() };
    let c = find_curved_openings_with(&mesh, &prm);
    println!("accepted with confidence >= 0:");
    for o in &c {
        let ctr = o.cap.centroid();
        if (ctr[2] + 60.0).abs() < 4.0 && o.perimeter < 100.0 {
            println!("  area {:>6.1} perim {:>5.1} off-plane {:>4.2} conf {:.2} centre ({:.1},{:.1},{:.1}) n=({:.2},{:.2},{:.2})", o.cap.area(), o.perimeter, o.planarity_dev, o.confidence, ctr[0], ctr[1], ctr[2], o.cap.normal[0], o.cap.normal[1], o.cap.normal[2]);
        }
    }
}
