use peregrine_core::{find_port_candidates, planar_regions, stl};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = stl::read_stl(std::path::Path::new(&p)).unwrap();
    let b = m.bounds();
    println!("tris {} verts {}  bounds min {:?} max {:?}", m.triangles.len(), m.vertices.len(), b.min, b.max);
    println!("signed volume {:.1} mm3 (negative = inward-wound)", m.signed_volume());
    let c = find_port_candidates(&m, 0.0);
    println!("{} hole candidates (area >= 0)", c.len());
    for (i, x) in c.iter().enumerate().take(40) {
        let ctr = x.cap.centroid();
        println!("#{i}: area {:.2} mm2  centre ({:.1},{:.1},{:.1}) normal ({:.2},{:.2},{:.2}) pts {}", x.area, ctr[0], ctr[1], ctr[2], x.cap.normal[0], x.cap.normal[1], x.cap.normal[2], x.cap.polygon.len());
    }
    let r = planar_regions(&m.welded(1e-5), 1.0, 1e-3);
    println!("{} planar regions", r.len());
}
