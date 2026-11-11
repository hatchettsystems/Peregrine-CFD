use peregrine_core::find_port_candidates;
use peregrine_geom::load_bodies;
use peregrine_project::Project;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let project = Project::new("x");
    let t = std::time::Instant::now();
    let bodies = load_bodies(std::path::Path::new(&p), &project).unwrap();
    println!("loaded {} bodies in {:.1}s", bodies.len(), t.elapsed().as_secs_f64());
    for b in &bodies {
        let bb = b.mesh.bounds();
        let c = find_port_candidates(&b.mesh, 0.5);
        let big = c.iter().filter(|x| !x.is_screw_hole(6.0)).count();
        println!("{:<40} tris {:>7}  vol {:>9.0} mm3  size ({:.1} x {:.1} x {:.1})  planar holes {} (non-screw {})",
            b.name, b.mesh.triangles.len(), b.mesh.signed_volume(), bb.max[0]-bb.min[0], bb.max[1]-bb.min[1], bb.max[2]-bb.min[2], c.len(), big);
    }
}
