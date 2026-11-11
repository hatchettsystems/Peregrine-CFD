use peregrine_geom::{duct_port_candidates, load_bodies};
use peregrine_project::{DuctDef, Project};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let project = Project::new("x");
    let bodies = load_bodies(std::path::Path::new(&p), &project).unwrap();
    let names: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
    println!("parts: {names:?}");
    let ducts = vec![DuctDef { name: "duct".into(), parts: names }];
    let t = std::time::Instant::now();
    let c = duct_port_candidates(&bodies, &ducts, 0.5, Some(6.0), true);
    println!("{} openings in {:.1}s", c.len(), t.elapsed().as_secs_f64());
    for (i, (_, x)) in c.iter().enumerate() {
        let ctr = x.cap.centroid();
        println!("#{i}: {:>8.1} mm2 {} islands {} visible {:>3.0}% clear {:>3.0}%  centre ({:.1},{:.1},{:.1}) normal ({:.2},{:.2},{:.2})", x.area, if x.cap.is_surface() { "curved" } else { "flat  " }, x.cap.holes.len(), x.visibility * 100.0, x.clearance * 100.0, ctr[0], ctr[1], ctr[2], x.cap.normal[0], x.cap.normal[1], x.cap.normal[2]);
    }
}
