use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut project = Project::new("dk");
    let mut bodies = load_bodies(std::path::Path::new(&path), &project).unwrap();
    let names: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
    let parts: Vec<String> = ["Structural Duct", "duct v4"].iter().map(|s| s.to_string()).collect();
    project.define_duct("duct", &parts);
    apply_roles(&project, &mut bodies, false);
    let _ = names;
    let c = duct_port_candidates(&bodies, &project.ducts, 0.5, None, true);
    for (i, (b, c)) in c.iter().enumerate() {
        let o = c.cap.centroid();
        println!("#{i:<3} {b:<8} area {:>7.1} surf {} vis {:.2} clr {:.2} ctr ({:>6.1},{:>6.1},{:>6.1}) n ({:>5.2},{:>5.2},{:>5.2}) holes {}", c.area, c.cap.is_surface() as u8, c.visibility, c.clearance, o[0], o[1], o[2], c.cap.normal[0], c.cap.normal[1], c.cap.normal[2], c.cap.holes.len());
    }
}
