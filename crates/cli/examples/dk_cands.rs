use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let mut bodies = load_bodies(std::path::Path::new(&args[2]), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    let t = std::time::Instant::now();
    let c = duct_port_candidates(&bodies, &project.ducts, 0.5, if args.len() > 3 { None } else { Some(6.0) }, true);
    println!("{} candidates in {:.1}s", c.len(), t.elapsed().as_secs_f64());
    for (i, (_, c)) in c.iter().enumerate() {
        let o = c.cap.centroid();
        println!("#{i:<3} area {:>7.1} surf {} vis {:.2} clr {:.2} ctr ({:>6.1},{:>6.1},{:>6.1}) n ({:>5.2},{:>5.2},{:>5.2})", c.area, c.cap.is_surface() as u8, c.visibility, c.clearance, o[0], o[1], o[2], c.cap.normal[0], c.cap.normal[1], c.cap.normal[2]);
    }
}
