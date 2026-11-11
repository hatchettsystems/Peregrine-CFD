use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let mut bodies = load_bodies(std::path::Path::new(&args[2]), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    project.solver.cell_size_mm = args[3].parse().unwrap();
    for c in args[4..].iter().filter_map(|v| v.parse::<f64>().ok()) {
        project.domain.inlet_collar_mm = c;
        let d = build_project_domain(&project, &bodies).unwrap();
        println!("collar {c}: {:?}", leak_warning(&project, &d).map(|m| m[..m.find(':').unwrap_or(m.len()).min(150)].to_string()));
    }
}
