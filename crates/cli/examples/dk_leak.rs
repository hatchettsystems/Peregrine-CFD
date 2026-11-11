use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let geo = std::path::PathBuf::from(&args[2]);
    let mut bodies = load_bodies(&geo, &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    for dx in args[3..].iter().filter_map(|v| v.parse::<f64>().ok()) {
        project.solver.cell_size_mm = dx;
        let d = build_project_domain(&project, &bodies).unwrap();
        println!("dx {dx}: grid {:?}, enclosed removed {}, leak: {:?}", d.grid.dims, d.sealed_cells_removed, leak_warning(&project, &d));
    }
}
