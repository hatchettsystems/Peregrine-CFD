use peregrine_core::{Aabb, TriMesh};
use peregrine_geom::*;
use peregrine_project::*;
use peregrine_voxel::{voxelize_inside, Grid};
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::new("dk");
    let mut bodies = load_bodies(std::path::Path::new(&args[1]), &project).unwrap();
    let parts: Vec<String> = ["Structural Duct", "duct v4"].iter().map(|s| s.to_string()).collect();
    project.define_duct("duct", &parts);
    apply_roles(&project, &mut bodies, false);
    let mut m = TriMesh::default();
    for b in &bodies { if b.name == "Structural Duct" || b.name == "duct v4" { m.append(&b.mesh); } }
    let mut bb = Aabb::empty();
    bb.grow([-3.0, 3.0, 70.0]);
    bb.grow([-1.0, 20.0, 84.0]);
    for dx in [1.0, 0.5, 0.2] {
        let g = Grid::from_bounds(&bb, dx);
        let (inside, _) = voxelize_inside(&m, &g);
        println!("--- dx {dx}: x = {:.1}; columns y 3..20, rows z 84 down to 70 ('#' = wall, '.' = air)", g.origin[0] + 1.0 * dx);
        let x = 2usize.min(g.dims[0] - 1);
        for k in (0..g.dims[2]).rev() {
            let mut row = format!("z{:5.1} ", g.origin[2] + (k as f64 + 0.5) * dx);
            for j in 0..g.dims[1] { row.push(if inside[g.idx(x, j, k)] { '#' } else { '.' }); }
            println!("{row}");
        }
    }
}
