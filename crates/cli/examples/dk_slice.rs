use peregrine_geom::*;
use peregrine_project::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let mut bodies = load_bodies(std::path::Path::new(&args[2]), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    let dx: f64 = args[3].parse().unwrap();
    let xc: f64 = args[4].parse().unwrap();
    let (y0, y1, z0, z1): (f64, f64, f64, f64) = (args[5].parse().unwrap(), args[6].parse().unwrap(), args[7].parse().unwrap(), args[8].parse().unwrap());
    project.solver.cell_size_mm = dx;
    let d = build_project_domain(&project, &bodies).unwrap();
    let g = &d.grid;
    let i = ((xc - g.origin[0]) / dx) as usize;
    let mut z = z1;
    while z >= z0 {
        let k = ((z - g.origin[2]) / dx) as usize;
        let mut row = format!("z{:6.1} ", z);
        let mut y = y0;
        while y <= y1 {
            let j = ((y - g.origin[1]) / dx) as usize;
            let c = d.cells[g.idx(i, j, k)];
            row.push(match c { 0 => '.', 1 => '#', n => (b'0' + (n - 2)) as char });
            y += dx;
        }
        println!("{row}");
        z -= dx;
    }
}
