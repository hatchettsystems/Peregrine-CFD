use peregrine_geom::*;
use peregrine_project::*;
fn coords(m: &str) -> Option<([f64; 3], f64)> {
    let i = m.find('(')?;
    let j = m[i..].find(')')? + i;
    let v: Vec<f64> = m[i + 1..j].split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let k = m.find("about ")? + 6;
    let w: f64 = m[k..].split_whitespace().next()?.parse().ok()?;
    Some(([v[0], v[1], v[2]], w))
}
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&args[1])).unwrap();
    let mut bodies = load_bodies(std::path::Path::new(&args[2]), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    let dx: f64 = args[3].parse().unwrap();
    project.solver.cell_size_mm = dx;
    for round in 0..8 {
        let d = build_project_domain(&project, &bodies).unwrap();
        match leak_warning(&project, &d) {
            None => { println!("round {round}: sealed, no leak"); break; }
            Some(m) => {
                println!("round {round}: {}", &m[..m.find(':').unwrap_or(m.len()).min(140)]);
                if let Some((c, w)) = coords(&m) {
                    project.plugs.push(Plug { name: format!("plug_{round}"), center: c, radius_mm: w / 2.0 + 0.7 });
                }
            }
        }
    }
    println!("plugs: {:?}", project.plugs);
    project.save(std::path::Path::new(&args[4])).unwrap();
}
