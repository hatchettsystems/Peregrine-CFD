//! Is the way to each outlet open at a given cell size, and how narrow is its narrowest point?
//! cargo run --release -p peregrine-cli --example throat_widths -- zoom.toml geometry.step dx [dx ...]
//! For every outlet: the widest path from the inlets (bottleneck = the smallest distance to a wall along the path),
//! printed as the free diameter in mm and in cells.
use peregrine_geom::*;
use peregrine_project::*;
use std::collections::BinaryHeap;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut project = Project::load(std::path::Path::new(&a[1])).unwrap();
    project.geometry_file = Some(std::path::PathBuf::from(&a[2]));
    let mut bodies = load_bodies(project.geometry_file.as_ref().unwrap(), &project).unwrap();
    apply_roles(&project, &mut bodies, false);
    for dx in a[3..].iter().filter_map(|v| v.parse::<f64>().ok()) {
        project.solver.cell_size_mm = dx;
        let d = build_project_domain(&project, &bodies).unwrap();
        let g = &d.grid;
        let [nx, ny, nz] = g.dims;
        let n = g.len();
        // distance (in cells) from every cell to the nearest solid: two chamfer passes
        let mut dist = vec![1e4f32; n];
        for i in 0..n { if d.cells[i] == 1 { dist[i] = 0.0; } }
        let w = |dx: i32, dy: i32, dz: i32| ((dx * dx + dy * dy + dz * dz) as f32).sqrt();
        let fwd: Vec<(i32, i32, i32)> = (-1..=1).flat_map(|z| (-1..=1).flat_map(move |y| (-1..=1).map(move |x| (x, y, z)))).filter(|&(x, y, z)| (z, y, x) < (0, 0, 0)).collect();
        let bwd: Vec<(i32, i32, i32)> = fwd.iter().map(|&(x, y, z)| (-x, -y, -z)).collect();
        for (pass, offs) in [(0, &fwd), (1, &bwd)] {
            let order: Vec<usize> = if pass == 0 { (0..n).collect() } else { (0..n).rev().collect() };
            for i in order {
                let [x, y, z] = g.xyz(i);
                let mut best = dist[i];
                for &(ox, oy, oz) in offs.iter() {
                    let (xx, yy, zz) = (x as i32 + ox, y as i32 + oy, z as i32 + oz);
                    if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                    let j = g.idx(xx as usize, yy as usize, zz as usize);
                    best = best.min(dist[j] + w(ox, oy, oz));
                }
                dist[i] = best;
            }
        }
        let outs: Vec<usize> = project.ports.iter().enumerate().filter(|(_, p)| !p.kind.is_inlet()).map(|(k, _)| k).collect();
        let key = |v: f32| (v * 1000.0) as u32;
        let mut best = vec![0u32; n];
        let mut heap: BinaryHeap<(u32, usize)> = BinaryHeap::new();
        for i in 0..n {
            let code = d.cells[i];
            if code >= peregrine_voxel::PORT_BASE && project.ports.get((code - peregrine_voxel::PORT_BASE) as usize).map_or(false, |p| p.kind.is_inlet()) {
                let [x, y, z] = g.xyz(i);
                for (ox, oy, oz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                    let (xx, yy, zz) = (x as i32 + ox, y as i32 + oy, z as i32 + oz);
                    if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                    let j = g.idx(xx as usize, yy as usize, zz as usize);
                    if d.cells[j] == peregrine_voxel::FLUID && key(dist[j]) > best[j] { best[j] = key(dist[j]); heap.push((best[j], j)); }
                }
            }
        }
        let mut reach = vec![0u32; project.ports.len()];
        while let Some((b, i)) = heap.pop() {
            if b < best[i] { continue; }
            let [x, y, z] = g.xyz(i);
            for (ox, oy, oz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                let (xx, yy, zz) = (x as i32 + ox, y as i32 + oy, z as i32 + oz);
                if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                let j = g.idx(xx as usize, yy as usize, zz as usize);
                let code = d.cells[j];
                if code >= peregrine_voxel::PORT_BASE {
                    // reached the exit plane of port (code - base): the bottleneck of the best path that got here
                    let k = (code - peregrine_voxel::PORT_BASE) as usize;
                    if k < reach.len() { reach[k] = reach[k].max(b); }
                    continue;
                }
                if code != peregrine_voxel::FLUID { continue; }
                let cand = b.min(key(dist[j]));
                if cand > best[j] { best[j] = cand; heap.push((cand, j)); }
            }
        }
        print!("dx {dx:.2} mm ({} cells): ", n);
        for &k in outs.iter() {
            let cells_w = 2.0 * reach[k] as f64 / 1000.0; // free diameter in cells (distance*2)
            print!("{} {:.2} mm ({:.1} cells)   ", project.ports[k].name, cells_w * dx, cells_w);
        }
        println!();
    }
}
