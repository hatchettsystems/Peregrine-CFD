use peregrine_core::{rim::*, stl};
use std::collections::HashMap;
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = stl::read_stl(std::path::Path::new(&p)).unwrap();
    let t = std::time::Instant::now();
    // label smooth regions
    let nt = m.triangles.len();
    let mut label = vec![usize::MAX; nt];
    let mut n_reg = 0;
    let mut seeds = vec![];
    let mut covered = vec![false; nt];
    for s in 0..nt {
        if covered[s] { continue; }
        // reuse smooth_region_loops only for loops; here we just flood to mark coverage cheaply via loops call? use a simple local BFS
        seeds.push(s);
        // BFS using the same rule as smooth_region_loops (re-implemented minimal)
        let mut stack = vec![s];
        covered[s] = true; label[s] = n_reg;
        // build adjacency lazily
        let _ = &mut stack;
        n_reg += 1;
        // the flood is done in smooth_region_loops; to mark coverage we need region triangles, so call loops fn and mark via a second pass below
        break;
    }
    let _ = (label, seeds, covered, n_reg);
    // simpler: time one call on a seed and report loops of several seeds
    let mut stats: HashMap<String, usize> = HashMap::new();
    for seed in (0..nt).step_by(nt / 40) {
        let loops = smooth_region_loops(&m, seed, 25.0);
        for l in &loops {
            if l.perimeter >= 12.0 {
                let kind = if l.planarity_dev > 0.15 { "nonplanar" } else { "planar" };
                *stats.entry(kind.to_string()).or_default() += 1;
            }
        }
    }
    println!("40 seeds probed in {:.2}s: {:?}", t.elapsed().as_secs_f64(), stats);
}
