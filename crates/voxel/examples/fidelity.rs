use peregrine_core::stl;
use peregrine_voxel::{voxelize_inside, Grid};
fn main() {
    let p = std::env::args().nth(1).unwrap();
    let m = stl::read_stl(std::path::Path::new(&p)).unwrap();
    let vol = m.signed_volume();
    let b = m.bounds().expanded(3.0);
    println!("mesh volume {vol:.0} mm3");
    for dx in [1.0, 0.7, 0.5, 0.35] {
        let g = Grid::from_bounds(&b, dx);
        let (inside, odd) = voxelize_inside(&m, &g);
        let n = inside.iter().filter(|&&x| x).count() as f64 * dx * dx * dx;
        // wall thickness along z columns: run lengths of solid cells
        let [nx, ny, nz] = g.dims;
        let mut runs = vec![0usize; 12];
        let mut thin = 0usize; let mut total = 0usize;
        for y in 0..ny { for x in 0..nx {
            let mut run = 0;
            for z in 0..=nz {
                let s = z < nz && inside[x + nx * (y + ny * z)];
                if s { run += 1; } else if run > 0 { total += 1; if run <= 2 { thin += 1; } runs[run.min(11)] += 1; run = 0; }
            }
        } }
        println!("dx {dx}: voxel solid {n:.0} mm3 ({:.1}% of mesh), odd columns {odd}, z-runs <=2 cells: {:.0}% of {total}", 100.0 * n / vol, 100.0 * thin as f64 / total as f64);
    }
}
