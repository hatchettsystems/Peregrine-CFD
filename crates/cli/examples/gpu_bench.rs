//! Times the GPU solver on a duct-like grid: cargo run --release -p peregrine-cli --example gpu_bench -- [nx ny nz steps]
use peregrine_solver::gpu::GpuLbm;
use peregrine_solver::lbm::*;
fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|v| v.parse().ok()).collect();
    let (nx, ny, nz, steps) = (*a.first().unwrap_or(&160), *a.get(1).unwrap_or(&96), *a.get(2).unwrap_or(&96), *a.get(3).unwrap_or(&200));
    let mut cells = vec![0u8; nx * ny * nz];
    for z in 0..nz { for y in 0..ny { for x in 0..nx {
        let i = x + nx * (y + ny * z);
        // a square tube in the middle of open air, with an inlet and outlet plane
        let (dy, dz) = ((y as f64 - ny as f64 / 2.0).abs(), (z as f64 - nz as f64 / 2.0).abs());
        let r = dy.max(dz);
        if x > 10 && x < nx - 10 && r >= 12.0 && r < 16.0 { cells[i] = 1; }
    } } }
    let cfg = LbmConfig { dims: [nx, ny, nz], cells, tau0: 0.6, smagorinsky: 0.12, open_faces: [true; 6], body_force: [0.0; 3], regularized: false, sponge_cells: 4, ports: vec![] };
    let mut gpu = GpuLbm::new(&cfg).unwrap();
    gpu.step_n(20);
    let _ = gpu.mac();
    let t = std::time::Instant::now();
    for _ in 0..steps / 25 { gpu.step_n(25); }
    let _ = gpu.mac();
    let s = t.elapsed().as_secs_f64();
    println!("{} cells, {} steps in {:.2} s = {:.1} steps/s = {:.0} M cell-updates/s ({})", nx * ny * nz, steps, s, steps as f64 / s, (nx * ny * nz * steps) as f64 / s / 1e6, gpu.name());
}
