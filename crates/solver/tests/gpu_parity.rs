#![cfg(feature = "gpu")]
use peregrine_solver::gpu::GpuLbm;
use peregrine_solver::lbm::*;

fn config() -> LbmConfig {
    let (nx, ny, nz) = (40usize, 12usize, 12usize);
    let mut cells = vec![0u8; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                if y == 0 || y == ny - 1 || z == 0 || z == nz - 1 {
                    cells[i] = 1;
                } else if x == 0 {
                    cells[i] = 2;
                } else if x == nx - 1 {
                    cells[i] = 3;
                }
            }
        }
    }
    // a solid obstacle and one open face to exercise those paths
    for z in 4..8 { for y in 4..8 { cells[20 + nx * (y + ny * z)] = 1; } }
    LbmConfig {
        dims: [nx, ny, nz],
        cells,
        tau0: 0.6,
        smagorinsky: 0.12,
        open_faces: [false, false, false, false, true, false],
        body_force: [0.0; 3],
        ports: vec![
            PortInfo { bc: PortBc::Velocity([0.03, 0.0, 0.0]), inward: [1, 0, 0], normal: [0.0; 3] },
            PortInfo { bc: PortBc::Pressure(1.0), inward: [-1, 0, 0], normal: [0.0; 3] },
        ],
    }
}

#[test]
fn gpu_matches_cpu() {
    let cfg = config();
    let mut gpu = match GpuLbm::new(&cfg) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let mut cpu = Lbm::new(cfg);
    cpu.step_n(300);
    gpu.step_n(300);
    let a = cpu.mac().to_vec();
    let b = gpu.mac().to_vec();
    let mut worst = 0f32;
    for (x, y) in a.iter().zip(&b) {
        for k in 0..4 {
            worst = worst.max((x[k] - y[k]).abs());
        }
    }
    assert!(worst < 2e-4, "max |cpu-gpu| = {worst}");
}
