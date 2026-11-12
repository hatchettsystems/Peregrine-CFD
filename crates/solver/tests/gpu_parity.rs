#![cfg(feature = "gpu")]
use peregrine_solver::gpu::GpuLbm;
use peregrine_solver::lbm::*;

fn config(regularized: bool) -> LbmConfig {
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
        regularized,
        sponge_cells: 4,
        ports: vec![
            PortInfo { bc: PortBc::Velocity([0.03, 0.0, 0.0]), inward: [1, 0, 0], normal: [0.0; 3] },
            PortInfo { bc: PortBc::Pressure(1.0), inward: [-1, 0, 0], normal: [0.0; 3] },
        ],
    }
}

/// One test (not two) on purpose: creating several GPU devices concurrently crashes some Windows drivers.
#[test]
fn gpu_matches_cpu() {
    check(false);
    check(true);
}

fn check(regularized: bool) {
    let cfg = config(regularized);
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
    // reading just some cells (what the control loops do) must match the full read-back exactly
    let want: Vec<usize> = (0..b.len()).step_by(37).chain([0usize, b.len() - 1]).collect();
    let got = gpu.read_cells(&want);
    assert_eq!(got.len(), want.len());
    for (g, &i) in got.iter().zip(&want) {
        assert_eq!(*g, b[i], "cell {i}");
    }
    // the same request again after more steps still follows the field
    gpu.step_n(20);
    let full = gpu.mac().to_vec();
    let again = gpu.read_cells(&want);
    for (g, &i) in again.iter().zip(&want) {
        assert_eq!(*g, full[i], "cell {i} after 20 more steps");
    }
}
