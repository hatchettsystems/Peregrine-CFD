use peregrine_solver::lbm::*;

/// Square duct, velocity inlet / pressure outlet, compared with the laminar f*Re = 56.91 solution.
#[test]
fn square_duct_pressure_drop_and_mass_balance() {
    check(false);
}

#[test]
fn square_duct_regularized() {
    check(true);
}

fn check(regularized: bool) {
    let (nx, ny, nz) = (80usize, 14usize, 14usize);
    let side = 12.0f64;
    let mut cells = vec![0u8; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = x + nx * (y + ny * z);
                let wall = y == 0 || y == ny - 1 || z == 0 || z == nz - 1;
                if wall {
                    cells[i] = 1;
                } else if x == 0 {
                    cells[i] = 2;
                } else if x == nx - 1 {
                    cells[i] = 3;
                }
            }
        }
    }
    let nu = 0.1f64; // tau = 0.8
    let u_in = 0.02f32;
    let cfg = LbmConfig {
        dims: [nx, ny, nz],
        cells,
        tau0: (3.0 * nu + 0.5) as f32,
        smagorinsky: 0.0,
        open_faces: [false; 6],
        body_force: [0.0; 3],
        regularized,
        sponge_cells: 0,
        ports: vec![
            PortInfo { bc: PortBc::Velocity([u_in, 0.0, 0.0]), inward: [1, 0, 0], normal: [0.0; 3] },
            PortInfo { bc: PortBc::Pressure(1.0), inward: [-1, 0, 0], normal: [0.0; 3] },
        ],
    };
    let mut lbm = Lbm::new(cfg);
    for _ in 0..9000 {
        lbm.step();
    }
    assert!(lbm.is_finite());
    let (r1, _, n1) = lbm.slice_x(20);
    let (r2, _, _) = lbm.slice_x(60);
    assert_eq!(n1, 144);
    let dp = (r1 - r2) as f64 / 3.0;
    let l = 40.0;
    let mean_u = lbm.slice_x(40).1 as f64 / 144.0;
    let expect = 28.455 * nu * mean_u * l / (side * side);
    assert!((dp - expect).abs() / expect < 0.12, "dp {dp} expected {expect}");
    // mass conservation along the duct
    // mass flux rho*u is conserved (density falls ~5% along the duct)
    let (ra, ua, _) = lbm.slice_x(15);
    let (rb, ub, _) = lbm.slice_x(65);
    let q1 = (ra * ua) as f64;
    let q2 = (rb * ub) as f64;
    assert!((q1 - q2).abs() / q1 < 0.02, "flux {q1} vs {q2}");
    assert!((mean_u - u_in as f64).abs() / (u_in as f64) < 0.06, "mean u {mean_u}");
}
