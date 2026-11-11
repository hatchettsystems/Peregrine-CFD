use rayon::prelude::*;

pub const Q: usize = 19;
pub const C: [[i32; 3]; Q] = [
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
];
pub const W: [f32; Q] = {
    let mut w = [1.0 / 36.0; Q];
    w[0] = 1.0 / 3.0;
    let mut i = 1;
    while i <= 6 {
        w[i] = 1.0 / 18.0;
        i += 1;
    }
    w
};
#[inline]
pub fn opp(i: usize) -> usize {
    if i == 0 { 0 } else if i % 2 == 1 { i + 1 } else { i - 1 }
}

#[inline]
pub fn feq(i: usize, rho: f32, u: [f32; 3]) -> f32 {
    let c = C[i];
    let cu = c[0] as f32 * u[0] + c[1] as f32 * u[1] + c[2] as f32 * u[2];
    let uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
    W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu)
}

#[derive(Clone, Copy, Debug)]
pub enum PortBc {
    /// Fixed velocity (lattice units); density taken from the inward neighbour.
    Velocity([f32; 3]),
    /// Fixed density (lattice units); velocity taken from the inward neighbour.
    Pressure(f32),
}

#[derive(Clone, Copy, Debug)]
pub struct PortInfo {
    pub bc: PortBc,
    /// Unit step (axis-snapped) from a port cell into the duct.
    pub inward: [i32; 3],
    /// True inward unit normal of the opening (zero = use `inward`). Needed for tilted openings.
    pub normal: [f32; 3],
}

impl PortInfo {
    /// Inward unit normal, falling back to the axis step.
    pub fn inward_unit(&self) -> [f32; 3] {
        let n = self.normal;
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if l > 1e-6 {
            [n[0] / l, n[1] / l, n[2] / l]
        } else {
            [self.inward[0] as f32, self.inward[1] as f32, self.inward[2] as f32]
        }
    }
}

/// Common interface of the CPU and GPU solvers.
pub trait Engine {
    fn step_n(&mut self, n: usize);
    fn set_port_bc(&mut self, k: usize, bc: PortBc);
    /// rho, ux, uy, uz per cell (lattice units). The GPU engine reads back on demand.
    fn mac(&mut self) -> &[[f32; 4]];
    fn steps(&self) -> usize;
    fn name(&self) -> String;
}

/// (cell index, port k, inward-neighbour index) for every port cell.
/// The inward neighbour is the fluid cell (among the 18 lattice neighbours) that best lines up with the
/// opening's true inward normal, so tilted openings read from a real fluid cell.
pub fn port_cell_list(cfg: &LbmConfig) -> Vec<(usize, usize, usize)> {
    let [nx, ny, nz] = cfg.dims;
    let mut out = vec![];
    for idx in 0..cfg.cells.len() {
        let code = cfg.cells[idx];
        if code < 2 {
            continue;
        }
        let k = (code - 2) as usize;
        let n = cfg.ports[k].inward_unit();
        let (x, y, z) = ((idx % nx) as i32, ((idx / nx) % ny) as i32, (idx / (nx * ny)) as i32);
        let mut best: Option<(f32, usize)> = None;
        for c in C.iter().skip(1) {
            let (xx, yy, zz) = (x + c[0], y + c[1], z + c[2]);
            if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 {
                continue;
            }
            let j = xx as usize + nx * (yy as usize + ny * zz as usize);
            if cfg.cells[j] != 0 {
                continue;
            }
            let l = ((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) as f32).sqrt();
            let score = (c[0] as f32 * n[0] + c[1] as f32 * n[1] + c[2] as f32 * n[2]) / l;
            if best.map_or(true, |b| score > b.0) {
                best = Some((score, j));
            }
        }
        out.push((idx, k, best.map_or(idx, |b| b.1)));
    }
    out
}

#[derive(Clone)]
pub struct LbmConfig {
    pub dims: [usize; 3],
    /// 0 = fluid, 1 = solid, 2+k = port k.
    pub cells: Vec<u8>,
    pub tau0: f32,
    pub smagorinsky: f32,
    /// Domain faces -X,+X,-Y,+Y,-Z,+Z; true = open (pressure ~ rho 1), false = wall.
    pub open_faces: [bool; 6],
    pub body_force: [f32; 3],
    pub ports: Vec<PortInfo>,
    /// Regularized collision (filters non-hydrodynamic moments; stable at much lower viscosity than TRT).
    pub regularized: bool,
    /// Thickness (cells) of the viscous sponge layer along open domain faces (0 = off).
    pub sponge_cells: usize,
}

pub struct Lbm {
    pub cfg: LbmConfig,
    f: Vec<f32>,
    fnew: Vec<f32>,
    /// rho, ux, uy, uz per cell (lattice units, physical velocity).
    pub mac: Vec<[f32; 4]>,
    mac_new: Vec<[f32; 4]>,
    port_cells: Vec<(usize, usize, usize)>, // (cell idx, port k, inward neighbour idx)
    pub step_count: usize,
}

impl Lbm {
    pub fn new(cfg: LbmConfig) -> Lbm {
        let n = cfg.dims[0] * cfg.dims[1] * cfg.dims[2];
        assert_eq!(cfg.cells.len(), n);
        let mut f = vec![0.0; n * Q];
        for idx in 0..n {
            if cfg.cells[idx] != 1 {
                for i in 0..Q {
                    f[idx * Q + i] = feq(i, 1.0, [0.0; 3]);
                }
            }
        }
        let mac = vec![[1.0, 0.0, 0.0, 0.0]; n];
        let port_cells = port_cell_list(&cfg);
        Lbm { fnew: f.clone(), f, mac_new: mac.clone(), mac, cfg, port_cells, step_count: 0 }
    }

    pub fn n_cells(&self) -> usize { self.cfg.cells.len() }

    pub fn set_port_bc(&mut self, k: usize, bc: PortBc) { self.cfg.ports[k].bc = bc; }

    pub fn step(&mut self) {
        let [nx, ny, nz] = self.cfg.dims;
        let cells = &self.cfg.cells;
        let f = &self.f;
        let tau0 = self.cfg.tau0;
        let cs = self.cfg.smagorinsky;
        let force = self.cfg.body_force;
        let open = self.cfg.open_faces;
        let has_force = force != [0.0; 3];
        let regularized = self.cfg.regularized;
        let sponge = self.cfg.sponge_cells as i32;
        self.fnew.par_chunks_mut(Q).zip(self.mac_new.par_chunks_mut(1)).enumerate().for_each(|(idx, (out, mout))| {
            if cells[idx] != 0 {
                return;
            }
            let x = (idx % nx) as i32;
            let y = ((idx / nx) % ny) as i32;
            let z = (idx / (nx * ny)) as i32;
            let mut fi = [0f32; Q];
            for i in 0..Q {
                let c = C[i];
                let (sx, sy, sz) = (x - c[0], y - c[1], z - c[2]);
                let face = if sx < 0 { Some(0) } else if sx >= nx as i32 { Some(1) } else if sy < 0 { Some(2) } else if sy >= ny as i32 { Some(3) } else if sz < 0 { Some(4) } else if sz >= nz as i32 { Some(5) } else { None };
                fi[i] = match face {
                    Some(fc) => {
                        if open[fc] {
                            // zero-gradient: copy the population of the neighbour one cell inside
                            let mut inner = [x, y, z];
                            let ax = fc / 2;
                            inner[ax] += if fc % 2 == 0 { 1 } else { -1 };
                            let ok = inner[ax] >= 0 && inner[ax] < [nx, ny, nz][ax] as i32;
                            let j = if ok { inner[0] as usize + nx * (inner[1] as usize + ny * inner[2] as usize) } else { idx };
                            if cells[j] == 0 {
                                // pressure boundary: ambient density (rho = 1) with the neighbour's velocity,
                                // plus its non-equilibrium part. Pins the far-field pressure so mass cannot drift.
                                let mut rj = 0.0f32;
                                let mut mj = [0.0f32; 3];
                                for k in 0..Q {
                                    let fk = f[j * Q + k];
                                    rj += fk;
                                    for a in 0..3 { mj[a] += C[k][a] as f32 * fk; }
                                }
                                let uj = [mj[0] / rj, mj[1] / rj, mj[2] / rj];
                                feq(i, 1.0, uj) + (f[j * Q + i] - feq(i, rj, uj))
                            } else { f[idx * Q + opp(i)] }
                        } else {
                            f[idx * Q + opp(i)]
                        }
                    }
                    None => {
                        let j = sx as usize + nx * (sy as usize + ny * sz as usize);
                        if cells[j] == 1 { f[idx * Q + opp(i)] } else { f[j * Q + i] }
                    }
                };
            }
            let mut rho = 0.0;
            let mut m = [0.0f32; 3];
            for i in 0..Q {
                rho += fi[i];
                for a in 0..3 {
                    m[a] += C[i][a] as f32 * fi[i];
                }
            }
            let inv = 1.0 / rho;
            let u = [m[0] * inv, m[1] * inv, m[2] * inv];
            // Smagorinsky relaxation time from the non-equilibrium stress.
            let mut tau = tau0;
            let mut pi = [[0f32; 3]; 3];
            if cs > 0.0 || regularized {
                for i in 0..Q {
                    let fneq = fi[i] - feq(i, rho, u);
                    for a in 0..3 {
                        for b in 0..3 {
                            pi[a][b] += (C[i][a] * C[i][b]) as f32 * fneq;
                        }
                    }
                }
                if cs > 0.0 {
                    let mut s = 0.0;
                    for a in 0..3 {
                        for b in 0..3 {
                            s += pi[a][b] * pi[a][b];
                        }
                    }
                    tau = 0.5 * (tau0 + (tau0 * tau0 + 18.0 * std::f32::consts::SQRT_2 * cs * cs * s.sqrt() * inv).sqrt());
                }
            }
            if sponge > 0 {
                // viscous buffer along open faces absorbs disturbances before they reach the boundary
                let dist = [x, nx as i32 - 1 - x, y, ny as i32 - 1 - y, z, nz as i32 - 1 - z];
                let mut dmin = i32::MAX;
                for f in 0..6 {
                    if open[f] { dmin = dmin.min(dist[f]); }
                }
                if dmin < sponge {
                    let w = (sponge - dmin) as f32 / sponge as f32;
                    tau += w * w * (1.0 - tau);
                }
            }
            let omega = 1.0 / tau;
            let mut ueq = u;
            let mut uphys = u;
            if has_force {
                for a in 0..3 {
                    ueq[a] += tau * force[a] * inv;
                    uphys[a] += 0.5 * force[a] * inv;
                }
            }
            // Two-relaxation-time collision (magic parameter 3/16): odd moments relax with a rate
            // chosen so walls stay accurate and tau can approach 0.5 without blowing up.
            let tau_m = 0.5 + 0.1875 / (tau - 0.5);
            let omega_m = 1.0 / tau_m;
            let mut fe = [0f32; Q];
            for i in 0..Q {
                fe[i] = feq(i, rho, ueq);
            }
            if regularized {
                // Regularized BGK: keep only the second-order (stress) part of the non-equilibrium
                // population, projected on the Hermite basis, and relax it.
                for i in 0..Q {
                    let c = [C[i][0] as f32, C[i][1] as f32, C[i][2] as f32];
                    let q = (c[0] * c[0] - 1.0 / 3.0) * pi[0][0] + (c[1] * c[1] - 1.0 / 3.0) * pi[1][1] + (c[2] * c[2] - 1.0 / 3.0) * pi[2][2]
                        + 2.0 * (c[0] * c[1] * pi[0][1] + c[0] * c[2] * pi[0][2] + c[1] * c[2] * pi[1][2]);
                    out[i] = fe[i] + (1.0 - omega) * W[i] * 4.5 * q;
                }
            } else {
                out[0] = fi[0] - omega * (fi[0] - fe[0]);
                for i in 1..Q {
                    let o = opp(i);
                    let fp = 0.5 * (fi[i] + fi[o]);
                    let fm = 0.5 * (fi[i] - fi[o]);
                    let ep = 0.5 * (fe[i] + fe[o]);
                    let em = 0.5 * (fe[i] - fe[o]);
                    out[i] = fi[i] - omega * (fp - ep) - omega_m * (fm - em);
                }
            }
            mout[0] = [rho, uphys[0], uphys[1], uphys[2]];
        });
        // Ports: equilibrium boundary populations.
        for &(idx, k, nb) in &self.port_cells {
            let m = self.mac_new[nb];
            let (rho, u) = match self.cfg.ports[k].bc {
                PortBc::Velocity(u) => (m[0], u),
                PortBc::Pressure(r) => (r, [m[1], m[2], m[3]]),
            };
            // Non-equilibrium extrapolation: equilibrium at the imposed state plus the neighbour's
            // deviation from equilibrium. Much more stable (and accurate) than pure equilibrium.
            for i in 0..Q {
                let neq = if nb != idx { self.fnew[nb * Q + i] - feq(i, m[0], [m[1], m[2], m[3]]) } else { 0.0 };
                self.fnew[idx * Q + i] = feq(i, rho, u) + neq;
            }
            self.mac_new[idx] = [rho, u[0], u[1], u[2]];
        }
        std::mem::swap(&mut self.f, &mut self.fnew);
        std::mem::swap(&mut self.mac, &mut self.mac_new);
        self.step_count += 1;
    }

    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize {
        x + self.cfg.dims[0] * (y + self.cfg.dims[1] * z)
    }

    /// Mean density and velocity over the fluid cells of the YZ-slice at x.
    pub fn slice_x(&self, x: usize) -> (f32, f32, usize) {
        let [_, ny, nz] = self.cfg.dims;
        let (mut r, mut u, mut n) = (0.0, 0.0, 0);
        for z in 0..nz {
            for y in 0..ny {
                let i = self.idx(x, y, z);
                if self.cfg.cells[i] == 0 {
                    r += self.mac[i][0];
                    u += self.mac[i][1];
                    n += 1;
                }
            }
        }
        (r / n.max(1) as f32, u, n)
    }

    pub fn max_speed(&self) -> f32 {
        self.mac.iter().zip(&self.cfg.cells).filter(|(_, &c)| c == 0).map(|(m, _)| (m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt()).fold(0.0, f32::max)
    }

    pub fn is_finite(&self) -> bool {
        self.mac.iter().all(|m| m[0].is_finite() && m[1].is_finite())
    }

    /// Mean density and inward velocity of the inward-neighbour layer of port `k`; also returns cell count.
    pub fn port_stats(&self, k: usize) -> (f32, f32, usize) {
        let inw = self.cfg.ports[k].inward_unit();
        let (mut r, mut u, mut n) = (0.0, 0.0, 0);
        for &(_, pk, nb) in &self.port_cells {
            if pk == k {
                let m = self.mac[nb];
                r += m[0];
                u += m[1] * inw[0] + m[2] * inw[1] + m[3] * inw[2];
                n += 1;
            }
        }
        let d = n.max(1) as f32;
        (r / d, u / d, n)
    }
}

impl Engine for Lbm {
    fn step_n(&mut self, n: usize) { for _ in 0..n { self.step(); } }
    fn set_port_bc(&mut self, k: usize, bc: PortBc) { self.cfg.ports[k].bc = bc; }
    fn mac(&mut self) -> &[[f32; 4]] { &self.mac }
    fn steps(&self) -> usize { self.step_count }
    fn name(&self) -> String { "CPU".into() }
}
