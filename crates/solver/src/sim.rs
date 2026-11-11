use crate::lbm::{port_cell_list, Engine, Lbm, LbmConfig, PortBc, PortInfo};
use crate::units::Units;
use peregrine_project::{FaceBc, SourceKind, SplitMode, PortKind, Project};
use peregrine_voxel::Domain;

#[derive(Clone, Debug)]
pub struct PortReport {
    pub name: String,
    pub inlet: bool,
    /// Volumetric flow into the duct (L/min); negative = flow out.
    pub flow_l_min: f64,
    /// Mean static gauge pressure near the port (Pa).
    pub pressure_pa: f64,
    pub cells: usize,
    /// Name of the port source this opening belongs to, if any.
    pub source: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SourceReport {
    pub name: String,
    /// Total flow into the duct through all members (L/min).
    pub total_flow_l_min: f64,
    /// Mean static gauge pressure at the members (Pa).
    pub pressure_pa: f64,
    pub split: SplitMode,
    /// Flow the controller is currently commanding (L/min) for forced splits, or plenum pressure (Pa).
    pub commanded: f64,
}

/// Share a source's total flow (L/min) between members. `ByArea` gives each member flow proportional
/// to its cross-section (equal velocity); `Weights` uses the manual weights. `Plenum` returns the
/// by-area split as an initial estimate (the real split then follows the duct resistances).
pub fn split_flows(areas: &[f64], weights: &[f64], split: SplitMode, q_total: f64) -> Vec<f64> {
    let basis: Vec<f64> = match split {
        SplitMode::Weights => weights.to_vec(),
        _ => areas.to_vec(),
    };
    let sum: f64 = basis.iter().sum();
    basis.iter().map(|b| if sum > 0.0 { q_total * b / sum } else { 0.0 }).collect()
}

struct SourceRt {
    name: String,
    source: SourceKind,
    split: SplitMode,
    members: Vec<usize>,
    weights: Vec<f64>,
    areas: Vec<f64>, // m^2, discrete cross-section of each member
    q_cmd: f64,      // L/min (forced splits)
    p_cmd: f64,      // Pa (plenum)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
    /// GPU if one is available, otherwise CPU.
    Auto,
}

const RAMP_STEPS: usize = 1500;
const CHUNK: usize = 25;

pub struct Simulation {
    engine: Box<dyn Engine>,
    pub units: Units,
    pub dims: [usize; 3],
    cells: Vec<u8>,
    ports: Vec<PortInfo>,
    port_cells: Vec<(usize, usize, usize)>,
    project_ports: Vec<(String, PortKind, f64)>, // name, kind, cap area m^2
    port_dir: Vec<[f64; 3]>,
    /// Fixed inlet velocity vectors (lattice) ramped in over `RAMP_STEPS`.
    ramp_base: Vec<Option<[f32; 3]>>,
    sources: Vec<SourceRt>,
}

fn snap(v: [f64; 3]) -> [i32; 3] {
    let a = [v[0].abs(), v[1].abs(), v[2].abs()];
    let k = if a[0] >= a[1] && a[0] >= a[2] { 0 } else if a[1] >= a[2] { 1 } else { 2 };
    let mut o = [0; 3];
    o[k] = if v[k] >= 0.0 { 1 } else { -1 };
    o
}

fn curve_at(curve: &[(f64, f64)], q: f64) -> f64 {
    if curve.is_empty() { return 0.0; }
    if q <= curve[0].0 { return curve[0].1; }
    for w in curve.windows(2) {
        if q <= w[1].0 {
            let t = (q - w[0].0) / (w[1].0 - w[0].0).max(1e-12);
            return w[0].1 + t * (w[1].1 - w[0].1);
        }
    }
    curve[curve.len() - 1].1
}

fn dir_vel(d: [f64; 3], speed_lat: f64) -> [f32; 3] {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
    [(d[0] / l * speed_lat) as f32, (d[1] / l * speed_lat) as f32, (d[2] / l * speed_lat) as f32]
}

impl Simulation {
    pub fn new(project: &Project, domain: &Domain, backend: Backend) -> anyhow::Result<Simulation> {
        let g = &domain.grid;
        let units = Units::new(g.dx * 1e-3, project.solver.reference_speed, 0.08, project.fluid.density, project.fluid.kinematic_viscosity);
        let mut infos = vec![];
        let mut meta = vec![];
        let mut dirs = vec![];
        for p in &project.ports {
            let inward_f = [-p.cap.normal[0], -p.cap.normal[1], -p.cap.normal[2]];
            let dx_m = g.dx * 1e-3;
            let area = (domain.port_cell_counts[meta.len()].max(1) as f64) * dx_m * dx_m;
            let bc = match &p.kind {
                PortKind::VelocityInlet { speed } => PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(*speed))),
                PortKind::FlowInlet { l_per_min } => {
                    let u = l_per_min / 60000.0 / area;
                    PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(u)))
                }
                PortKind::FanInlet { curve } => {
                    let q0 = curve.last().map_or(1.0, |c| c.0) * 0.5;
                    let u = q0 / 60000.0 / area;
                    PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(u)))
                }
                PortKind::PressureOutlet { pressure } => PortBc::Pressure(units.rho_lat(*pressure) as f32),
                PortKind::SourceInlet { .. } => PortBc::Velocity([0.0; 3]),
            };
            infos.push(PortInfo { bc, inward: snap(inward_f) });
            meta.push((p.name.clone(), p.kind.clone(), area));
            dirs.push(inward_f);
        }
        let cfg = LbmConfig {
            dims: g.dims,
            cells: domain.cells.clone(),
            tau0: units.tau_used() as f32,
            smagorinsky: project.solver.smagorinsky as f32,
            open_faces: project.domain.faces.map(|f| f == FaceBc::Open),
            body_force: [0.0; 3],
            ports: infos.clone(),
        };
        let ramp_base = infos.iter().zip(&meta).map(|(p, m)| match (p.bc, &m.1) {
            (PortBc::Velocity(u), PortKind::VelocityInlet { .. } | PortKind::FlowInlet { .. }) => Some(u),
            _ => None,
        }).collect();
        let port_cells = port_cell_list(&cfg);
        let engine: Box<dyn Engine> = match backend {
            Backend::Cpu => Box::new(Lbm::new(cfg)),
            #[cfg(feature = "gpu")]
            Backend::Gpu => Box::new(crate::gpu::GpuLbm::new(&cfg)?),
            #[cfg(feature = "gpu")]
            Backend::Auto => match crate::gpu::GpuLbm::new(&cfg) {
                Ok(g) => Box::new(g),
                Err(_) => Box::new(Lbm::new(cfg.clone())),
            },
            #[cfg(not(feature = "gpu"))]
            Backend::Gpu => anyhow::bail!("built without the gpu feature"),
            #[cfg(not(feature = "gpu"))]
            Backend::Auto => Box::new(Lbm::new(cfg)),
        };
        let mut sources = vec![];
        for grp in &project.sources {
            let members = project.members(&grp.name);
            let areas: Vec<f64> = members.iter().map(|&k| meta[k].2).collect();
            let weights: Vec<f64> = members.iter().map(|&k| match &project.ports[k].kind { PortKind::SourceInlet { weight, .. } => *weight, _ => 1.0 }).collect();
            let sum_a: f64 = areas.iter().sum();
            let q_target = match &grp.kind {
                SourceKind::Flow { l_per_min } => Some(*l_per_min),
                SourceKind::Velocity { speed } => Some(speed * sum_a * 60000.0),
                SourceKind::Fan { .. } => None,
            };
            let q_cmd = q_target.unwrap_or_else(|| match &grp.kind { SourceKind::Fan { curve } => curve.last().map_or(1.0, |c| c.0) * 0.5, _ => 1.0 });
            let p_cmd = match (&grp.kind, q_target) {
                (SourceKind::Fan { curve }, _) => curve.first().map_or(0.0, |c| c.1) * 0.3,
                (_, Some(q)) => 1.5 * 0.5 * project.fluid.density * (q / 60000.0 / sum_a.max(1e-12)).powi(2),
                _ => 0.0,
            };
            sources.push(SourceRt { name: grp.name.clone(), source: grp.kind.clone(), split: grp.split, members, weights, areas, q_cmd, p_cmd });
        }
        let mut sim = Simulation { engine, units, dims: g.dims, cells: domain.cells.clone(), ports: infos, port_cells, project_ports: meta, port_dir: dirs, ramp_base, sources };
        sim.apply_ramp();
        sim.apply_sources();
        Ok(sim)
    }

    pub fn engine_name(&self) -> String { self.engine.name() }
    pub fn steps(&self) -> usize { self.engine.steps() }
    pub fn cells(&self) -> &[u8] { &self.cells }
    pub fn mac(&mut self) -> &[[f32; 4]] { self.engine.mac() }

    fn apply_ramp(&mut self) {
        let t = (self.engine.steps() as f32 / RAMP_STEPS as f32).min(1.0);
        let f = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
        for k in 0..self.ramp_base.len() {
            if let Some(u) = self.ramp_base[k] {
                self.engine.set_port_bc(k, PortBc::Velocity([u[0] * f, u[1] * f, u[2] * f]));
            }
        }
    }

    /// Advance `n` steps (inlet ramp and fan-curve control are applied between chunks of 25).
    pub fn advance(&mut self, n: usize) {
        let mut left = n;
        while left > 0 {
            let c = CHUNK.min(left);
            if self.engine.steps() <= RAMP_STEPS {
                self.apply_ramp();
            }
            if self.engine.steps() % 100 == 0 && self.engine.steps() > 0 {
                self.update_fans();
                if self.engine.steps() >= RAMP_STEPS {
                    self.update_sources();
                }
            }
            self.apply_sources();
            self.engine.step_n(c);
            left -= c;
        }
    }

    fn port_stats(&mut self, k: usize) -> (f32, f32, usize) {
        let inw = self.ports[k].inward;
        let list: Vec<usize> = self.port_cells.iter().filter(|c| c.1 == k).map(|c| c.2).collect();
        let mac = self.engine.mac();
        let (mut r, mut u) = (0.0, 0.0);
        for &nb in &list {
            let m = mac[nb];
            r += m[0];
            u += m[1] * inw[0] as f32 + m[2] * inw[1] as f32 + m[3] * inw[2] as f32;
        }
        let d = list.len().max(1) as f32;
        (r / d, u / d, list.len())
    }

    fn update_fans(&mut self) {
        for k in 0..self.project_ports.len() {
            if let PortKind::FanInlet { curve } = self.project_ports[k].1.clone() {
                let area = self.project_ports[k].2;
                let (rho, u_in, _) = self.port_stats(k);
                let u_phys = self.units.vel_to_phys(u_in as f64);
                let q = u_phys * area * 60000.0;
                let p_in = self.units.pressure(rho as f64);
                let target = curve_at(&curve, q.max(0.0));
                let scale = target.abs().max(p_in.abs()).max(1.0);
                let factor = 1.0 + 0.1 * ((target - p_in) / scale).clamp(-1.0, 1.0);
                let new_u = (u_phys * factor).max(1e-3);
                let d = self.port_dir[k];
                self.engine.set_port_bc(k, PortBc::Velocity(dir_vel(d, self.units.vel_to_lat(new_u))));
            }
        }
    }

    fn ramp_factor(&self) -> f32 {
        let t = (self.engine.steps() as f32 / RAMP_STEPS as f32).min(1.0);
        0.5 - 0.5 * (std::f32::consts::PI * t).cos()
    }

    /// Push the source controllers' current commands to the member boundary conditions.
    fn apply_sources(&mut self) {
        let f = self.ramp_factor();
        let mut updates: Vec<(usize, PortBc)> = vec![];
        for g in &self.sources {
            match g.split {
                SplitMode::ByArea | SplitMode::Weights => {
                    let flows = split_flows(&g.areas, &g.weights, g.split, g.q_cmd);
                    for (i, &k) in g.members.iter().enumerate() {
                        let u = flows[i] / 60000.0 / g.areas[i].max(1e-12);
                        updates.push((k, PortBc::Velocity(dir_vel(self.port_dir[k], self.units.vel_to_lat(u) * f as f64))));
                    }
                }
                SplitMode::Plenum => {
                    let rho = self.units.rho_lat(g.p_cmd * f as f64) as f32;
                    for &k in &g.members {
                        updates.push((k, PortBc::Pressure(rho)));
                    }
                }
            }
        }
        for (k, bc) in updates {
            self.engine.set_port_bc(k, bc);
        }
    }

    /// Measured total flow (L/min) and mean pressure (Pa) over a source's members.
    fn source_measure(&mut self, gi: usize) -> (f64, f64) {
        let members = self.sources[gi].members.clone();
        let (mut q, mut p, mut n_tot) = (0.0, 0.0, 0.0);
        for &k in &members {
            let (rho, u_in, n) = self.port_stats(k);
            q += self.units.vel_to_phys(u_in as f64) * n as f64 * self.units.dx * self.units.dx * 60000.0;
            p += self.units.pressure(rho as f64) * n as f64;
            n_tot += n as f64;
        }
        (q, p / n_tot.max(1.0))
    }

    fn update_sources(&mut self) {
        for gi in 0..self.sources.len() {
            let (q, p_in) = self.source_measure(gi);
            let rho = self.units.rho;
            let g = &mut self.sources[gi];
            let sum_a: f64 = g.areas.iter().sum();
            match (&g.source, g.split) {
                (SourceKind::Flow { .. } | SourceKind::Velocity { .. }, SplitMode::ByArea | SplitMode::Weights) => {}
                (SourceKind::Fan { curve }, SplitMode::ByArea | SplitMode::Weights) => {
                    let target = curve_at(curve, q.max(0.0));
                    let scale = target.abs().max(p_in.abs()).max(1.0);
                    g.q_cmd = (g.q_cmd * (1.0 + 0.1 * ((target - p_in) / scale).clamp(-1.0, 1.0))).max(0.01);
                }
                (SourceKind::Fan { curve }, SplitMode::Plenum) => {
                    let target = curve_at(curve, q.max(0.0));
                    g.p_cmd += 0.2 * (target - g.p_cmd);
                }
                (src, SplitMode::Plenum) => {
                    let q_t = match src {
                        SourceKind::Flow { l_per_min } => *l_per_min,
                        SourceKind::Velocity { speed } => speed * sum_a * 60000.0,
                        _ => q,
                    };
                    let err = (q_t - q) / q_t.abs().max(1e-6);
                    let p_dyn = 0.5 * rho * (q_t / 60000.0 / sum_a.max(1e-12)).powi(2);
                    g.p_cmd += 0.15 * err.clamp(-1.0, 1.0) * g.p_cmd.abs().max(p_dyn);
                }
            }
        }
    }

    pub fn source_reports(&mut self) -> Vec<SourceReport> {
        (0..self.sources.len())
            .map(|gi| {
                let (q, p) = self.source_measure(gi);
                let g = &self.sources[gi];
                SourceReport {
                    name: g.name.clone(),
                    total_flow_l_min: q,
                    pressure_pa: p,
                    split: g.split,
                    commanded: if g.split == SplitMode::Plenum { g.p_cmd } else { g.q_cmd },
                }
            })
            .collect()
    }

    pub fn max_speed(&mut self) -> f32 {
        let cells = &self.cells;
        let mac = self.engine.mac();
        mac.iter().zip(cells).filter(|(_, &c)| c == 0).map(|(m, _)| (m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt()).fold(0.0, f32::max)
    }

    pub fn is_finite(&mut self) -> bool {
        self.engine.mac().iter().all(|m| m[0].is_finite() && m[1].is_finite())
    }

    pub fn report(&mut self) -> Vec<PortReport> {
        (0..self.project_ports.len())
            .map(|k| {
                let (rho, u_in, n) = self.port_stats(k);
                let dx2 = self.units.dx * self.units.dx;
                let flow = self.units.vel_to_phys(u_in as f64) * n as f64 * dx2 * 60000.0;
                PortReport {
                    name: self.project_ports[k].0.clone(),
                    inlet: self.project_ports[k].1.is_inlet(),
                    flow_l_min: flow,
                    pressure_pa: self.units.pressure(rho as f64),
                    cells: n,
                    source: self.project_ports[k].1.source().map(|s| s.to_string()),
                }
            })
            .collect()
    }
}
