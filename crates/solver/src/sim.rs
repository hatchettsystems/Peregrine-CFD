use crate::lbm::{Lbm, LbmConfig, PortBc, PortInfo};
use crate::units::Units;
use peregrine_project::{FaceBc, PortKind, Project};
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
}

const RAMP_STEPS: usize = 1500;

pub struct Simulation {
    pub lbm: Lbm,
    pub units: Units,
    project_ports: Vec<(String, PortKind, f64)>, // name, kind, cap area m^2
    port_dir: Vec<[f64; 3]>,
    /// Fixed inlet velocity vectors (lattice) that are ramped in over `RAMP_STEPS`.
    ramp_base: Vec<Option<[f32; 3]>>,
    pub domain_dims: [usize; 3],
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

impl Simulation {
    pub fn new(project: &Project, domain: &Domain) -> anyhow::Result<Simulation> {
        let g = &domain.grid;
        let units = Units::new(g.dx * 1e-3, project.solver.reference_speed, 0.05, project.fluid.density, project.fluid.kinematic_viscosity);
        let mut infos = vec![];
        let mut meta = vec![];
        let mut dirs = vec![];
        for p in &project.ports {
            let inward_f = [-p.cap.normal[0], -p.cap.normal[1], -p.cap.normal[2]];
            let area = p.cap.area() * 1e-6;
            let bc = match &p.kind {
                PortKind::VelocityInlet { speed } => PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(*speed))),
                PortKind::FlowInlet { l_per_min } => {
                    let u = l_per_min / 60000.0 / area;
                    PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(u)))
                }
                PortKind::FanInlet { curve } => {
                    // start from the free-delivery end of the curve, the controller adjusts it
                    let q0 = curve.last().map_or(1.0, |c| c.0) * 0.5;
                    let u = q0 / 60000.0 / area;
                    PortBc::Velocity(dir_vel(inward_f, units.vel_to_lat(u)))
                }
                PortKind::PressureOutlet { pressure } => PortBc::Pressure(units.rho_lat(*pressure) as f32),
            };
            infos.push(PortInfo { bc, inward: snap(inward_f) });
            meta.push((p.name.clone(), p.kind.clone(), area));
            dirs.push(inward_f);
        }
        let open_faces = project.domain.faces.map(|f| f == FaceBc::Open);
        let cfg = LbmConfig {
            dims: g.dims,
            cells: domain.cells.clone(),
            tau0: units.tau_used() as f32,
            smagorinsky: project.solver.smagorinsky as f32,
            open_faces,
            body_force: [0.0; 3],
            ports: infos,
        };
        let ramp_base = cfg.ports.iter().zip(&meta).map(|(p, m)| match (p.bc, &m.1) {
            (PortBc::Velocity(u), PortKind::VelocityInlet { .. } | PortKind::FlowInlet { .. }) => Some(u),
            _ => None,
        }).collect();
        let mut sim = Simulation { lbm: Lbm::new(cfg), units, project_ports: meta, port_dir: dirs, ramp_base, domain_dims: g.dims };
        sim.apply_ramp();
        Ok(sim)
    }

    fn apply_ramp(&mut self) {
        let t = (self.lbm.step_count as f32 / RAMP_STEPS as f32).min(1.0);
        let f = 0.5 - 0.5 * (std::f32::consts::PI * t).cos();
        for k in 0..self.ramp_base.len() {
            if let Some(u) = self.ramp_base[k] {
                self.lbm.set_port_bc(k, PortBc::Velocity([u[0] * f, u[1] * f, u[2] * f]));
            }
        }
    }

    /// Advance `n` steps, updating fan-curve ports every 100 steps.
    pub fn advance(&mut self, n: usize) {
        for s in 0..n {
            if self.lbm.step_count % 25 == 0 && self.lbm.step_count <= RAMP_STEPS {
                self.apply_ramp();
            }
            self.lbm.step();
            if self.lbm.step_count % 100 == 0 && s + 1 < n + 1 {
                self.update_fans();
            }
        }
    }

    fn update_fans(&mut self) {
        for k in 0..self.project_ports.len() {
            if let PortKind::FanInlet { curve } = self.project_ports[k].1.clone() {
                let area = self.project_ports[k].2;
                let (rho, u_in, _) = self.lbm.port_stats(k);
                let u_phys = self.units.vel_to_phys(u_in as f64);
                let q = u_phys * area * 60000.0;
                let p_in = self.units.pressure(rho as f64);
                let target = curve_at(&curve, q.max(0.0));
                let scale = target.abs().max(p_in.abs()).max(1.0);
                let factor = 1.0 + 0.1 * ((target - p_in) / scale).clamp(-1.0, 1.0);
                let new_u = (u_phys * factor).max(1e-3);
                let d = self.port_dir[k];
                self.lbm.set_port_bc(k, PortBc::Velocity(dir_vel(d, self.units.vel_to_lat(new_u))));
            }
        }
    }

    pub fn report(&self) -> Vec<PortReport> {
        (0..self.project_ports.len())
            .map(|k| {
                let (rho, u_in, n) = self.lbm.port_stats(k);
                let dx2 = self.units.dx * self.units.dx;
                let flow = self.units.vel_to_phys(u_in as f64) * n as f64 * dx2 * 60000.0;
                PortReport {
                    name: self.project_ports[k].0.clone(),
                    inlet: self.project_ports[k].1.is_inlet(),
                    flow_l_min: flow,
                    pressure_pa: self.units.pressure(rho as f64),
                    cells: n,
                }
            })
            .collect()
    }
}

fn dir_vel(d: [f64; 3], speed_lat: f64) -> [f32; 3] {
    let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
    [(d[0] / l * speed_lat) as f32, (d[1] / l * speed_lat) as f32, (d[2] / l * speed_lat) as f32]
}
