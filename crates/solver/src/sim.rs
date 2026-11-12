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
    /// Speed (m/s) of the mean velocity vector over the opening's cells.
    pub mean_speed: f64,
    /// Angle (degrees) between the mean velocity and the opening's normal (flow direction relative to the opening).
    pub angle_deg: f64,
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
    /// Per-member velocity correction so the delivered flow matches the commanded share.
    gain: Vec<f64>,
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
    /// Reference speed (m/s) that fixed the time step; may be higher than the project's setting.
    pub ref_speed: f64,
    pub dims: [usize; 3],
    /// Grid origin (mm) and cell size (mm), for reporting where things happen.
    origin_mm: [f64; 3],
    dx_mm: f64,
    face_open: [bool; 6],
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
        Self::new_with_ref(project, domain, backend, project.solver.reference_speed)
    }

    /// Rough peak speed (m/s) expected in the domain: the fastest inlet or the mean outlet speed.
    pub fn estimate_peak_speed(project: &Project, domain: &Domain) -> f64 {
        let dx_m = domain.grid.dx * 1e-3;
        let area = |k: usize| domain.port_cell_counts[k].max(1) as f64 * dx_m * dx_m;
        let (mut peak, mut q_total, mut a_out) = (0.0f64, 0.0f64, 0.0f64); // q in m^3/s
        for (k, p) in project.ports.iter().enumerate() {
            match &p.kind {
                PortKind::VelocityInlet { speed } => { peak = peak.max(*speed); q_total += speed * area(k); }
                PortKind::FlowInlet { l_per_min } => { let q = l_per_min / 60000.0; peak = peak.max(q / area(k)); q_total += q; }
                PortKind::FanInlet { curve } => { let q = curve.last().map_or(0.0, |c| c.0) * 0.7 / 60000.0; peak = peak.max(q / area(k)); q_total += q; }
                PortKind::PressureOutlet { .. } => a_out += area(k),
                PortKind::SourceInlet { .. } => {}
            }
        }
        for src in &project.sources {
            let members = project.members(&src.name);
            let sum_a: f64 = members.iter().map(|&k| area(k)).sum();
            let q = match &src.kind {
                SourceKind::Flow { l_per_min } => l_per_min / 60000.0,
                SourceKind::Velocity { speed } => speed * sum_a,
                SourceKind::Fan { curve } => curve.last().map_or(0.0, |c| c.0) * 0.7 / 60000.0,
            };
            q_total += q;
            // forced splits give each opening q*share/area; manual weights can concentrate flow
            let worst = members.iter().map(|&k| {
                let w = match &project.ports[k].kind { PortKind::SourceInlet { weight, .. } => *weight, _ => 1.0 };
                let wsum: f64 = members.iter().map(|&j| match &project.ports[j].kind { PortKind::SourceInlet { weight, .. } => *weight, _ => 1.0 }).sum();
                match src.split { SplitMode::Weights => q * w / wsum.max(1e-9) / area(k), _ => q / sum_a.max(1e-12) }
            }).fold(0.0, f64::max);
            peak = peak.max(if src.split == SplitMode::Plenum { 2.0 * q / sum_a.max(1e-12) } else { worst });
        }
        if a_out > 0.0 { peak = peak.max(q_total / a_out); }
        peak
    }

    /// Like [`Simulation::new`] but with an explicit minimum reference speed. The speed is raised
    /// automatically to 2.5x the estimated peak so lattice velocities stay in the low-Mach range.
    pub fn new_with_ref(project: &Project, domain: &Domain, backend: Backend, ref_speed: f64) -> anyhow::Result<Simulation> {
        let g = &domain.grid;
        let auto = (2.5 * Self::estimate_peak_speed(project, domain)).max(3.0);
        let ref_speed = if ref_speed <= 0.0 { auto } else { ref_speed.max(auto.min(2.5 * Self::estimate_peak_speed(project, domain))) };
        let units = Units::new(g.dx * 1e-3, ref_speed, 0.08, project.fluid.density, project.fluid.kinematic_viscosity).with_tau_floor(project.solver.min_tau);
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
            infos.push(PortInfo { bc, inward: snap(inward_f), normal: { let l = (inward_f[0] * inward_f[0] + inward_f[1] * inward_f[1] + inward_f[2] * inward_f[2]).sqrt().max(1e-12); [(inward_f[0] / l) as f32, (inward_f[1] / l) as f32, (inward_f[2] / l) as f32] } });
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
            regularized: true,
            sponge_cells: if project.domain.padding_mm > 0.0 { ((0.6 * project.domain.padding_mm / g.dx).round() as usize).clamp(4, 40) } else { 0 },
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
            sources.push(SourceRt { name: grp.name.clone(), source: grp.kind.clone(), split: grp.split, gain: vec![1.0; members.len()], members, weights, areas, q_cmd, p_cmd });
        }
        let mut sim = Simulation { engine, units, ref_speed, dims: g.dims, origin_mm: g.origin, dx_mm: g.dx, face_open: project.domain.faces.map(|f| f == FaceBc::Open), cells: domain.cells.clone(), ports: infos, port_cells, project_ports: meta, port_dir: dirs, ramp_base, sources };
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
        let inw = self.ports[k].inward_unit();
        let list: Vec<usize> = self.port_cells.iter().filter(|c| c.1 == k).map(|c| c.2).collect();
        let vals = self.engine.read_cells(&list);
        let (mut r, mut u) = (0.0, 0.0);
        for m in vals.iter().copied() {
            r += m[0];
            u += m[1] * inw[0] + m[2] * inw[1] + m[3] * inw[2];
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
                        updates.push((k, PortBc::Velocity(dir_vel(self.port_dir[k], self.units.vel_to_lat(u) * g.gain[i] * f as f64))));
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
            // forced splits: nudge each inlet's velocity until the flow it delivers matches its share
            if self.sources[gi].split != SplitMode::Plenum {
                let members = self.sources[gi].members.clone();
                let flows = split_flows(&self.sources[gi].areas, &self.sources[gi].weights, self.sources[gi].split, self.sources[gi].q_cmd);
                for (i, &k) in members.iter().enumerate() {
                    let (_, u_in, n) = self.port_stats(k);
                    let meas = self.units.vel_to_phys(u_in as f64) * n as f64 * self.units.dx * self.units.dx * 60000.0;
                    if flows[i] > 1e-9 && meas > 1e-9 {
                        let err = ((flows[i] - meas) / flows[i]).clamp(-0.5, 0.5);
                        let g = &mut self.sources[gi].gain[i];
                        *g = (*g * (1.0 + 0.25 * err)).clamp(0.5, 3.0);
                    }
                }
            }
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

    /// Describes where the fastest (or first non-finite) fluid cell is, to help find the cause of an instability.
    pub fn hotspot(&mut self) -> String {
        let [nx, ny, nz] = self.dims;
        let u = self.units;
        let cells = self.cells.clone();
        let mac = self.engine.mac();
        let mut best = (-1.0f32, 0usize, false);
        for (i, m) in mac.iter().enumerate() {
            if cells[i] != 0 { continue; }
            let finite = m[0].is_finite() && m[1].is_finite() && m[2].is_finite() && m[3].is_finite();
            let sp = if finite { (m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() } else { f32::INFINITY };
            if (!finite && !best.2) || (finite == !best.2 && sp > best.0) { best = (sp, i, !finite); }
        }
        let i = best.1;
        let (x, y, z) = (i % nx, (i / nx) % ny, i / (nx * ny));
        let mm = |a: usize, k: usize| self.origin_mm[k] + (a as f64 + 0.5) * self.dx_mm;
        // proximity checks (within 3 cells)
        let faces = [(x, 0, "-X"), (nx - 1 - x, 1, "+X"), (y, 2, "-Y"), (ny - 1 - y, 3, "+Y"), (z, 4, "-Z"), (nz - 1 - z, 5, "+Z")];
        let mut near = String::from("in free fluid");
        for (d, f, name) in faces {
            if d <= 3 { near = format!("next to the {} domain face ({})", name, if self.face_open[f] { "open" } else { "wall" }); break; }
        }
        if near == "in free fluid" {
            'search: for dz in -3i32..=3 { for dy in -3i32..=3 { for dx in -3i32..=3 {
                let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                let c = cells[xx as usize + nx * (yy as usize + ny * zz as usize)];
                if c >= 2 {
                    let name = &self.project_ports[(c - 2) as usize].0;
                    near = format!("next to port '{name}'");
                    break 'search;
                }
            } } }
        }
        if near == "in free fluid" {
            'w: for dz in -2i32..=2 { for dy in -2i32..=2 { for dx in -2i32..=2 {
                let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                if cells[xx as usize + nx * (yy as usize + ny * zz as usize)] == 1 { near = "next to a wall".into(); break 'w; }
            } } }
        }
        let speed = if best.2 { "non-finite".to_string() } else { format!("{:.1} m/s", u.vel_to_phys(best.0 as f64)) };
        format!("{} at ({:.1}, {:.1}, {:.1}) mm, {}", speed, mm(x, 0), mm(y, 1), mm(z, 2), near)
    }

    /// True when the run has blown up or lattice speeds are far into the compressible range.
    pub fn unstable(&mut self) -> bool {
        !self.is_finite() || self.max_speed() > 0.35
    }

    pub fn is_finite(&mut self) -> bool {
        self.engine.mac().iter().all(|m| m[0].is_finite() && m[1].is_finite())
    }

    /// Mean velocity vector (m/s) over the cells of port `k`.
    fn port_mean_velocity(&mut self, k: usize) -> [f64; 3] {
        let idxs: Vec<usize> = self.port_cells.iter().filter(|c| c.1 == k).map(|c| c.0).collect();
        let u = self.units;
        let vals = self.engine.read_cells(&idxs);
        let mut v = [0.0f64; 3];
        for m in &vals {
            for a in 0..3 { v[a] += u.vel_to_phys(m[a + 1] as f64); }
        }
        let n = idxs.len().max(1) as f64;
        [v[0] / n, v[1] / n, v[2] / n]
    }

    pub fn report(&mut self) -> Vec<PortReport> {
        (0..self.project_ports.len())
            .map(|k| {
                let v = self.port_mean_velocity(k);
                let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                // flow direction relative to the opening's axis (inward for inlets, outward for outlets)
                let d = self.port_dir[k];
                let sign = if self.project_ports[k].1.is_inlet() { 1.0 } else { -1.0 };
                let cosang = if speed > 1e-9 { sign * (v[0] * d[0] + v[1] * d[1] + v[2] * d[2]) / speed / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12) } else { 1.0 };
                let angle_deg = cosang.clamp(-1.0, 1.0).acos().to_degrees();
                let (rho, u_in, n) = self.port_stats(k);
                let dx2 = self.units.dx * self.units.dx;
                let flow = self.units.vel_to_phys(u_in as f64) * n as f64 * dx2 * 60000.0;
                PortReport {
                    name: self.project_ports[k].0.clone(),
                    inlet: self.project_ports[k].1.is_inlet(),
                    flow_l_min: flow,
                    pressure_pa: self.units.pressure(rho as f64),
                    cells: n,
                    mean_speed: speed,
                    angle_deg,
                    source: self.project_ports[k].1.source().map(|s| s.to_string()),
                }
            })
            .collect()
    }
}
