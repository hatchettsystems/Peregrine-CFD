//! Project file (TOML): body role rules, ports, fluid, domain and solver settings.
pub mod presets;

use peregrine_core::PortCap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BodyRole {
    /// Hollow duct solid: its channel carries the air, its openings may become ports.
    Duct,
    /// Any other obstacle (nozzle, heater block, fan housing, ...).
    Solid,
    Ignore,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RoleRule {
    pub pattern: String,
    pub role: BodyRole,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometrySettings {
    pub rules: Vec<RoleRule>,
    /// Explicit per-body role, overrides rules.
    #[serde(default)]
    pub overrides: BTreeMap<String, BodyRole>,
}

impl Default for GeometrySettings {
    fn default() -> Self {
        GeometrySettings {
            rules: vec![
                RoleRule { pattern: r"(?i)^duct([_\- ]\w+)?$".into(), role: BodyRole::Duct },
                RoleRule { pattern: r"(?i)(fan|blower)".into(), role: BodyRole::Solid },
            ],
            overrides: BTreeMap::new(),
        }
    }
}

impl GeometrySettings {
    pub fn classify(&self, name: &str) -> BodyRole {
        if let Some(r) = self.overrides.get(name) {
            return *r;
        }
        for rule in &self.rules {
            if Regex::new(&rule.pattern).map_or(false, |re| re.is_match(name)) {
                return rule.role;
            }
        }
        BodyRole::Solid
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PortKind {
    /// Uniform inflow speed (m/s) along the inward normal.
    VelocityInlet { speed: f64 },
    /// Volumetric flow (litres / minute), uniform over the cap.
    FlowInlet { l_per_min: f64 },
    /// Fan: static pressure rise (Pa) versus flow (L/min), decreasing curve.
    FanInlet { curve: Vec<(f64, f64)> },
    /// Static (gauge) pressure in Pa.
    PressureOutlet { pressure: f64 },
    /// Inlet fed by a [`Source`]; `weight` is used by the `Weights` split.
    SourceInlet { source: String, weight: f64 },
}

impl PortKind {
    pub fn is_inlet(&self) -> bool {
        !matches!(self, PortKind::PressureOutlet { .. })
    }
    pub fn source(&self) -> Option<&str> {
        match self {
            PortKind::SourceInlet { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// What drives a port source as a whole.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SourceKind {
    /// Total volumetric flow (L/min) shared by all members.
    Flow { l_per_min: f64 },
    /// Mean speed (m/s) over the combined cross-section of all members.
    Velocity { speed: f64 },
    /// One fan feeding all members: pressure rise (Pa) versus total flow (L/min).
    Fan { curve: Vec<(f64, f64)> },
}

/// How the source's total is shared between its member openings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplitMode {
    /// Each member takes flow in proportion to its cross-section (equal velocity at every member).
    ByArea,
    /// Each member takes flow in proportion to its manual weight.
    Weights,
    /// Members share one plenum pressure; the flow divides by the resistance of each duct.
    Plenum,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    pub name: String,
    pub kind: SourceKind,
    pub split: SplitMode,
    /// Name of the library fan the curve was taken from, if any (for estimate warnings).
    #[serde(default)]
    pub preset: Option<String>,
    /// Number of flow trajectories (streamlines) drawn from this source's inlets.
    #[serde(default = "default_streamlines")]
    pub streamlines: usize,
}

fn default_streamlines() -> usize {
    40
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Port {
    pub name: String,
    pub body: String,
    pub kind: PortKind,
    pub cap: PortCap,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Fluid {
    pub density: f64,
    pub kinematic_viscosity: f64,
}
impl Default for Fluid {
    fn default() -> Self { Fluid { density: 1.2, kinematic_viscosity: 1.5e-5 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaceBc { Open, Wall }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DomainSettings {
    /// Padding (mm) around the bounding box of all active bodies.
    pub padding_mm: f64,
    /// Faces in order -X, +X, -Y, +Y, -Z, +Z.
    pub faces: [FaceBc; 6],
}
impl Default for DomainSettings {
    fn default() -> Self { DomainSettings { padding_mm: 10.0, faces: [FaceBc::Open; 6] } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverSettings {
    /// Cell size in mm.
    pub cell_size_mm: f64,
    /// Largest expected speed (m/s); sets the time step.
    pub reference_speed: f64,
    pub max_steps: usize,
    pub smagorinsky: f64,
}
impl Default for SolverSettings {
    fn default() -> Self {
        SolverSettings { cell_size_mm: 0.5, reference_speed: 20.0, max_steps: 20000, smagorinsky: 0.12 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub name: String,
    /// STEP or STL file, relative to the project file.
    pub geometry_file: Option<PathBuf>,
    #[serde(default)]
    pub geometry: GeometrySettings,
    #[serde(default)]
    pub fluid: Fluid,
    #[serde(default)]
    pub domain: DomainSettings,
    #[serde(default)]
    pub solver: SolverSettings,
    #[serde(default)]
    pub ports: Vec<Port>,
    #[serde(default)]
    pub sources: Vec<Source>,
}

impl Project {
    /// Indices of the ports that belong to `source`.
    pub fn members(&self, source: &str) -> Vec<usize> {
        (0..self.ports.len()).filter(|&i| self.ports[i].kind.source() == Some(source)).collect()
    }

    /// Human-readable problems that would stop a run (empty = ok).
    pub fn validate(&self) -> Vec<String> {
        let mut p = vec![];
        for (i, a) in self.ports.iter().enumerate() {
            if self.ports[..i].iter().any(|b| b.name == a.name) {
                p.push(format!("two ports are named '{}'", a.name));
            }
            if let PortKind::SourceInlet { source, weight } = &a.kind {
                match self.sources.iter().find(|g| &g.name == source) {
                    None => p.push(format!("port '{}' belongs to unknown source '{source}'", a.name)),
                    Some(g) if g.split == SplitMode::Weights && *weight <= 0.0 => p.push(format!("port '{}' has a non-positive weight", a.name)),
                    _ => {}
                }
            }
        }
        for g in &self.sources {
            if self.members(&g.name).is_empty() {
                p.push(format!("source '{}' has no inlet ports assigned", g.name));
            }
        }
        if !self.ports.iter().any(|p| p.kind.is_inlet()) {
            p.push("define at least one inlet".into());
        }
        if !self.ports.iter().any(|p| !p.kind.is_inlet()) {
            p.push("define at least one outlet".into());
        }
        p
    }

    pub fn new(name: &str) -> Self {
        Project {
            name: name.into(),
            geometry_file: None,
            geometry: Default::default(),
            fluid: Default::default(),
            domain: Default::default(),
            solver: Default::default(),
            ports: vec![],
            sources: vec![],
        }
    }
    pub fn load(path: &Path) -> anyhow::Result<Project> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        std::fs::write(path, toml::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roles() {
        let g = GeometrySettings::default();
        for n in ["duct", "duct_1", "Duct_L", "duct_R", "DUCT-2"] {
            assert_eq!(g.classify(n), BodyRole::Duct, "{n}");
        }
        assert_eq!(g.classify("nozzle"), BodyRole::Solid);
        assert_eq!(g.classify("ductwork_cover"), BodyRole::Solid);
    }
    #[test]
    fn source_validation() {
        let cap = PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0; 3], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]] };
        let mut p = Project::new("g");
        p.sources.push(Source { name: "fan".into(), kind: SourceKind::Flow { l_per_min: 10.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
        p.ports.push(Port { name: "a".into(), body: "d".into(), kind: PortKind::SourceInlet { source: "fan".into(), weight: 1.0 }, cap: cap.clone() });
        p.ports.push(Port { name: "o".into(), body: "d".into(), kind: PortKind::PressureOutlet { pressure: 0.0 }, cap: cap.clone() });
        assert!(p.validate().is_empty(), "{:?}", p.validate());
        assert_eq!(p.members("fan"), vec![0]);
        p.sources.push(Source { name: "empty".into(), kind: SourceKind::Flow { l_per_min: 1.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
        assert!(p.validate().iter().any(|m| m.contains("no inlet ports")));
        let s = toml::to_string_pretty(&p).unwrap();
        let q: Project = toml::from_str(&s).unwrap();
        assert_eq!(q.sources.len(), 2);
    }

    #[test]
    fn roundtrip() {
        let mut p = Project::new("t");
        p.ports.push(Port { name: "in".into(), body: "duct".into(), kind: PortKind::FanInlet { curve: vec![(0.0, 300.0), (100.0, 0.0)] },
            cap: PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]] } });
        let s = toml::to_string_pretty(&p).unwrap();
        let q: Project = toml::from_str(&s).unwrap();
        assert_eq!(q.ports.len(), 1);
    }
}
