//! Project file (TOML): body role rules, ports, fluid, domain and solver settings.
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
}

impl PortKind {
    pub fn is_inlet(&self) -> bool {
        !matches!(self, PortKind::PressureOutlet { .. })
    }
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
}

impl Project {
    pub fn new(name: &str) -> Self {
        Project {
            name: name.into(),
            geometry_file: None,
            geometry: Default::default(),
            fluid: Default::default(),
            domain: Default::default(),
            solver: Default::default(),
            ports: vec![],
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
    fn roundtrip() {
        let mut p = Project::new("t");
        p.ports.push(Port { name: "in".into(), body: "duct".into(), kind: PortKind::FanInlet { curve: vec![(0.0, 300.0), (100.0, 0.0)] },
            cap: PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]] } });
        let s = toml::to_string_pretty(&p).unwrap();
        let q: Project = toml::from_str(&s).unwrap();
        assert_eq!(q.ports.len(), 1);
    }
}
