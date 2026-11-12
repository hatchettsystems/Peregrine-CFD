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

/// A duct is one or more bodies (parts) treated as a single air passage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuctDef {
    pub name: String,
    /// Names of the bodies that make up this duct.
    pub parts: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometrySettings {
    /// Name patterns that only HIGHLIGHT likely duct bodies; the user decides what is a duct.
    pub rules: Vec<RoleRule>,
    /// Legacy per-body role overrides (used for hints and old projects).
    #[serde(default)]
    pub overrides: BTreeMap<String, BodyRole>,
    /// Bodies excluded from the simulation entirely.
    #[serde(default)]
    pub ignored: std::collections::BTreeSet<String>,
}

impl Default for GeometrySettings {
    fn default() -> Self {
        GeometrySettings {
            ignored: Default::default(),
            rules: vec![
                RoleRule { pattern: r"(?i)(duct|shroud|scoop|tunnel|channel)".into(), role: BodyRole::Duct },
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

fn default_true() -> bool { true }
fn default_gap_seal() -> f64 { 0.9 }
fn default_seam_tolerance() -> f64 {
    0.5
}

fn default_outlet_clearance() -> f64 {
    30.0
}

fn default_min_tau() -> f64 {
    0.51
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
#[serde(default)]
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
#[serde(default)]
pub struct DomainSettings {
    /// Padding (mm) around the bounding box of all active bodies.
    pub padding_mm: f64,
    /// The box is extended this far (mm) beyond every outlet along its outward normal so jets can
    /// develop and converge before they reach the domain boundary.
    #[serde(default = "default_outlet_clearance")]
    pub outlet_clearance_mm: f64,
    /// Gaps up to this size (mm) between two parts of the same duct are sealed. Larger gaps leak and are reported.
    #[serde(default = "default_seam_tolerance")]
    pub seam_tolerance_mm: f64,
    /// Fan mount seal (mm): fluid cells within this distance of the rim of every inlet are closed, like the housing
    /// of a mounted fan. Use when a fan mount leaks around the inlet because the fan body is not in the model. 0 = off.
    #[serde(default)]
    pub inlet_collar_mm: f64,
    /// Simulate only this box (xmin, ymin, zmin, xmax, ymax, zmax in mm) instead of the whole model: a zoom run.
    #[serde(default)]
    pub crop: Option<[f64; 6]>,
    /// Gap sealer (mm): any air gap this thin between two solid surfaces (between parts, or inside one part) is closed.
    /// Real assemblies are tight where the CAD has small clearances. 0 = off.
    #[serde(default = "default_gap_seal")]
    pub gap_seal_mm: f64,
    /// Automatically close small leaks (about 1 mm wide at most; anything larger is intentional) found between the inlets and the edge of the domain.
    /// On by default; what was closed is reported in the Exceptions tab. Set false to turn off.
    #[serde(default = "default_true")]
    pub auto_seal_leaks: bool,
    /// Faces in order -X, +X, -Y, +Y, -Z, +Z.
    pub faces: [FaceBc; 6],
}
impl Default for DomainSettings {
    fn default() -> Self { DomainSettings { padding_mm: 10.0, outlet_clearance_mm: 30.0, seam_tolerance_mm: 0.5, inlet_collar_mm: 0.0, crop: None, gap_seal_mm: 0.9, auto_seal_leaks: true, faces: [FaceBc::Open; 6] } }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SolverSettings {
    /// Cell size in mm.
    pub cell_size_mm: f64,
    /// Largest expected speed (m/s); sets the time step. 0 = choose automatically from the sources.
    pub reference_speed: f64,
    pub max_steps: usize,
    pub smagorinsky: f64,
    /// Lowest relaxation time the solver will use. Lower = less artificial viscosity but less stable.
    #[serde(default = "default_min_tau")]
    pub min_tau: f64,
    /// End the run early once flows and pressures have stopped changing.
    pub stop_when_converged: bool,
}
impl Default for SolverSettings {
    fn default() -> Self {
        SolverSettings { cell_size_mm: 0.5, reference_speed: 0.0, max_steps: 20000, smagorinsky: 0.12, min_tau: 0.51, stop_when_converged: true }
    }
}

/// A closed opening (screw hole, vent, unused slot): its cap is filled solid so no air passes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Blocker {
    pub name: String,
    pub body: String,
    pub cap: PortCap,
}

/// A small solid sphere that seals a leak at its narrowest point.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plug {
    pub name: String,
    pub center: [f64; 3],
    pub radius_mm: f64,
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
    /// Ducts defined by the user (each made of one or more bodies).
    #[serde(default)]
    pub ducts: Vec<DuctDef>,
    #[serde(default)]
    pub ports: Vec<Port>,
    #[serde(default)]
    pub sources: Vec<Source>,
    /// Openings the user has shut (screw holes and the like).
    #[serde(default)]
    pub blockers: Vec<Blocker>,
    /// Leaks the user sealed with a plug.
    #[serde(default)]
    pub plugs: Vec<Plug>,
}

impl Project {
    /// The duct a body belongs to, if the user has put it in one.
    pub fn duct_of(&self, body: &str) -> Option<&DuctDef> {
        self.ducts.iter().find(|d| d.parts.iter().any(|p| p == body))
    }

    /// Role of a body from the user's choices (ducts, ignore list); everything else is a solid obstacle.
    pub fn role_of(&self, body: &str) -> BodyRole {
        if self.duct_of(body).is_some() {
            BodyRole::Duct
        } else if self.geometry.ignored.contains(body) {
            BodyRole::Ignore
        } else {
            BodyRole::Solid
        }
    }

    /// True when the body's name looks like a duct and the user has not assigned it yet (a highlight hint).
    pub fn is_suggested(&self, body: &str) -> bool {
        self.duct_of(body).is_none() && !self.geometry.ignored.contains(body) && self.geometry.classify(body) == BodyRole::Duct
    }

    /// Ducts to use for opening detection. Projects (or tests) that never defined ducts fall back to one duct
    /// per body whose name looks like a duct, so older files and scripts keep working.
    pub fn effective_ducts(&self, body_names: &[String]) -> Vec<DuctDef> {
        if !self.ducts.is_empty() {
            return self.ducts.clone();
        }
        body_names
            .iter()
            .filter(|n| self.geometry.classify(n) == BodyRole::Duct && !self.geometry.ignored.contains(*n))
            .map(|n| DuctDef { name: n.clone(), parts: vec![n.clone()] })
            .collect()
    }

    /// Put `parts` into a new duct named `name` (removing them from other ducts). Returns the final name.
    pub fn define_duct(&mut self, name: &str, parts: &[String]) -> String {
        for d in &mut self.ducts {
            d.parts.retain(|p| !parts.contains(p));
        }
        self.ducts.retain(|d| !d.parts.is_empty());
        let mut final_name = name.trim().to_string();
        if final_name.is_empty() {
            final_name = "duct".into();
        }
        let base = final_name.clone();
        let mut n = 2;
        while self.ducts.iter().any(|d| d.name == final_name) {
            final_name = format!("{base} {n}");
            n += 1;
        }
        for p in parts {
            self.geometry.ignored.remove(p);
        }
        self.ducts.push(DuctDef { name: final_name.clone(), parts: parts.to_vec() });
        final_name
    }

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
            ducts: vec![],
            ports: vec![],
            sources: vec![],
            blockers: vec![],
            plugs: vec![],
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
        // hints are deliberately loose: the user makes the final call
        assert_eq!(g.classify("ductwork_cover"), BodyRole::Duct);
        assert_eq!(g.classify("Part Cooling Shroud"), BodyRole::Duct);
        assert_eq!(g.classify("heater block"), BodyRole::Solid);
    }
    #[test]
    fn source_validation() {
        let cap = PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0; 3], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]], triangles: vec![], holes: vec![] };
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
    fn ducts_are_user_defined_and_names_only_suggest() {
        let mut p = Project::new("d");
        // a name hint highlights but does not decide
        assert!(p.is_suggested("duct_L"));
        assert_eq!(p.role_of("duct_L"), BodyRole::Solid);
        let n = p.define_duct("left", &["a1276918-Duct".to_string(), "lip".to_string()]);
        assert_eq!(n, "left");
        assert_eq!(p.role_of("a1276918-Duct"), BodyRole::Duct);
        assert_eq!(p.role_of("lip"), BodyRole::Duct);
        assert_eq!(p.duct_of("lip").unwrap().parts.len(), 2);
        assert!(!p.is_suggested("a1276918-Duct"));
        // a part can only live in one duct
        p.define_duct("other", &["lip".to_string()]);
        assert_eq!(p.duct_of("lip").unwrap().name, "other");
        assert_eq!(p.duct_of("a1276918-Duct").unwrap().name, "left");
        // duplicate names get a suffix
        assert_eq!(p.define_duct("left", &["x".to_string()]), "left 2");
        // fallback for projects that never defined ducts
        let q = Project::new("old");
        let names = vec!["duct_1".to_string(), "nozzle".to_string()];
        let e = q.effective_ducts(&names);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0].name, "duct_1");
    }

    #[test]
    fn curved_cap_survives_save_and_load() {
        let tri = [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.5, 1.0, 1.5]];
        let cap = PortCap { origin: [0.1, 0.9, 0.4], normal: [1.0, 0.0, 0.0], polygon: vec![tri[0], tri[1], tri[2]], triangles: vec![tri], holes: vec![] };
        let mut p = Project::new("c");
        p.ports.push(Port { name: "scoop".into(), body: "duct".into(), kind: PortKind::PressureOutlet { pressure: 0.0 }, cap });
        let q: Project = toml::from_str(&toml::to_string_pretty(&p).unwrap()).unwrap();
        assert!(q.ports[0].cap.is_surface());
        assert_eq!(q.ports[0].cap.triangles.len(), 1);
        assert!((q.ports[0].cap.area() - p.ports[0].cap.area()).abs() < 1e-12);
    }

    #[test]
    fn roundtrip() {
        let mut p = Project::new("t");
        p.ports.push(Port { name: "in".into(), body: "duct".into(), kind: PortKind::FanInlet { curve: vec![(0.0, 300.0), (100.0, 0.0)] },
            cap: PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]], triangles: vec![], holes: vec![] } });
        let s = toml::to_string_pretty(&p).unwrap();
        let q: Project = toml::from_str(&s).unwrap();
        assert_eq!(q.ports.len(), 1);
    }
}
