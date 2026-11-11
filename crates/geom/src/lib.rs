//! Loading bodies from geometry files and assembling the voxel domain for a project.
use anyhow::{bail, Context, Result};
use peregrine_core::{find_port_candidates, stl, Aabb, PortCandidate, TriMesh};
use peregrine_project::{BodyRole, Project};
use peregrine_voxel::{build_domain, Domain, PortSpec};
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub role: BodyRole,
    pub mesh: TriMesh,
}

/// CAD exporters append instance counters such as `duct:1`; drop them so name rules match.
fn clean_name(n: &str) -> String {
    match n.rsplit_once(':') {
        Some((head, tail)) if !head.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => head.to_string(),
        _ => n.to_string(),
    }
}

/// Load bodies. A `.stl` file is one body named after the file; a directory loads every `.stl` in it.
/// STEP files (`.step`/`.stp`) are read through OpenCASCADE: every leaf body of the assembly is returned.
pub fn load_bodies(path: &Path, project: &Project) -> Result<Vec<Body>> {
    let mut files = vec![];
    if path.is_dir() {
        for e in std::fs::read_dir(path)? {
            let p = e?.path();
            if p.extension().map_or(false, |x| ["stl", "step", "stp"].iter().any(|e| x.eq_ignore_ascii_case(e))) {
                files.push(p);
            }
        }
        files.sort();
    } else {
        files.push(path.to_path_buf());
    }
    let mut bodies = vec![];
    for f in files {
        let ext = f.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "stl" => {
                let name = f.file_stem().and_then(|s| s.to_str()).unwrap_or("body").to_string();
                let mesh = stl::read_stl(&f).with_context(|| format!("reading {}", f.display()))?;
                let role = project.geometry.classify(&name);
                bodies.push(Body { name, role, mesh });
            }
            "step" | "stp" => {
                let defl = (project.solver.cell_size_mm * 0.2).clamp(0.01, 0.1);
                for b in peregrine_occt::read_step(&f, defl).with_context(|| format!("reading {}", f.display()))? {
                    let name = clean_name(&b.name);
                    let role = project.geometry.classify(&name);
                    bodies.push(Body { name, role, mesh: b.mesh });
                }
            }
            _ => bail!("unsupported geometry file: {}", f.display()),
        }
    }
    Ok(bodies)
}

/// Port candidates (planar holes) of every Duct body, as (body name, candidate).
/// `hide_round_below`: skip small round holes (screw holes) with an equivalent diameter up to this (mm).
pub fn duct_port_candidates(bodies: &[Body], min_area: f64, hide_round_below: Option<f64>) -> Vec<(String, PortCandidate)> {
    let mut out = vec![];
    for b in bodies.iter().filter(|b| b.role == BodyRole::Duct) {
        for c in find_port_candidates(&b.mesh, min_area) {
            if hide_round_below.map_or(false, |d| c.is_screw_hole(d)) {
                continue;
            }
            out.push((b.name.clone(), c));
        }
    }
    out
}

pub fn domain_bounds(bodies: &[Body], padding: f64) -> Aabb {
    let mut b = Aabb::empty();
    for body in bodies.iter().filter(|b| b.role != BodyRole::Ignore) {
        b.union(&body.mesh.bounds());
    }
    b.expanded(padding)
}

/// Domain box: all active bodies plus padding, extended beyond every outlet along its outward normal
/// by `outlet_clearance_mm` so free jets have room to develop.
pub fn project_bounds(project: &Project, bodies: &[Body]) -> Aabb {
    let mut b = domain_bounds(bodies, project.domain.padding_mm);
    for p in project.ports.iter().filter(|p| !p.kind.is_inlet()) {
        let c = project.domain.outlet_clearance_mm;
        for v in &p.cap.polygon {
            let far = [v[0] + p.cap.normal[0] * c, v[1] + p.cap.normal[1] * c, v[2] + p.cap.normal[2] * c];
            b.grow(far);
        }
    }
    b
}

pub fn build_project_domain(project: &Project, bodies: &[Body]) -> Result<Domain> {
    let active: Vec<&Body> = bodies.iter().filter(|b| b.role != BodyRole::Ignore).collect();
    if active.is_empty() {
        bail!("no active bodies");
    }
    if project.ports.is_empty() {
        bail!("project has no ports; define at least one inlet and one outlet");
    }
    let meshes: Vec<&TriMesh> = active.iter().map(|b| &b.mesh).collect();
    let ports: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet() }).collect();
    let bounds = project_bounds(project, bodies);
    let d = build_domain(&meshes, &ports, &bounds, project.solver.cell_size_mm);
    for (p, n) in project.ports.iter().zip(&d.port_cell_counts) {
        if *n == 0 {
            bail!("port '{}' covers no cells at {} mm; refine the grid or check the cap", p.name, project.solver.cell_size_mm);
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn step_names_are_cleaned_and_classified() {
        assert_eq!(clean_name("duct:1"), "duct");
        assert_eq!(clean_name("duct_L:12"), "duct_L");
        assert_eq!(clean_name("a:b"), "a:b");
        let p = Project::new("t");
        assert_eq!(p.geometry.classify(&clean_name("duct_R:2")), BodyRole::Duct);
    }

    #[test]
    fn loads_step_assembly() {
        if !peregrine_occt::AVAILABLE { return; }
        let f = std::env::temp_dir().join("peregrine_geom_test.step");
        peregrine_occt::write_demo_step(&f).unwrap();
        let bodies = load_bodies(&f, &Project::new("t")).unwrap();
        let duct = bodies.iter().find(|b| b.name == "duct").unwrap();
        assert_eq!(duct.role, BodyRole::Duct);
        assert_eq!(duct_port_candidates(&bodies, 1.0, None).len(), 2);
    }
}
