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

/// Load bodies. A `.stl` file is one body named after the file; a directory loads every `.stl` in it.
/// STEP import (`.step`/`.stp`) needs the OCCT backend and is not built yet.
pub fn load_bodies(path: &Path, project: &Project) -> Result<Vec<Body>> {
    let mut files = vec![];
    if path.is_dir() {
        for e in std::fs::read_dir(path)? {
            let p = e?.path();
            if p.extension().map_or(false, |x| x.eq_ignore_ascii_case("stl")) {
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
            "step" | "stp" => bail!("STEP import is not available in this build yet (OCCT backend is milestone 2). Export bodies as STL for now."),
            _ => bail!("unsupported geometry file: {}", f.display()),
        }
    }
    Ok(bodies)
}

/// Port candidates (planar holes) of every Duct body, as (body name, candidate).
pub fn duct_port_candidates(bodies: &[Body], min_area: f64) -> Vec<(String, PortCandidate)> {
    let mut out = vec![];
    for b in bodies.iter().filter(|b| b.role == BodyRole::Duct) {
        for c in find_port_candidates(&b.mesh, min_area) {
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
    let bounds = domain_bounds(bodies, project.domain.padding_mm);
    let d = build_domain(&meshes, &ports, &bounds, project.solver.cell_size_mm);
    for (p, n) in project.ports.iter().zip(&d.port_cell_counts) {
        if *n == 0 {
            bail!("port '{}' covers no cells at {} mm; refine the grid or check the cap", p.name, project.solver.cell_size_mm);
        }
    }
    Ok(d)
}
