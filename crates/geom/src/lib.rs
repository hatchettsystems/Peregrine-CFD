//! Loading bodies from geometry files and assembling the voxel domain for a project.
use anyhow::{bail, Context, Result};
use peregrine_core::{find_port_candidates, stl, Aabb, PortCandidate, TriMesh};
use peregrine_project::{BodyRole, DuctDef, Project};
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
                bodies.push(Body { name, role: BodyRole::Solid, mesh });
            }
            "step" | "stp" => {
                let defl = (project.solver.cell_size_mm * 0.2).clamp(0.01, 0.1);
                for b in peregrine_occt::read_step(&f, defl).with_context(|| format!("reading {}", f.display()))? {
                    let name = clean_name(&b.name);
                    bodies.push(Body { name, role: BodyRole::Solid, mesh: b.mesh });
                }
            }
            _ => bail!("unsupported geometry file: {}", f.display()),
        }
    }
    apply_roles(project, &mut bodies, true);
    Ok(bodies)
}

/// Set each body's role from the project: parts of a defined duct are `Duct`, the ignore list is `Ignore`,
/// everything else is a solid obstacle. With `legacy_hints`, a project that never defined ducts falls back to
/// treating bodies whose name looks like a duct as ducts (command line and old project files).
pub fn apply_roles(project: &Project, bodies: &mut [Body], legacy_hints: bool) {
    let names: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
    let ducts = if legacy_hints { project.effective_ducts(&names) } else { project.ducts.clone() };
    for b in bodies.iter_mut() {
        b.role = if project.geometry.ignored.contains(&b.name) || project.geometry.overrides.get(&b.name) == Some(&BodyRole::Ignore) {
            BodyRole::Ignore
        } else if ducts.iter().any(|d| d.parts.contains(&b.name)) {
            BodyRole::Duct
        } else {
            BodyRole::Solid
        };
    }
}

/// Port candidates (planar holes) of every duct, as (duct name, candidate). A duct may be made of several
/// parts: openings are found on all parts together, and the openings where two parts join (equal and
/// opposite, coincident) are removed because they are internal.
/// `hide_round_below`: skip small round holes (screw holes) with an equivalent diameter up to this (mm).
pub fn duct_port_candidates(bodies: &[Body], ducts: &[DuctDef], min_area: f64, hide_round_below: Option<f64>, curved: bool) -> Vec<(String, PortCandidate)> {
    use peregrine_core::vec3::{dot, len, sub};
    let mut out = vec![];
    for d in ducts {
        let mut mesh = TriMesh::default();
        for part in &d.parts {
            if let Some(b) = bodies.iter().find(|b| &b.name == part) {
                mesh.append(&b.mesh);
            }
        }
        if mesh.triangles.is_empty() {
            continue;
        }
        let cands = find_port_candidates(&mesh, min_area);
        let mut keep = vec![true; cands.len()];
        for i in 0..cands.len() {
            for j in (i + 1)..cands.len() {
                let (a, b) = (&cands[i], &cands[j]);
                let joint = dot(a.cap.normal, b.cap.normal) < -0.99 && len(sub(a.cap.centroid(), b.cap.centroid())) < 0.1 && (a.area - b.area).abs() <= 0.02 * a.area.max(b.area);
                if joint {
                    keep[i] = false;
                    keep[j] = false;
                }
            }
        }
        let mut planar_kept: Vec<PortCandidate> = vec![];
        for (c, k) in cands.into_iter().zip(keep) {
            if k && !hide_round_below.map_or(false, |dm| c.is_screw_hole(dm)) {
                planar_kept.push(c);
            }
        }
        if curved {
            for o in peregrine_core::find_curved_openings(&mesh, 12.0, 300.0, 25.0) {
                let area = o.cap.area();
                if area < min_area.max(8.0) { continue; }
                let circ = (4.0 * std::f64::consts::PI * area / (o.perimeter * o.perimeter)).min(1.0);
                let cand = PortCandidate { area, parent_area: 0.0, circularity: circ, equiv_diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), cap: o.cap };
                if hide_round_below.map_or(false, |dm| cand.is_screw_hole(dm)) { continue; }
                planar_kept.push(cand);
            }
        }
        for c in planar_kept {
            out.push((d.name.clone(), c));
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
        let names: Vec<String> = bodies.iter().map(|b| b.name.clone()).collect();
        let ducts = Project::new("t").effective_ducts(&names);
        assert_eq!(duct_port_candidates(&bodies, &ducts, 1.0, None, false).len(), 2);
    }

    #[test]
    fn multi_part_duct_cancels_the_internal_joint() {
        use peregrine_core::shapes;
        let mut a = shapes::hollow_tube(8.0, 4.0, 15.0);
        let mut b = shapes::hollow_tube(8.0, 4.0, 15.0);
        b.translate([15.0, 0.0, 0.0]);
        a.translate([0.0, 0.0, 0.0]);
        let bodies = vec![
            Body { name: "part A".into(), role: BodyRole::Solid, mesh: a },
            Body { name: "part B".into(), role: BodyRole::Solid, mesh: b },
        ];
        // as two separate ducts each part shows both of its ends
        let separate = vec![
            DuctDef { name: "A".into(), parts: vec!["part A".into()] },
            DuctDef { name: "B".into(), parts: vec!["part B".into()] },
        ];
        assert_eq!(duct_port_candidates(&bodies, &separate, 1.0, None, false).len(), 4);
        // as one two-part duct the joint disappears and only the two real ends remain
        let joined = vec![DuctDef { name: "duct".into(), parts: vec!["part A".into(), "part B".into()] }];
        let c = duct_port_candidates(&bodies, &joined, 1.0, None, false);
        assert_eq!(c.len(), 2, "{:?}", c.iter().map(|x| x.1.cap.centroid()).collect::<Vec<_>>());
        let xs: Vec<f64> = c.iter().map(|x| x.1.cap.centroid()[0]).collect();
        assert!(xs.iter().any(|x| x.abs() < 1e-3) && xs.iter().any(|x| (x - 30.0).abs() < 1e-3), "{xs:?}");
    }

    #[test]
    fn roles_follow_user_ducts() {
        use peregrine_core::shapes;
        let mut p = Project::new("t");
        let mut bodies = vec![
            Body { name: "duct".into(), role: BodyRole::Solid, mesh: shapes::hollow_tube(8.0, 4.0, 10.0) },
            Body { name: "Part1".into(), role: BodyRole::Solid, mesh: shapes::solid_box([0.0; 3], [1.0; 3]) },
        ];
        // explicit mode: a duct-looking name alone does nothing
        apply_roles(&p, &mut bodies, false);
        assert!(bodies.iter().all(|b| b.role == BodyRole::Solid));
        p.define_duct("mine", &["Part1".to_string()]);
        apply_roles(&p, &mut bodies, false);
        assert_eq!(bodies[1].role, BodyRole::Duct);
        assert_eq!(bodies[0].role, BodyRole::Solid);
    }
}
