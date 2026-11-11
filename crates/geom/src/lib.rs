//! Loading bodies from geometry files and assembling the voxel domain for a project.
use anyhow::{bail, Context, Result};
use peregrine_core::{find_port_candidates, stl, Aabb, PortCandidate, TriMesh};
use peregrine_project::{BodyRole, DuctDef, Project};
use peregrine_voxel::{behind_volumes, build_domain_full, leak_check, Domain, PortSpec, PORT_BASE};
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
                let defl = (project.solver.cell_size_mm * 0.12).clamp(0.02, 0.06);
                for b in peregrine_occt::read_step(&f, defl).with_context(|| format!("reading {}", f.display()))? {
                    let name = clean_name(&b.name);
                    bodies.push(Body { name, role: BodyRole::Solid, mesh: b.mesh });
                }
            }
            _ => bail!("unsupported geometry file: {}", f.display()),
        }
    }
    for b in &mut bodies {
        b.mesh.orient_outward();
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
            let prm = peregrine_core::CurvedParams { include_flat: true, ..Default::default() };
            for o in peregrine_core::find_curved_openings_with(&mesh, &prm) {
                let area = o.cap.area();
                if area < min_area.max(8.0) { continue; }
                // flat loops the flat-hole detector already reported are not new
                let dup = planar_kept.iter().any(|c| !c.cap.is_surface() && !o.cap.is_surface()
                    && dot(c.cap.normal, o.cap.normal).abs() > 0.99
                    && len(sub(c.cap.centroid(), o.cap.centroid())) < 0.5
                    && (c.area - area).abs() <= 0.06 * c.area.max(area));
                if dup { continue; }
                let circ = (4.0 * std::f64::consts::PI * area / (o.perimeter * o.perimeter)).min(1.0);
                let cand = PortCandidate { area, parent_area: 0.0, circularity: circ, equiv_diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), visibility: 1.0, clearance: 1.0, inline: false, cap: o.cap };
                if hide_round_below.map_or(false, |dm| cand.is_screw_hole(dm)) { continue; }
                planar_kept.push(cand);
            }
        }
        if curved {
            // second opinion: mouths found from surface exposure (any shape; no flat face or sharp edge needed)
            for o in peregrine_core::find_exposure_openings(&mesh, 25.0, 6.0, 400.0) {
                let area = o.cap.area();
                if area < min_area.max(3.0) { continue; }
                let dup = planar_kept.iter().any(|c| {
                    let near = len(sub(c.cap.centroid(), o.cap.centroid())) < 1.5 + 0.15 * (c.area.max(area)).sqrt();
                    near && (c.area - area).abs() <= 0.35 * c.area.max(area)
                });
                if dup { continue; }
                let circ = (4.0 * std::f64::consts::PI * area / (o.perimeter * o.perimeter)).min(1.0);
                let cand = PortCandidate { area, parent_area: 0.0, circularity: circ, equiv_diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), visibility: 1.0, clearance: 1.0, inline: false, cap: o.cap };
                if hide_round_below.map_or(false, |dm| cand.is_screw_hole(dm)) { continue; }
                planar_kept.push(cand);
            }
        }
        // A loop lying inside another coplanar, same-facing loop is an island of the larger opening (a hub or
        // boss): subtract it, so the opening is the area BETWEEN the loops.
        nest_islands(&mut planar_kept);
        // an opening that admits air must be visible from outside the duct
        let grid = peregrine_core::RayGrid::new(&mesh);
        let mut duct_cands: Vec<PortCandidate> = vec![];
        for mut c in planar_kept {
            c.visibility = peregrine_core::external_visibility(&c.cap, &grid);
            duct_cands.push(c);
        }
        // Air behind a real opening is the whole duct interior; behind a pocket it is a small closed cavity.
        // Score each opening by the connected volume behind it relative to the largest.
        let part_meshes: Vec<&TriMesh> = d.parts.iter().filter_map(|p| bodies.iter().find(|b| &b.name == p).map(|b| &b.mesh)).collect();
        if !duct_cands.is_empty() && !part_meshes.is_empty() {
            let mut bb = mesh.bounds().expanded(2.0);
            if !bb.is_valid() { bb = mesh.bounds(); }
            let size = [bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]];
            let dx = (size[0] * size[1] * size[2] / 6.0e6).cbrt().clamp(0.4, 1.2);
            // plug islands too: air must not escape through a hub or boss region
            let caps: Vec<peregrine_core::PortCap> = duct_cands.iter().map(|c| { let mut k = c.cap.clone(); k.holes.clear(); k }).collect();
            // A thin neck between a pocket-like slot and the channel can be closed by one grid and open in another:
            // probe three slightly different grids and take the best connection for each opening.
            let mut vols = behind_volumes(&part_meshes, &caps, &bb, dx);
            for f in [0.78, 1.3] {
                let alt = behind_volumes(&part_meshes, &caps, &bb, dx * f);
                for (v, a) in vols.iter_mut().zip(alt) {
                    if a.is_finite() && (!v.is_finite() || a > *v) { *v = a; }
                }
            }
            // Openings into the same channel system report the same volume. Normalise by the largest volume that at
            // least two openings share (the duct interior), so a single cap that sits on open air (a huge volume of
            // its own) cannot make every real opening look like a pocket.
            let mut finite: Vec<f64> = vols.iter().cloned().filter(|v| v.is_finite()).collect();
            finite.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let shared = finite.iter().cloned().find(|v| finite.iter().filter(|w| (**w - *v).abs() <= 0.01 * v.abs().max(1e-9)).count() >= 2);
            let vmax = shared.unwrap_or_else(|| finite.first().cloned().unwrap_or(1.0)).max(1e-9);
            if std::env::var_os("PG_DEBUG").is_some() { eprintln!("behind volumes (dx {dx:.2}): max {:.0}, shared {:?}, all {:?}", finite.first().cloned().unwrap_or(0.0), shared, vols.iter().map(|v| v.round()).collect::<Vec<_>>()); }
            for (c, v) in duct_cands.iter_mut().zip(vols) {
                c.clearance = if v.is_finite() { (v / vmax).min(1.0) } else { 1.0 };
            }
        }
        // An opening whose outward direction runs on, through air, to another opening is only a cross-section of the
        // tube: the opening farther down the tube is the port.
        let n_c = duct_cands.len();
        let mut inline = vec![false; n_c];
        for i in 0..n_c {
            let a_cap = &duct_cands[i].cap;
            let dir = a_cap.normal;
            let samples = peregrine_core::rim::cap_samples(a_cap);
            if samples.is_empty() { continue; }
            let step = (samples.len() / 24).max(1);
            let (mut tried, mut reached) = (0usize, 0usize);
            for p in samples.iter().step_by(step) {
                tried += 1;
                let o = [p[0] + dir[0] * 0.05, p[1] + dir[1] * 0.05, p[2] + dir[2] * 0.05];
                let hit = (0..n_c).filter(|&j| j != i && dot(dir, duct_cands[j].cap.normal) >= 0.8).any(|j| {
                    ray_hits_cap(o, dir, &duct_cands[j].cap).map_or(false, |t| t > 0.5 && t < 80.0 && !grid.any_hit(o, dir, t - 0.2))
                });
                if hit { reached += 1; }
            }
            // most of the air leaving this opening runs on to another one
            if tried > 0 && reached * 2 >= tried { inline[i] = true; }
        }
        for (c, f) in duct_cands.iter_mut().zip(inline) { c.inline = f; }
        for c in duct_cands {
            out.push((d.name.clone(), c));
        }
    }
    out
}

/// Distance along the ray `o + t d` to where it crosses an opening (planar or curved), if it does.
fn ray_hits_cap(o: [f64; 3], d: [f64; 3], cap: &peregrine_core::PortCap) -> Option<f64> {
    use peregrine_core::vec3::{cross, dot, sub};
    if cap.is_surface() {
        let mut best: Option<f64> = None;
        for t in &cap.triangles {
            let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
            let p = cross(d, e2);
            let det = dot(e1, p);
            if det.abs() < 1e-12 { continue; }
            let inv = 1.0 / det;
            let s = sub(o, t[0]);
            let u = dot(s, p) * inv;
            if !(0.0..=1.0).contains(&u) { continue; }
            let q = cross(s, e1);
            let v = dot(d, q) * inv;
            if v < 0.0 || u + v > 1.0 { continue; }
            let tt = dot(e2, q) * inv;
            if tt > 1e-6 && best.map_or(true, |b| tt < b) { best = Some(tt); }
        }
        return best;
    }
    let denom = dot(d, cap.normal);
    if denom.abs() < 1e-9 { return None; }
    let t = dot(sub(cap.origin, o), cap.normal) / denom;
    if t <= 0.0 { return None; }
    let hit = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
    let poly = cap.polygon_2d();
    let holes = cap.holes_2d();
    cap.contains_in_opening(&poly, &holes, hit).then_some(t)
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
    let ports: Vec<PortSpec> = project.ports.iter().map(|p| PortSpec { cap: p.cap.clone(), plug: p.kind.is_inlet(), collar_mm: project.domain.inlet_collar_mm }).collect();
    let bounds = project_bounds(project, bodies);
    // parts of one duct: seal small gaps between them
    let names: Vec<String> = active.iter().map(|b| b.name.clone()).collect();
    let groups: Vec<Vec<usize>> = project
        .effective_ducts(&bodies.iter().map(|b| b.name.clone()).collect::<Vec<_>>())
        .iter()
        .map(|d| d.parts.iter().filter_map(|p| names.iter().position(|n| n == p)).collect::<Vec<usize>>())
        .filter(|g| g.len() >= 2)
        .collect();
    let blockers: Vec<peregrine_core::PortCap> = project.blockers.iter().map(|b| b.cap.clone()).collect();
    let plugs: Vec<([f64; 3], f64)> = project.plugs.iter().map(|p| (p.center, p.radius_mm)).collect();
    let mut d = build_domain_full(&meshes, &groups, project.domain.seam_tolerance_mm, project.domain.gap_seal_mm, &ports, &blockers, &plugs, &bounds, project.solver.cell_size_mm);
    let inlets: Vec<u8> = project.ports.iter().enumerate().filter(|(_, p)| p.kind.is_inlet()).map(|(k, _)| PORT_BASE + k as u8).collect();
    if project.domain.auto_seal_leaks {
        d.repaired = peregrine_voxel::auto_seal_leaks(&mut d, &inlets, 150, 2.6);
    }
    peregrine_voxel::keep_relevant_seals(&mut d, &inlets);
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
    fn an_opening_with_another_down_the_same_tube_is_inline() {
        use peregrine_core::shapes;
        // two tubes end to end with a 3 mm gap: the gap faces are cross-sections of one longer tube
        let a = shapes::hollow_tube(8.0, 4.0, 15.0);
        let mut b = shapes::hollow_tube(8.0, 4.0, 15.0);
        b.translate([18.0, 0.0, 0.0]);
        let bodies = vec![
            Body { name: "part A".into(), role: BodyRole::Solid, mesh: a },
            Body { name: "part B".into(), role: BodyRole::Solid, mesh: b },
        ];
        let duct = vec![DuctDef { name: "duct".into(), parts: vec!["part A".into(), "part B".into()] }];
        let c = duct_port_candidates(&bodies, &duct, 1.0, None, false);
        assert_eq!(c.len(), 4, "{:?}", c.iter().map(|x| (x.1.cap.centroid()[0], x.1.cap.normal[0])).collect::<Vec<_>>());
        for (_, cand) in &c {
            let x = cand.cap.centroid()[0];
            let facing_out = cand.cap.normal[0] > 0.0;
            // A's far end (x=15, facing +x) leads on through part B to its far end; B's near end (x=18, facing -x)
            // leads back through A to its start. The two true ends are not inline.
            let expect_inline = (facing_out && (x - 15.0).abs() < 1e-3) || (!facing_out && (x - 18.0).abs() < 1e-3);
            assert_eq!(cand.inline, expect_inline, "opening at x = {x}, normal {:?}", cand.cap.normal);
        }
        assert_eq!(c.iter().filter(|x| is_real_opening(&x.1)).count(), 2);
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
    fn inner_loop_becomes_an_island_of_the_outer_opening() {
        use peregrine_core::shapes;
        // an 8x8 opening with a smaller coplanar loop (a hub bore) inside it
        let bodies = vec![
            Body { name: "shell".into(), role: BodyRole::Solid, mesh: shapes::hollow_tube(12.0, 8.0, 10.0) },
            Body { name: "hub".into(), role: BodyRole::Solid, mesh: shapes::hollow_tube(5.0, 3.0, 10.0) },
        ];
        let ducts = vec![DuctDef { name: "d".into(), parts: vec!["shell".into(), "hub".into()] }];
        let c = duct_port_candidates(&bodies, &ducts, 1.0, None, false);
        // find the outer opening at x = 0 (normal -x) with the larger area
        let outer = c.iter().filter(|x| x.1.cap.normal[0] < -0.99).max_by(|a, b| (a.1.cap.polygon.len(), a.1.area).partial_cmp(&(b.1.cap.polygon.len(), b.1.area)).unwrap()).unwrap();
        let big = c.iter().filter(|x| x.1.cap.normal[0] < -0.99 && !x.1.cap.holes.is_empty()).collect::<Vec<_>>();
        assert_eq!(big.len(), 1, "{:?}", c.iter().map(|x| (x.1.area, x.1.cap.holes.len())).collect::<Vec<_>>());
        let a = &big[0].1;
        assert!((a.area - (64.0 - 9.0)).abs() < 0.01, "outer opening should be 64 - 9 = 55 mm2, got {}", a.area);
        assert_eq!(a.cap.holes.len(), 1);
        let _ = outer;
        // the inner loop stays available as its own opening
        assert!(c.iter().any(|x| (x.1.area - 9.0).abs() < 0.01));
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


/// A warning when air can leave the duct through something that is not a port (a gap between parts wider than the
/// seam tolerance, or an opening that was not defined as a port).
pub fn leak_warning(project: &Project, domain: &Domain) -> Option<String> {
    let inlets: Vec<u8> = project.ports.iter().enumerate().filter(|(_, p)| p.kind.is_inlet()).map(|(k, _)| PORT_BASE + k as u8).collect();
    let leak = leak_check(domain, &inlets)?;
    let p = leak.throat;
    Some(format!(
        "Air can leave the duct through something that is not a port, near ({:.1}, {:.1}, {:.1}) mm (gap about {:.1} mm wide): a gap between parts wider than {:.1} mm, or an opening you did not define as a port. Seal it, block it, or define it as a port.",
        p[0], p[1], p[2], leak.throat_width, project.domain.seam_tolerance_mm
    ))
}

/// What the automatic sealer did, if anything (shown as an informational note with a marker).
pub fn repair_note(domain: &Domain) -> Option<String> {
    let regions = domain.sealed_regions(25);
    let first = regions.first()?;
    let cells: usize = regions.iter().map(|r| r.cells).sum();
    let vol: f64 = regions.iter().map(|r| r.volume_mm3).sum();
    let mut lines = String::new();
    for (k, r) in regions.iter().take(8).enumerate() {
        lines.push_str(&format!("\n  {}. near ({:.1}, {:.1}, {:.1}) mm, about {:.1} mm across, {:.1} mm³", k + 1, r.center[0], r.center[1], r.center[2], 2.0 * r.radius_mm, r.volume_mm3));
    }
    if regions.len() > 8 { lines.push_str(&format!("\n  ... and {} more regions", regions.len() - 8)); }
    let _ = first;
    Some(format!(
        "Auto-sealed small gaps (thin air gaps between solid surfaces) in {} region(s), {} cells, {:.0} mm³ in total. No gap that thin should exist in a real part, so they were closed.{}\nReal openings are never closed automatically: define them as ports or blockers. Tick 'Show auto-sealed regions' in the Results display options to see them, or turn the sealer off with gap_seal_mm = 0 / auto_seal_leaks = false in the project's [domain] section.",
        regions.len(), cells, vol, lines
    ))
}

/// The regions the automatic sealers closed (centre x/y/z in mm, cell count, radius mm), for display.
pub fn sealed_regions(domain: &Domain) -> Vec<[f64; 5]> {
    domain.sealed_regions(25).iter().map(|r| [r.center[0], r.center[1], r.center[2], r.cells as f64, r.radius_mm]).collect()
}

/// True unless the opening is a pocket: hard to see from outside AND with no confirmed air behind it.
pub fn is_real_opening(c: &PortCandidate) -> bool {
    !c.inline && (c.clearance >= 0.5 || c.visibility >= 0.5)
}

/// Make every planar opening that lies wholly inside another coplanar, same-facing opening an island of it.
fn nest_islands(cands: &mut [PortCandidate]) {
    use peregrine_core::port_cap::point_in_polygon;
    use peregrine_core::vec3::{dot, sub};
    for a in 0..cands.len() {
        if cands[a].cap.is_surface() {
            continue;
        }
        let poly_a = cands[a].cap.polygon_2d();
        let mut islands: Vec<Vec<peregrine_core::vec3::V3>> = vec![];
        for b in 0..cands.len() {
            if a == b || cands[b].cap.is_surface() || cands[b].cap.area() >= cands[a].cap.area() {
                continue;
            }
            let (ca, cb) = (&cands[a].cap, &cands[b].cap);
            let same_plane = dot(ca.normal, cb.normal) > 0.999 && dot(sub(cb.origin, ca.origin), ca.normal).abs() < 0.05;
            if !same_plane {
                continue;
            }
            let inside = cb.polygon.iter().all(|p| point_in_polygon(&poly_a, ca.project2d(*p)));
            if inside {
                islands.push(cb.polygon.clone());
            }
        }
        if !islands.is_empty() {
            cands[a].cap.holes = islands;
            let area = cands[a].cap.area();
            cands[a].area = area;
            cands[a].equiv_diameter = 2.0 * (area / std::f64::consts::PI).sqrt();
        }
    }
}