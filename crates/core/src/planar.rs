//! Planar face detection from a triangle mesh and extraction of holes (port candidates).
use crate::mesh::TriMesh;
use crate::port_cap::PortCap;
use crate::vec3::*;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct PlanarRegion {
    pub normal: V3,
    pub origin: V3,
    pub triangles: Vec<usize>,
    pub area: f64,
    pub outer: Vec<V3>,
    pub holes: Vec<Vec<V3>>,
}

#[derive(Clone, Debug)]
pub struct PortCandidate {
    pub cap: PortCap,
    pub area: f64,
    /// Area of the planar face that contains the hole.
    pub parent_area: f64,
}

/// Group coplanar connected triangles and extract their boundary loops.
/// The mesh must be welded. `angle_tol_deg`/`dist_tol` define coplanarity against the seed triangle.
pub fn planar_regions(mesh: &TriMesh, angle_tol_deg: f64, dist_tol: f64) -> Vec<PlanarRegion> {
    let nt = mesh.triangles.len();
    let cos_tol = angle_tol_deg.to_radians().cos();
    let mut normals = vec![[0.0; 3]; nt];
    let mut areas = vec![0.0; nt];
    for i in 0..nt {
        let [a, b, c] = mesh.tri(i);
        let n = cross(sub(b, a), sub(c, a));
        areas[i] = len(n) * 0.5;
        normals[i] = normalize(n);
    }
    // undirected edge -> triangles
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let mut region_of = vec![usize::MAX; nt];
    let mut regions = Vec::new();
    for seed in 0..nt {
        if region_of[seed] != usize::MAX || areas[seed] < 1e-12 {
            continue;
        }
        let rid = regions.len();
        let n0 = normals[seed];
        let o0 = mesh.tri(seed)[0];
        let mut tris = vec![];
        let mut stack = vec![seed];
        region_of[seed] = rid;
        while let Some(t) = stack.pop() {
            tris.push(t);
            let tv = mesh.triangles[t];
            for k in 0..3 {
                let (a, b) = (tv[k], tv[(k + 1) % 3]);
                for &t2 in &edges[&(a.min(b), a.max(b))] {
                    if region_of[t2] != usize::MAX || areas[t2] < 1e-12 {
                        continue;
                    }
                    if dot(normals[t2], n0) < cos_tol {
                        continue;
                    }
                    let flat = mesh.tri(t2).iter().all(|p| dot(sub(*p, o0), n0).abs() <= dist_tol);
                    if flat {
                        region_of[t2] = rid;
                        stack.push(t2);
                    }
                }
            }
        }
        let area: f64 = tris.iter().map(|&t| areas[t]).sum();
        // directed boundary edges (edge used by exactly one triangle of the region)
        let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
        for &t in &tris {
            let tv = mesh.triangles[t];
            for k in 0..3 {
                let (a, b) = (tv[k], tv[(k + 1) % 3]);
                let cnt = edges[&(a.min(b), a.max(b))].iter().filter(|&&x| region_of[x] == rid).count();
                if cnt == 1 {
                    next.entry(a).or_default().push(b);
                }
            }
        }
        let mut loops: Vec<Vec<u32>> = vec![];
        let starts: Vec<u32> = next.keys().copied().collect();
        for s in starts {
            while next.get(&s).map_or(false, |v| !v.is_empty()) {
                let mut lp = vec![s];
                let mut cur = s;
                loop {
                    let nx = match next.get_mut(&cur).and_then(|v| v.pop()) {
                        Some(n) => n,
                        None => break,
                    };
                    if nx == s {
                        break;
                    }
                    lp.push(nx);
                    cur = nx;
                }
                if lp.len() >= 3 {
                    loops.push(lp);
                }
            }
        }
        let mut outer = vec![];
        let mut holes = vec![];
        for lp in loops {
            let pts: Vec<V3> = lp.iter().map(|&i| mesh.vertices[i as usize]).collect();
            let mut a2 = [0.0; 3];
            for i in 0..pts.len() {
                a2 = add(a2, cross(pts[i], pts[(i + 1) % pts.len()]));
            }
            let signed = dot(a2, n0) * 0.5;
            if signed > 0.0 {
                if outer.is_empty() || pts.len() > outer.len() {
                    outer = pts;
                }
            } else {
                holes.push(pts);
            }
        }
        regions.push(PlanarRegion { normal: n0, origin: o0, triangles: tris, area, outer, holes });
    }
    regions
}

/// All planar-face holes with area >= `min_area` (mm^2), largest first.
pub fn find_port_candidates(mesh: &TriMesh, min_area: f64) -> Vec<PortCandidate> {
    let welded = mesh.welded(1e-5);
    let mut out = vec![];
    for r in planar_regions(&welded, 1.0, 1e-3) {
        for h in &r.holes {
            let mut c = [0.0; 3];
            for p in h {
                c = add(c, *p);
            }
            let c = scale(c, 1.0 / h.len() as f64);
            let cap = PortCap { origin: c, normal: r.normal, polygon: h.clone() };
            let area = cap.area();
            if area >= min_area {
                out.push(PortCandidate { cap, area, parent_area: r.area });
            }
        }
    }
    out.sort_by(|a, b| b.area.partial_cmp(&a.area).unwrap());
    out
}
