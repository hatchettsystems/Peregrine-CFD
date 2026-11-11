//! Non-planar openings: find the closed edge loops that bound a smooth surface patch the user clicked
//! (a rim lip, an inner wall, ...), and span a loop with a triangulated cap surface.
use crate::mesh::TriMesh;
use crate::port_cap::PortCap;
use crate::vec3::*;
use std::collections::HashMap;

/// A closed boundary loop of a smooth surface patch.
#[derive(Clone, Debug)]
pub struct RimLoop {
    pub points: Vec<V3>,
    pub perimeter: f64,
    /// Largest distance (mm) of any loop point from the loop's best-fit plane. ~0 for planar loops.
    pub planarity_dev: f64,
}

/// Newell normal (robust for non-planar polygons) and centroid of a loop.
pub fn loop_frame(pts: &[V3]) -> (V3, V3) {
    let mut c = [0.0; 3];
    for p in pts {
        c = add(c, *p);
    }
    let c = scale(c, 1.0 / pts.len().max(1) as f64);
    let mut n = [0.0; 3];
    for i in 0..pts.len() {
        let (a, b) = (sub(pts[i], c), sub(pts[(i + 1) % pts.len()], c));
        n = add(n, cross(a, b));
    }
    (c, normalize(n))
}

/// Closed boundary loops of the smooth patch around triangle `seed`. Neighbouring triangles belong to the
/// same patch when the angle between their normals is below `feature_angle_deg`. The mesh must be welded.
pub fn smooth_region_loops(mesh: &TriMesh, seed: usize, feature_angle_deg: f64) -> Vec<RimLoop> {
    let nt = mesh.triangles.len();
    if seed >= nt {
        return vec![];
    }
    let cos_tol = feature_angle_deg.to_radians().cos();
    let normals: Vec<V3> = (0..nt)
        .map(|i| {
            let [a, b, c] = mesh.tri(i);
            normalize(cross(sub(b, a), sub(c, a)))
        })
        .collect();
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let mut inside = vec![false; nt];
    let mut stack = vec![seed];
    inside[seed] = true;
    let mut region = vec![];
    while let Some(t) = stack.pop() {
        region.push(t);
        let tv = mesh.triangles[t];
        for k in 0..3 {
            let (a, b) = (tv[k], tv[(k + 1) % 3]);
            let nb = &edges[&(a.min(b), a.max(b))];
            if nb.len() != 2 {
                continue; // boundary or non-manifold edge: treated as a crease
            }
            for &t2 in nb {
                if !inside[t2] && dot(normals[t], normals[t2]) >= cos_tol {
                    inside[t2] = true;
                    stack.push(t2);
                }
            }
        }
    }
    let mut next: HashMap<u32, Vec<u32>> = HashMap::new();
    for &t in &region {
        let tv = mesh.triangles[t];
        for k in 0..3 {
            let (a, b) = (tv[k], tv[(k + 1) % 3]);
            let nb = &edges[&(a.min(b), a.max(b))];
            let other_inside = nb.iter().any(|&x| x != t && inside[x]);
            if nb.len() != 2 || !other_inside {
                next.entry(a).or_default().push(b);
            }
        }
    }
    let mut loops = vec![];
    let starts: Vec<u32> = next.keys().copied().collect();
    for s in starts {
        while next.get(&s).map_or(false, |v| !v.is_empty()) {
            let mut lp = vec![s];
            let mut cur = s;
            let mut closed = false;
            loop {
                let Some(nx) = next.get_mut(&cur).and_then(|v| v.pop()) else { break };
                if nx == s {
                    closed = true;
                    break;
                }
                lp.push(nx);
                cur = nx;
            }
            if closed && lp.len() >= 3 {
                let points: Vec<V3> = lp.iter().map(|&i| mesh.vertices[i as usize]).collect();
                let perimeter: f64 = (0..points.len()).map(|i| len(sub(points[(i + 1) % points.len()], points[i]))).sum();
                let (c, n) = loop_frame(&points);
                let planarity_dev = points.iter().map(|p| dot(sub(*p, c), n).abs()).fold(0.0, f64::max);
                loops.push(RimLoop { points, perimeter, planarity_dev });
            }
        }
    }
    loops.sort_by(|a, b| b.perimeter.partial_cmp(&a.perimeter).unwrap());
    loops
}

/// Triangulate a closed 3D loop into a surface using its own vertices: ear clipping in the loop's
/// best-fit plane, falling back to a fan from the centroid when the projection folds over.
pub fn triangulate_loop(pts: &[V3]) -> Vec<[V3; 3]> {
    let n = pts.len();
    if n < 3 {
        return vec![];
    }
    let (c, nrm) = loop_frame(pts);
    let (u, v) = basis(nrm);
    let p2: Vec<[f64; 2]> = pts.iter().map(|p| { let d = sub(*p, c); [dot(d, u), dot(d, v)] }).collect();
    // Newell normal means the projected loop is counter-clockwise in (u, v)
    let cross2 = |a: [f64; 2], b: [f64; 2], c: [f64; 2]| (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    let mut idx: Vec<usize> = (0..n).collect();
    let mut tris: Vec<[usize; 3]> = vec![];
    let mut guard = 0;
    while idx.len() > 3 && guard < 4 * n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for k in 0..m {
            let (ia, ib, ic) = (idx[(k + m - 1) % m], idx[k], idx[(k + 1) % m]);
            if cross2(p2[ia], p2[ib], p2[ic]) <= 1e-12 {
                continue; // reflex or degenerate
            }
            let inside_any = idx.iter().any(|&j| {
                if j == ia || j == ib || j == ic { return false; }
                let (d1, d2, d3) = (cross2(p2[ia], p2[ib], p2[j]), cross2(p2[ib], p2[ic], p2[j]), cross2(p2[ic], p2[ia], p2[j]));
                d1 >= -1e-12 && d2 >= -1e-12 && d3 >= -1e-12
            });
            if !inside_any {
                tris.push([ia, ib, ic]);
                idx.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            break;
        }
    }
    if idx.len() == 3 {
        tris.push([idx[0], idx[1], idx[2]]);
    }
    if tris.len() == n - 2 {
        return tris.iter().map(|t| [pts[t[0]], pts[t[1]], pts[t[2]]]).collect();
    }
    // fan from the centroid
    (0..n).map(|i| [c, pts[i], pts[(i + 1) % n]]).collect()
}

fn ray_hits(mesh: &TriMesh, o: V3, d: V3, max_t: f64) -> bool {
    for i in 0..mesh.triangles.len() {
        let [a, b, c] = mesh.tri(i);
        let (e1, e2) = (sub(b, a), sub(c, a));
        let p = cross(d, e2);
        let det = dot(e1, p);
        if det.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / det;
        let t0 = sub(o, a);
        let u = dot(t0, p) * inv;
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = cross(t0, e1);
        let v = dot(d, q) * inv;
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        let t = dot(e2, q) * inv;
        if t > 1e-6 && t < max_t {
            return true;
        }
    }
    false
}

/// How enclosed a point is: number of 26 directions that hit the mesh within 60 mm.
fn enclosure(mesh: &TriMesh, p: V3) -> usize {
    let mut n = 0;
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dy == 0 && dz == 0 { continue; }
                let d = normalize([dx as f64, dy as f64, dz as f64]);
                if ray_hits(mesh, p, d, 60.0) { n += 1; }
            }
        }
    }
    n
}

/// Build a cap surface for the loop. `duct` is the duct's mesh (all parts); it decides which side of the cap
/// is the inside of the duct so that the normal points out of it.
pub fn cap_from_loop(points: &[V3], duct: &TriMesh) -> Option<PortCap> {
    let tris = triangulate_loop(points);
    if tris.is_empty() {
        return None;
    }
    let mut a_tot = 0.0;
    let mut n0 = [0.0; 3];
    let mut ctr = [0.0; 3];
    for t in &tris {
        let cr = cross(sub(t[1], t[0]), sub(t[2], t[0]));
        let a = 0.5 * len(cr);
        a_tot += a;
        n0 = add(n0, scale(cr, 0.5));
        ctr = add(ctr, scale(scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0), a));
    }
    if a_tot < 1e-9 {
        return None;
    }
    let ctr = scale(ctr, 1.0 / a_tot);
    let n0 = normalize(n0);
    // which side is enclosed by the duct?
    let (mut plus, mut minus) = (0usize, 0usize);
    for dist in [1.5, 3.0, 5.0] {
        plus += enclosure(duct, add(ctr, scale(n0, dist)));
        minus += enclosure(duct, sub(ctr, scale(n0, dist)));
    }
    // lumen on the minus side => n0 already points out of the duct
    let n_out = if plus > minus { scale(n0, -1.0) } else { n0 };
    let flip = dot(n0, n_out) < 0.0;
    let triangles: Vec<[V3; 3]> = tris.into_iter().map(|t| if flip { [t[0], t[2], t[1]] } else { t }).collect();
    Some(PortCap { origin: ctr, normal: n_out, polygon: points.to_vec(), triangles })
}
