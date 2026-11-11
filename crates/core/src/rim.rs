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
    let mut starts: Vec<u32> = next.keys().copied().collect();
    starts.sort_unstable();
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

/// Build a cap surface for the loop. `duct` is the duct's mesh (all parts); it decides which side of the cap
/// is the inside of the duct so that the normal points out of it.
pub fn cap_from_loop(points: &[V3], duct: &TriMesh) -> Option<PortCap> {
    cap_from_loop_with_grid(points, &RayGrid::new(duct))
}

/// Same as [`cap_from_loop`] with a prebuilt [`RayGrid`] (build it once when testing many loops).
pub fn cap_from_loop_with_grid(points: &[V3], grid: &RayGrid) -> Option<PortCap> {
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
    // probe from the point deepest inside the cap (the area centroid can lie outside a bent slot)
    let cents: Vec<V3> = tris.iter().map(|t| scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0)).collect();
    let base = cents
        .iter()
        .max_by(|a, b| {
            let da = points.iter().map(|p| len(sub(*p, **a))).fold(f64::INFINITY, f64::min);
            let db = points.iter().map(|p| len(sub(*p, **b))).fold(f64::INFINITY, f64::min);
            da.partial_cmp(&db).unwrap()
        })
        .copied()
        .unwrap_or(ctr);
    let (mut plus, mut minus) = (0usize, 0usize);
    for dist in [1.5, 3.0, 5.0] {
        plus += grid.enclosure(add(base, scale(n0, dist)), 60.0);
        minus += grid.enclosure(sub(base, scale(n0, dist)), 60.0);
    }
    // lumen on the minus side => n0 already points out of the duct
    let n_out = if plus > minus { scale(n0, -1.0) } else { n0 };
    let flip = dot(n0, n_out) < 0.0;
    let triangles: Vec<[V3; 3]> = tris.into_iter().map(|t| if flip { [t[0], t[2], t[1]] } else { t }).collect();
    Some(PortCap { origin: ctr, normal: n_out, polygon: points.to_vec(), triangles, holes: vec![] })
}

/// Fraction (0..1) of the opening that can be seen from outside the duct: sample points over the cap are tested
/// with a few rays in a cone around the outward normal; a point counts as visible when at least one ray escapes
/// without touching the duct. Openings facing into the duct's own walls score near 0.
pub fn external_visibility(cap: &PortCap, grid: &RayGrid) -> f64 {
    let pts = cap_samples(cap);
    external_visibility_at(cap, grid, &pts)
}

/// Sample points spread over an opening (islands excluded).
pub fn cap_samples(cap: &PortCap) -> Vec<V3> {
    let mut pts: Vec<V3> = vec![];
    if cap.is_surface() {
        let areas: Vec<f64> = cap.triangles.iter().map(|t| 0.5 * len(cross(sub(t[1], t[0]), sub(t[2], t[0])))).collect();
        let total: f64 = areas.iter().sum();
        if total < 1e-12 {
            return vec![];
        }
        let mut cum = vec![];
        let mut acc = 0.0;
        for a in &areas { acc += a; cum.push(acc); }
        let frac = |x: f64| x - x.floor();
        for i in 0..120 {
            let u = frac(0.5 + i as f64 * 0.754_877_666_246_692_7) * total;
            let ti = cum.partition_point(|&c| c < u).min(cap.triangles.len() - 1);
            let (a, b) = (frac(0.5 + i as f64 * 0.618_033_988_749_894_9), frac(0.5 + i as f64 * 0.414_213_562_373_095));
            let (ra, rb) = (a.sqrt(), b);
            let t = cap.triangles[ti];
            pts.push(add(add(scale(t[0], 1.0 - ra), scale(t[1], ra * (1.0 - rb))), scale(t[2], ra * rb)));
        }
    } else {
        let poly = cap.polygon_2d();
        let holes = cap.holes_2d();
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for p in &poly {
            for k in 0..2 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); }
        }
        let (u, v) = cap.basis();
        let h = (cap.area() / 100.0).sqrt().max(0.3);
        let mut y = lo[1] + h * 0.5;
        while y < hi[1] {
            let mut x = lo[0] + h * 0.5;
            while x < hi[0] {
                if crate::port_cap::point_in_polygon(&poly, [x, y]) && !holes.iter().any(|hh| crate::port_cap::point_in_polygon(hh, [x, y])) {
                    pts.push(add(cap.origin, add(scale(u, x), scale(v, y))));
                }
                x += h;
            }
            y += h;
        }
    }
    pts
}

fn external_visibility_at(cap: &PortCap, grid: &RayGrid, pts: &[V3]) -> f64 {
    if pts.is_empty() {
        return 0.0;
    }
    let n = cap.normal;
    let (a, b) = basis(n);
    // a viewer may look from any direction in the hemisphere in front of the opening: sample it evenly
    // (Fibonacci spiral), because a notch or recess may only be visible through a narrow window of angles
    let dirs: Vec<V3> = {
        let count = 96;
        let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        (0..count)
            .map(|k| {
                let cz = 0.26 + 0.74 * (1.0 - (k as f64 + 0.5) / count as f64); // cos of the angle from the normal
                let r = (1.0 - cz * cz).max(0.0).sqrt();
                let phi = golden * k as f64;
                add(add(scale(a, r * phi.cos()), scale(b, r * phi.sin())), scale(n, cz))
            })
            .collect()
    };
    let visible = pts
        .iter()
        .filter(|p| {
            // start a hair in front of the cap so the cap's own rim does not block the ray
            let o = add(**p, scale(n, 0.15));
            dirs.iter().any(|d| !grid.any_hit(o, *d, 1.0e5))
        })
        .count();
    visible as f64 / pts.len() as f64
}

/// Fraction (0..1) of the opening with open air continuing at least `min_depth` mm behind it (into the duct).
/// A real opening leads into a channel; a recess or pocket in a face ends against a floor within a few mm.
pub fn inward_clearance(cap: &PortCap, grid: &RayGrid, min_depth: f64) -> f64 {
    let pts = cap_samples(cap);
    if pts.is_empty() {
        return 0.0;
    }
    let n = scale(cap.normal, -1.0); // into the duct
    let (a, b) = basis(n);
    let tilt = 20f64.to_radians();
    let mut dirs = vec![n];
    for k in 0..4 {
        let ang = std::f64::consts::TAU * k as f64 / 4.0;
        let side = add(scale(a, ang.cos()), scale(b, ang.sin()));
        dirs.push(normalize(add(scale(n, tilt.cos()), scale(side, tilt.sin()))));
    }
    let ok = pts
        .iter()
        .filter(|p| {
            let o = add(**p, scale(n, 0.2));
            dirs.iter().any(|d| !grid.any_hit(o, *d, min_depth))
        })
        .count();
    ok as f64 / pts.len() as f64
}

/// Uniform-grid acceleration for ray queries against a triangle mesh.
pub struct RayGrid {
    tris: Vec<[V3; 3]>,
    min: V3,
    cell: f64,
    dims: [usize; 3],
    cells: Vec<Vec<u32>>,
}

impl RayGrid {
    pub fn new(mesh: &TriMesh) -> RayGrid {
        let b = mesh.bounds().expanded(1e-6);
        let t = mesh.triangles.len().max(1) as f64;
        let size = sub(b.max, b.min);
        let vol = (size[0] * size[1] * size[2]).max(1e-9);
        // about 2 triangles per cell
        let cell = (vol / (t / 2.0)).cbrt().max(size.iter().cloned().fold(0.0, f64::max) / 128.0);
        let dims = [0, 1, 2].map(|i| ((size[i] / cell).ceil() as usize).max(1));
        let mut cells: Vec<Vec<u32>> = vec![vec![]; dims[0] * dims[1] * dims[2]];
        let tris: Vec<[V3; 3]> = (0..mesh.triangles.len()).map(|i| mesh.tri(i)).collect();
        for (ti, t) in tris.iter().enumerate() {
            let mut lo = t[0];
            let mut hi = t[0];
            for v in t {
                for k in 0..3 {
                    lo[k] = lo[k].min(v[k]);
                    hi[k] = hi[k].max(v[k]);
                }
            }
            let l = [0, 1, 2].map(|k| (((lo[k] - b.min[k]) / cell).floor().max(0.0)) as usize);
            let h = [0, 1, 2].map(|k| ((((hi[k] - b.min[k]) / cell).floor()).max(0.0) as usize).min(dims[k] - 1));
            for z in l[2]..=h[2] {
                for y in l[1]..=h[1] {
                    for x in l[0]..=h[0] {
                        cells[x + dims[0] * (y + dims[1] * z)].push(ti as u32);
                    }
                }
            }
        }
        RayGrid { tris, min: b.min, cell, dims, cells }
    }

    fn hit_t(&self, ti: u32, o: V3, d: V3) -> Option<f64> {
        let [a, b, c] = self.tris[ti as usize];
        let (e1, e2) = (sub(b, a), sub(c, a));
        let p = cross(d, e2);
        let det = dot(e1, p);
        if det.abs() < 1e-14 {
            return None;
        }
        let inv = 1.0 / det;
        let t0 = sub(o, a);
        let u = dot(t0, p) * inv;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = cross(t0, e1);
        let v = dot(d, q) * inv;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let t = dot(e2, q) * inv;
        if t > 1e-7 { Some(t) } else { None }
    }

    /// Walk the cells along the ray up to `max_t`; `visit(t)` is called for every hit inside the current cell
    /// (so a triangle spanning several cells is counted once). Stops early when `visit` returns false.
    fn walk(&self, o: V3, d: V3, max_t: f64, mut visit: impl FnMut(f64) -> bool) {
        // clip the ray to the grid box
        let mut t_near = 0.0f64;
        let mut t_far = max_t;
        for k in 0..3 {
            let lo = self.min[k];
            let hi = self.min[k] + self.dims[k] as f64 * self.cell;
            if d[k].abs() < 1e-12 {
                if o[k] < lo || o[k] > hi { return; }
            } else {
                let (mut t1, mut t2) = ((lo - o[k]) / d[k], (hi - o[k]) / d[k]);
                if t1 > t2 { std::mem::swap(&mut t1, &mut t2); }
                t_near = t_near.max(t1);
                t_far = t_far.min(t2);
                if t_near > t_far { return; }
            }
        }
        let p = add(o, scale(d, t_near + 1e-9));
        let mut ix = [0i64; 3];
        let mut step = [0i64; 3];
        let mut t_max = [f64::INFINITY; 3];
        let mut t_delta = [f64::INFINITY; 3];
        for k in 0..3 {
            ix[k] = (((p[k] - self.min[k]) / self.cell).floor() as i64).clamp(0, self.dims[k] as i64 - 1);
            if d[k] > 1e-12 {
                step[k] = 1;
                t_delta[k] = self.cell / d[k];
                t_max[k] = (self.min[k] + (ix[k] + 1) as f64 * self.cell - o[k]) / d[k];
            } else if d[k] < -1e-12 {
                step[k] = -1;
                t_delta[k] = -self.cell / d[k];
                t_max[k] = (self.min[k] + ix[k] as f64 * self.cell - o[k]) / d[k];
            }
        }
        let mut t_enter = t_near;
        loop {
            let c = ix[0] as usize + self.dims[0] * (ix[1] as usize + self.dims[1] * ix[2] as usize);
            let t_exit = t_max[0].min(t_max[1]).min(t_max[2]).min(t_far);
            for &ti in &self.cells[c] {
                if let Some(t) = self.hit_t(ti, o, d) {
                    if t >= t_enter - 1e-9 && t <= t_exit + 1e-9 && t < max_t && !visit(t) {
                        return;
                    }
                }
            }
            if t_exit >= t_far - 1e-12 {
                return;
            }
            let k = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] { 0 } else if t_max[1] <= t_max[2] { 1 } else { 2 };
            ix[k] += step[k];
            if ix[k] < 0 || ix[k] >= self.dims[k] as i64 {
                return;
            }
            t_enter = t_max[k];
            t_max[k] += t_delta[k];
        }
    }

    pub fn any_hit(&self, o: V3, d: V3, max_t: f64) -> bool {
        let mut hit = false;
        self.walk(o, d, max_t, |_| { hit = true; false });
        hit
    }

    pub fn count_hits(&self, o: V3, d: V3) -> usize {
        let mut n = 0;
        self.walk(o, d, 1e9, |_| { n += 1; true });
        n
    }

    /// True when `p` is in air (even number of surface crossings along a skewed ray).
    pub fn is_air(&self, p: V3) -> bool {
        let a = self.count_hits(p, normalize([1.0, 0.0137, 0.0071]));
        let b = self.count_hits(p, normalize([-0.0113, 1.0, 0.0159]));
        // parity must agree; when it does not (ray grazed an edge) treat as air
        a % 2 == 0 || b % 2 == 0
    }

    /// How enclosed a point is: number of 26 directions that hit the mesh within `max_t`.
    pub fn enclosure(&self, p: V3, max_t: f64) -> usize {
        let mut n = 0;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 { continue; }
                    if self.any_hit(p, normalize([dx as f64, dy as f64, dz as f64]), max_t) { n += 1; }
                }
            }
        }
        n
    }
}

/// A curved opening found automatically.
#[derive(Clone, Debug)]
pub struct CurvedOpening {
    pub cap: PortCap,
    pub perimeter: f64,
    pub planarity_dev: f64,
    /// 0..1: how clearly one side of the cap is an enclosed duct interior and the other side open air.
    pub confidence: f64,
}

/// Find curved (non-planar) openings automatically: closed edge loops of smooth surface patches that are
/// clearly non-flat, spanned by a cap surface, with enclosed air behind it and open air in front.
pub fn find_curved_openings(mesh: &TriMesh, min_perimeter: f64, max_perimeter: f64, feature_angle_deg: f64) -> Vec<CurvedOpening> {
    find_curved_openings_with(mesh, &CurvedParams { min_perimeter, max_perimeter, feature_angle_deg, ..Default::default() })
}

/// Tuning knobs for [`find_curved_openings_with`].
#[derive(Clone, Debug)]
pub struct CurvedParams {
    pub min_perimeter: f64,
    pub max_perimeter: f64,
    pub feature_angle_deg: f64,
    /// A loop must stray at least this far (mm) from its best-fit plane to count as non-planar.
    pub min_off_plane: f64,
    /// ... and at least this fraction of its perimeter.
    pub min_off_plane_rel: f64,
    pub min_confidence: f64,
    pub min_area: f64,
    /// Also return loops that are flat (the caller removes those the flat-hole detector already found).
    pub include_flat: bool,
}

impl Default for CurvedParams {
    fn default() -> Self {
        CurvedParams { min_perimeter: 12.0, max_perimeter: 300.0, feature_angle_deg: 25.0, min_off_plane: 0.3, min_off_plane_rel: 0.02, min_confidence: 0.2, min_area: 4.0, include_flat: false }
    }
}

pub fn find_curved_openings_with(mesh: &TriMesh, prm: &CurvedParams) -> Vec<CurvedOpening> {
    let (min_perimeter, max_perimeter, feature_angle_deg) = (prm.min_perimeter, prm.max_perimeter, prm.feature_angle_deg);
    let mesh = mesh.welded(1e-5);
    let nt = mesh.triangles.len();
    if nt == 0 {
        return vec![];
    }
    let cos_tol = feature_angle_deg.to_radians().cos();
    let normals: Vec<V3> = (0..nt).map(|i| { let [a, b, c] = mesh.tri(i); normalize(cross(sub(b, a), sub(c, a))) }).collect();
    let mut edges: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (i, t) in mesh.triangles.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            edges.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    // label smooth regions
    let mut region = vec![usize::MAX; nt];
    let mut n_reg = 0;
    for s in 0..nt {
        if region[s] != usize::MAX { continue; }
        let mut stack = vec![s];
        region[s] = n_reg;
        while let Some(t) = stack.pop() {
            let tv = mesh.triangles[t];
            for k in 0..3 {
                let (a, b) = (tv[k], tv[(k + 1) % 3]);
                let nb = &edges[&(a.min(b), a.max(b))];
                if nb.len() != 2 { continue; }
                for &t2 in nb {
                    if region[t2] == usize::MAX && dot(normals[t], normals[t2]) >= cos_tol {
                        region[t2] = n_reg;
                        stack.push(t2);
                    }
                }
            }
        }
        n_reg += 1;
    }
    // boundary edges per region, chained into loops
    let mut per_region: Vec<HashMap<u32, Vec<u32>>> = vec![HashMap::new(); n_reg];
    for t in 0..nt {
        let tv = mesh.triangles[t];
        for k in 0..3 {
            let (a, b) = (tv[k], tv[(k + 1) % 3]);
            let nb = &edges[&(a.min(b), a.max(b))];
            let other_same = nb.iter().any(|&x| x != t && region[x] == region[t]);
            if nb.len() != 2 || !other_same {
                per_region[region[t]].entry(a).or_default().push(b);
            }
        }
    }
    let mut seen: std::collections::HashSet<u64> = Default::default();
    let mut loops: Vec<RimLoop> = vec![];
    let dbg = std::env::var("PG_DEBUG").is_ok();
    let (mut c_closed, mut c_perim, mut c_flat) = (0usize, 0usize, 0usize);
    for mut next in per_region {
        let mut starts: Vec<u32> = next.keys().copied().collect();
        starts.sort_unstable();
        for s in starts {
            while next.get(&s).map_or(false, |v| !v.is_empty()) {
                let mut lp = vec![s];
                let mut cur = s;
                let mut closed = false;
                loop {
                    let Some(nx) = next.get_mut(&cur).and_then(|v| v.pop()) else { break };
                    if nx == s { closed = true; break; }
                    lp.push(nx);
                    cur = nx;
                    if lp.len() > 20000 { break; }
                }
                if !closed || lp.len() < 8 { continue; }
                c_closed += 1;
                let points: Vec<V3> = lp.iter().map(|&i| mesh.vertices[i as usize]).collect();
                let perimeter: f64 = (0..points.len()).map(|i| len(sub(points[(i + 1) % points.len()], points[i]))).sum();
                if perimeter < min_perimeter || perimeter > max_perimeter { continue; }
                c_perim += 1;
                let (c, n) = loop_frame(&points);
                let dev = points.iter().map(|p| dot(sub(*p, c), n).abs()).fold(0.0, f64::max);
                if !prm.include_flat && dev < (prm.min_off_plane_rel * perimeter).max(prm.min_off_plane) { c_flat += 1; if dbg && perimeter < 100.0 { eprintln!("  too flat: perim {:.0} off-plane {:.2} at ({:.1},{:.1},{:.1})", perimeter, dev, c[0], c[1], c[2]); } continue; } // flat loops are the planar detector's job
                // the same physical loop borders two patches; keep one copy
                let mut ids: Vec<u32> = lp.clone();
                ids.sort_unstable();
                let key = ids.iter().fold(1469598103934665603u64, |h, &v| (h ^ v as u64).wrapping_mul(1099511628211));
                if !seen.insert(key) { continue; }
                loops.push(RimLoop { points, perimeter, planarity_dev: dev });
            }
        }
    }
    let grid = RayGrid::new(&mesh);
    let mut out = vec![];
    let (mut c_cap, mut c_mat, mut c_air, mut c_conf) = (0usize, 0usize, 0usize, 0usize);
    if dbg { eprintln!("loops: closed {c_closed}, perimeter-ok {c_perim}, too flat {c_flat}, kept for cap test {}", loops.len()); }
    for l in loops {
        let tris = triangulate_loop(&l.points);
        if tris.is_empty() { continue; }
        // Interior of the cap: triangle centroids far from the rim (the rim itself sits against the wall).
        let cents: Vec<V3> = tris.iter().map(|t| scale(add(add(t[0], t[1]), t[2]), 1.0 / 3.0)).collect();
        let depth: Vec<f64> = cents.iter().map(|c| l.points.iter().map(|p| len(sub(*p, *c))).fold(f64::INFINITY, f64::min)).collect();
        let dmax = depth.iter().cloned().fold(0.0, f64::max);
        let interior: Vec<usize> = (0..cents.len()).filter(|&i| depth[i] >= 0.5 * dmax).collect();
        // the cap must run through air, not through the material
        let in_material = interior.iter().filter(|&&i| !grid.is_air(cents[i])).count();
        c_cap += 1;
        if in_material * 10 > interior.len() * 3 {
            c_mat += 1;
            if dbg { { let (cc, _) = loop_frame(&l.points); eprintln!("  rejected (cap inside material): perim {:.0} off-plane {:.2} at ({:.1},{:.1},{:.1}) samples {} in material {}", l.perimeter, l.planarity_dev, cc[0], cc[1], cc[2], interior.len(), in_material); } }
            continue;
        }
        let probe_base = cents[(0..cents.len()).max_by(|&a, &b| depth[a].partial_cmp(&depth[b]).unwrap()).unwrap()];
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
        if a_tot < prm.min_area { continue; }
        let ctr = scale(ctr, 1.0 / a_tot);
        let n0 = normalize(n0);
        // Probe both sides of the cap. The nearest probe must be in air (otherwise the cap lies against material);
        // farther probes that land in material (tight spaces) are skipped, and enclosure is averaged over the rest.
        let mut side = [(0.0f64, 0usize); 2]; // (sum of enclosure, valid probes) for [plus, minus]
        let mut near_air = [true, true];
        for (di, dist) in [1.5, 3.0, 5.0].iter().enumerate() {
            for (si, sign) in [1.0, -1.0].iter().enumerate() {
                let p = add(probe_base, scale(n0, sign * dist));
                if grid.is_air(p) {
                    side[si].0 += grid.enclosure(p, 60.0) as f64;
                    side[si].1 += 1;
                } else if di == 0 {
                    near_air[si] = false;
                }
            }
        }
        if side[0].1 == 0 || side[1].1 == 0 { c_air += 1; if dbg { { let (cc, _) = loop_frame(&l.points); eprintln!("  rejected (probe in material): perim {:.0} off-plane {:.2} at ({:.1},{:.1},{:.1}) near_air {:?} valid probes {} / {}", l.perimeter, l.planarity_dev, cc[0], cc[1], cc[2], near_air, side[0].1, side[1].1); } } continue; } // against material: not an opening
        let (plus, minus) = (side[0].0 / side[0].1 as f64, side[1].0 / side[1].1 as f64);
        let (lumen_enc, open_enc, n_out) = if minus >= plus { (minus, plus, n0) } else { (plus, minus, scale(n0, -1.0)) };
        let confidence = (lumen_enc - open_enc) / 26.0;
        if confidence < prm.min_confidence { c_conf += 1; if dbg { { let (cc, _) = loop_frame(&l.points); eprintln!("  rejected (confidence): perim {:.0} area {:.0} at ({:.1},{:.1},{:.1}) enc lumen {:.1} open {:.1} (conf {:.2})", l.perimeter, a_tot, cc[0], cc[1], cc[2], lumen_enc, open_enc, confidence); } } continue; }
        let flip = dot(n0, n_out) < 0.0;
        let triangles: Vec<[V3; 3]> = tris.into_iter().map(|t| if flip { [t[0], t[2], t[1]] } else { t }).collect();
        let flat = l.planarity_dev < 0.05;
        let cap = if flat {
            // a flat loop is an ordinary planar opening; use the loop's own plane
            PortCap { origin: ctr, normal: n_out, polygon: l.points.clone(), triangles: vec![], holes: vec![] }
        } else {
            PortCap { origin: ctr, normal: n_out, polygon: l.points.clone(), triangles, holes: vec![] }
        };
        out.push(CurvedOpening { cap, perimeter: l.perimeter, planarity_dev: l.planarity_dev, confidence });
    }
    if dbg { eprintln!("cap tested {c_cap}, in material {c_mat}, solid on a side {c_air}, low confidence {c_conf}, accepted {}", out.len()); }
    // A rim has an outer and an inner edge loop around the same opening; keep the inner (smaller) one.
    let mut keep = vec![true; out.len()];
    for i in 0..out.len() {
        for j in 0..out.len() {
            if i == j || !keep[i] { continue; }
            let (a, b) = (&out[i], &out[j]); // is `a` the inner loop of `b`?
            let (ca, cb) = (a.cap.centroid(), b.cap.centroid());
            if a.cap.area() < b.cap.area() && len(sub(ca, cb)) < 0.35 * b.cap.area().sqrt() && dot(a.cap.normal, b.cap.normal) > 0.9 {
                keep[j] = false;
            }
        }
    }
    let mut out: Vec<CurvedOpening> = out.into_iter().zip(keep).filter(|x| x.1).map(|x| x.0).collect();
    out.sort_by(|a, b| b.cap.area().partial_cmp(&a.cap.area()).unwrap());
    out
}
