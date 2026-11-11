//! Flow trajectories: RK4 streamlines through a velocity field, seeded on inlet openings.
use peregrine_core::port_cap::{point_in_polygon, PortCap};
use peregrine_core::vec3::*;
use rayon::prelude::*;

/// Velocity field on the solver grid. `vel` is in m/s (zero in solid cells); lengths are mm.
#[derive(Clone, Copy)]
pub struct VelocityField<'a> {
    pub dims: [usize; 3],
    pub origin: V3,
    pub dx: f64,
    /// 1 = solid.
    pub cells: &'a [u8],
    pub vel: &'a [[f32; 3]],
}

#[derive(Clone, Debug)]
pub struct Streamline {
    /// Index of the source (or pseudo-source) that seeded the line.
    pub source: usize,
    pub pts: Vec<[f32; 3]>,
    /// Speed (m/s) at each point.
    pub speed: Vec<f32>,
}

/// Seeds requested on one source: openings and how many lines start on each.
#[derive(Clone, Debug)]
pub struct SeedSpec {
    pub source: usize,
    pub caps: Vec<(PortCap, usize)>,
}

/// Spread `n` points over the opening, nudged 1.5 cells into the duct. Large counts also get a ring of
/// seeds hugging the edge of the opening, where walls, scoops and ramps act on the flow.
pub fn seed_points(cap: &PortCap, n: usize, dx: f64) -> Vec<V3> {
    if n == 0 || cap.polygon.len() < 3 {
        return vec![];
    }
    if cap.is_surface() {
        return seed_points_surface(cap, n, dx);
    }
    let poly = cap.polygon_2d();
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    let mut ctr = [0.0; 2];
    for p in &poly {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
            ctr[k] += p[k] / poly.len() as f64;
        }
    }
    let (u, v) = cap.basis();
    let to3 = |p: [f64; 2]| {
        let on_plane = add(cap.origin, add(scale(u, p[0]), scale(v, p[1])));
        sub(on_plane, scale(cap.normal, 1.5 * dx))
    };
    // edge ring: evenly spaced along the outline, pulled one cell towards the centre
    let ring_n = if n >= 8 { (n / 4).max(6) } else { 0 };
    let mut ring: Vec<[f64; 2]> = vec![];
    if ring_n > 0 {
        let m = poly.len();
        let seg: Vec<f64> = (0..m).map(|i| { let (a, b) = (poly[i], poly[(i + 1) % m]); ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() }).collect();
        let perim: f64 = seg.iter().sum();
        for k in 0..ring_n {
            let mut t = perim * (k as f64 + 0.5) / ring_n as f64;
            let mut i = 0;
            while i + 1 < m && t > seg[i] { t -= seg[i]; i += 1; }
            let (a, b) = (poly[i], poly[(i + 1) % m]);
            let f = if seg[i] > 0.0 { t / seg[i] } else { 0.0 };
            let q = [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
            let d = [ctr[0] - q[0], ctr[1] - q[1]];
            let dl = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
            let inset = dx.min(dl * 0.5);
            ring.push([q[0] + d[0] / dl * inset, q[1] + d[1] / dl * inset]);
        }
    }
    let n_inner = n - ring.len();
    let area = cap.area().max(1e-9);
    let mut h = (area / n_inner.max(1) as f64).sqrt();
    let mut inner: Vec<[f64; 2]> = vec![];
    for _ in 0..24 {
        let mut pts = vec![];
        let mut y = lo[1] + h * 0.5;
        while y < hi[1] {
            let mut x = lo[0] + h * 0.5;
            while x < hi[0] {
                if point_in_polygon(&poly, [x, y]) {
                    pts.push([x, y]);
                }
                x += h;
            }
            y += h;
        }
        if pts.len() >= n_inner || h < dx * 0.05 {
            if pts.len() > n_inner && n_inner > 0 {
                let step = pts.len() as f64 / n_inner as f64;
                pts = (0..n_inner).map(|i| pts[(i as f64 * step) as usize]).collect();
            }
            inner = pts;
            break;
        }
        inner = pts;
        h *= 0.8;
    }
    inner.into_iter().chain(ring).map(to3).collect()
}

/// Seeds for a curved opening: low-discrepancy points over the cap triangles (area weighted) plus a ring
/// along its edge, all moved 1.5 cells into the duct.
fn seed_points_surface(cap: &PortCap, n: usize, dx: f64) -> Vec<V3> {
    let frac = |x: f64| x - x.floor();
    let tris = &cap.triangles;
    let areas: Vec<f64> = tris.iter().map(|t| 0.5 * len(cross(sub(t[1], t[0]), sub(t[2], t[0])))).collect();
    let total: f64 = areas.iter().sum();
    if total < 1e-12 {
        return vec![];
    }
    let mut cum = Vec::with_capacity(areas.len());
    let mut acc = 0.0;
    for a in &areas {
        acc += a;
        cum.push(acc);
    }
    let ring_n = if n >= 8 { (n / 4).max(6) } else { 0 };
    let mut out = vec![];
    for i in 0..(n - ring_n) {
        let u = frac(0.5 + i as f64 * 0.754_877_666_246_692_7) * total;
        let ti = cum.partition_point(|&c| c < u).min(tris.len() - 1);
        let (a, b) = (frac(0.5 + i as f64 * 0.618_033_988_749_894_9), frac(0.5 + i as f64 * 0.414_213_562_373_095));
        let (ra, rb) = (a.sqrt(), b);
        let t = tris[ti];
        let p = add(add(scale(t[0], 1.0 - ra), scale(t[1], ra * (1.0 - rb))), scale(t[2], ra * rb));
        // step back along the opening's mean normal (not the local one) so seeds do not drift sideways
        out.push(sub(p, scale(cap.normal, 1.5 * dx)));
    }
    if ring_n > 0 {
        let m = cap.polygon.len();
        let seg: Vec<f64> = (0..m).map(|i| len(sub(cap.polygon[(i + 1) % m], cap.polygon[i]))).collect();
        let perim: f64 = seg.iter().sum();
        let ctr = cap.centroid();
        for k in 0..ring_n {
            let mut t = perim * (k as f64 + 0.5) / ring_n as f64;
            let mut i = 0;
            while i + 1 < m && t > seg[i] { t -= seg[i]; i += 1; }
            let f = if seg[i] > 0.0 { t / seg[i] } else { 0.0 };
            let (a, b) = (cap.polygon[i], cap.polygon[(i + 1) % m]);
            let q = add(a, scale(sub(b, a), f));
            let inset = normalize(sub(ctr, q));
            out.push(sub(add(q, scale(inset, 1.5 * dx)), scale(cap.normal, 1.5 * dx)));
        }
    }
    out
}

fn cell_index(f: &VelocityField, p: V3) -> Option<usize> {
    let g = [(p[0] - f.origin[0]) / f.dx, (p[1] - f.origin[1]) / f.dx, (p[2] - f.origin[2]) / f.dx];
    if g.iter().zip(f.dims).any(|(&a, d)| a < 0.0 || a >= d as f64) {
        return None;
    }
    Some(g[0] as usize + f.dims[0] * (g[1] as usize + f.dims[1] * g[2] as usize))
}

/// Trilinear velocity at `p` (cell-centred samples). `None` outside the grid.
pub fn sample(f: &VelocityField, p: V3) -> Option<V3> {
    let g = [(p[0] - f.origin[0]) / f.dx - 0.5, (p[1] - f.origin[1]) / f.dx - 0.5, (p[2] - f.origin[2]) / f.dx - 0.5];
    let mut i0 = [0usize; 3];
    let mut t = [0.0f64; 3];
    for k in 0..3 {
        if g[k] < -0.5 || g[k] > f.dims[k] as f64 - 0.5 {
            return None;
        }
        let fl = g[k].floor().clamp(0.0, (f.dims[k] - 1) as f64);
        i0[k] = fl as usize;
        t[k] = (g[k] - fl).clamp(0.0, 1.0);
    }
    let mut out = [0.0; 3];
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let (x, y, z) = ((i0[0] + dx).min(f.dims[0] - 1), (i0[1] + dy).min(f.dims[1] - 1), (i0[2] + dz).min(f.dims[2] - 1));
                let w = (if dx == 0 { 1.0 - t[0] } else { t[0] }) * (if dy == 0 { 1.0 - t[1] } else { t[1] }) * (if dz == 0 { 1.0 - t[2] } else { t[2] });
                let v = f.vel[x + f.dims[0] * (y + f.dims[1] * z)];
                for a in 0..3 {
                    out[a] += w * v[a] as f64;
                }
            }
        }
    }
    Some(out)
}

/// Follow the flow from `seed` until it leaves the domain, hits a wall, stalls or reaches `max_len` mm.
pub fn trace(f: &VelocityField, seed: V3, source: usize, max_len: f64) -> Option<Streamline> {
    let ds = 0.5 * f.dx;
    let max_steps = (max_len / ds) as usize;
    let vmin = 0.03; // m/s: slower than this counts as stagnant
    let dir = |p: V3| -> Option<V3> {
        let v = sample(f, p)?;
        let s = len(v);
        if s < vmin { None } else { Some(scale(v, 1.0 / s)) }
    };
    let mut p = seed;
    let mut line = Streamline { source, pts: vec![], speed: vec![] };
    for step in 0..max_steps {
        let Some(idx) = cell_index(f, p) else { break };
        if f.cells[idx] == 1 {
            break;
        }
        let v0 = sample(f, p)?;
        if len(v0) < vmin {
            break;
        }
        if step % 2 == 0 {
            line.pts.push([p[0] as f32, p[1] as f32, p[2] as f32]);
            line.speed.push(len(v0) as f32);
        }
        let Some(k1) = dir(p) else { break };
        let Some(k2) = dir(add(p, scale(k1, 0.5 * ds))) else { break };
        let Some(k3) = dir(add(p, scale(k2, 0.5 * ds))) else { break };
        let Some(k4) = dir(add(p, scale(k3, ds))) else { break };
        let d = add(add(k1, scale(add(k2, k3), 2.0)), k4);
        p = add(p, scale(d, ds / 6.0));
    }
    if line.pts.len() >= 3 { Some(line) } else { None }
}

/// Trace all requested seeds in parallel. Seeds that start inside a (voxelised) wall are moved towards
/// the centre of their opening until they sit in fluid, instead of being dropped.
pub fn streamlines(f: &VelocityField, specs: &[SeedSpec], max_len: f64) -> Vec<Streamline> {
    let mut seeds: Vec<(usize, V3)> = vec![];
    for s in specs {
        for (cap, n) in &s.caps {
            let centre = cap.centroid();
            for p in seed_points(cap, *n, f.dx) {
                let mut q = p;
                let mut ok = false;
                for _ in 0..10 {
                    match cell_index(f, q) {
                        Some(i) if f.cells[i] != 1 => { ok = true; break; }
                        None => break,
                        _ => {}
                    }
                    q = add(q, scale(normalize(sub(centre, q)), 0.5 * f.dx));
                }
                if ok {
                    seeds.push((s.source, q));
                }
            }
        }
    }
    seeds.par_iter().filter_map(|(src, p)| trace(f, *p, *src, max_len)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform(n: usize, solid_from_x: Option<usize>) -> (Vec<u8>, Vec<[f32; 3]>) {
        let mut cells = vec![0u8; n * n * n];
        let mut vel = vec![[2.0f32, 0.0, 0.0]; n * n * n];
        if let Some(sx) = solid_from_x {
            for z in 0..n { for y in 0..n { for x in sx..n {
                let i = x + n * (y + n * z);
                cells[i] = 1;
                vel[i] = [0.0; 3];
            } } }
        }
        (cells, vel)
    }

    #[test]
    fn straight_line_in_uniform_flow() {
        let (cells, vel) = uniform(40, None);
        let f = VelocityField { dims: [40; 3], origin: [0.0; 3], dx: 1.0, cells: &cells, vel: &vel };
        let l = trace(&f, [5.0, 20.0, 20.0], 0, 500.0).unwrap();
        let last = l.pts.last().unwrap();
        assert!(last[0] > 38.0, "{last:?}");
        assert!((last[1] - 20.0).abs() < 1e-3 && (last[2] - 20.0).abs() < 1e-3);
        assert!(l.speed.iter().all(|&s| (s - 2.0).abs() < 1e-4));
    }

    #[test]
    fn stops_at_walls() {
        let (cells, vel) = uniform(40, Some(30));
        let f = VelocityField { dims: [40; 3], origin: [0.0; 3], dx: 1.0, cells: &cells, vel: &vel };
        let l = trace(&f, [5.0, 20.0, 20.0], 0, 500.0).unwrap();
        assert!(l.pts.last().unwrap()[0] <= 30.0);
    }

    #[test]
    fn seeds_cover_a_curved_opening() {
        use peregrine_core::{cap_from_loop, shapes, smooth_region_loops};
        let tube = shapes::scooped_round_tube(6.0, 4.0, 30.0, 1.5, 64).welded(1e-6);
        let seed = (0..tube.triangles.len()).find(|&i| {
            let [a, b, c] = tube.tri(i);
            let ctr = scale(add(add(a, b), c), 1.0 / 3.0);
            let r = (ctr[1] * ctr[1] + ctr[2] * ctr[2]).sqrt();
            ctr[0] > 27.0 && r > 4.3 && r < 5.7
        }).unwrap();
        let loops = smooth_region_loops(&tube, seed, 25.0);
        let inner = loops.iter().min_by(|a, b| a.perimeter.partial_cmp(&b.perimeter).unwrap()).unwrap();
        let cap = cap_from_loop(&inner.points, &tube).unwrap();
        for n in [4usize, 20, 60] {
            let s = seed_points(&cap, n, 0.5);
            assert!(s.len() as f64 >= 0.8 * n as f64 && s.len() <= n, "n {n}: {}", s.len());
            for p in &s {
                let r = (p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!(r <= 4.05, "seed outside the bore: r {r}");
                assert!(p[0] < 30.0 + 1.5, "seed in front of the cap: {p:?}");
            }
        }
    }

    #[test]
    fn seeds_fill_the_opening() {
        let cap = PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0, -5.0, -5.0], [0.0, 5.0, -5.0], [0.0, 5.0, 5.0], [0.0, -5.0, 5.0]], triangles: vec![] };
        for n in [1usize, 5, 20, 60] {
            let s = seed_points(&cap, n, 0.5);
            assert!(!s.is_empty() && s.len() <= n, "n {n}: {}", s.len());
            assert!(s.len() as f64 >= 0.5 * n as f64, "n {n}: {}", s.len());
            if n >= 20 {
                // some seeds must sit close to the edge of the 10x10 opening
                let edge = s.iter().filter(|p| p[1].abs().max(p[2].abs()) > 4.0).count();
                assert!(edge >= n / 6, "n {n}: only {edge} near the edge");
            }
            for p in &s {
                assert!((p[0] + 0.75).abs() < 1e-9, "inside the duct (-x side): {p:?}");
                assert!(p[1].abs() <= 5.0 && p[2].abs() <= 5.0);
            }
        }
    }
}
