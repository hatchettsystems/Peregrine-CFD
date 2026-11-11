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

/// Evenly spread `n` points over the opening, nudged 1.5 cells into the duct.
pub fn seed_points(cap: &PortCap, n: usize, dx: f64) -> Vec<V3> {
    if n == 0 || cap.polygon.len() < 3 {
        return vec![];
    }
    let poly = cap.polygon_2d();
    let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
    for p in &poly {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    let (u, v) = cap.basis();
    let to3 = |p: [f64; 2]| {
        let on_plane = add(cap.origin, add(scale(u, p[0]), scale(v, p[1])));
        sub(on_plane, scale(cap.normal, 1.5 * dx))
    };
    let area = cap.area().max(1e-9);
    let mut h = (area / n as f64).sqrt();
    let mut best: Vec<[f64; 2]> = vec![];
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
        if pts.len() >= n || h < dx * 0.05 {
            // thin evenly down to n
            if pts.len() > n {
                let step = pts.len() as f64 / n as f64;
                pts = (0..n).map(|i| pts[(i as f64 * step) as usize]).collect();
            }
            best = pts;
            break;
        }
        best = pts;
        h *= 0.8;
    }
    best.into_iter().map(to3).collect()
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

/// Trace all requested seeds in parallel.
pub fn streamlines(f: &VelocityField, specs: &[SeedSpec], max_len: f64) -> Vec<Streamline> {
    let mut seeds: Vec<(usize, V3)> = vec![];
    for s in specs {
        for (cap, n) in &s.caps {
            for p in seed_points(cap, *n, f.dx) {
                seeds.push((s.source, p));
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
    fn seeds_fill_the_opening() {
        let cap = PortCap { origin: [0.0; 3], normal: [1.0, 0.0, 0.0], polygon: vec![[0.0, -5.0, -5.0], [0.0, 5.0, -5.0], [0.0, 5.0, 5.0], [0.0, -5.0, 5.0]] };
        for n in [1usize, 5, 20, 60] {
            let s = seed_points(&cap, n, 0.5);
            assert!(!s.is_empty() && s.len() <= n, "n {n}: {}", s.len());
            assert!(s.len() as f64 >= 0.5 * n as f64, "n {n}: {}", s.len());
            for p in &s {
                assert!((p[0] + 0.75).abs() < 1e-9, "inside the duct (-x side): {p:?}");
                assert!(p[1].abs() <= 5.0 && p[2].abs() <= 5.0);
            }
        }
    }
}
