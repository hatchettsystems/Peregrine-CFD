//! Cartesian voxelisation: solids by ray parity, port caps, fluid-region flood fill.
use peregrine_core::port_cap::{point_in_polygon, PortCap};
use peregrine_core::vec3::*;
use peregrine_core::{Aabb, TriMesh};
use rayon::prelude::*;

pub const FLUID: u8 = 0;
pub const SOLID: u8 = 1;
/// Cell code of port `k` is `PORT_BASE + k`.
pub const PORT_BASE: u8 = 2;

#[derive(Clone, Debug)]
pub struct Grid {
    pub origin: V3,
    /// Cell size in mm.
    pub dx: f64,
    pub dims: [usize; 3],
}

impl Grid {
    pub fn from_bounds(b: &Aabb, dx: f64) -> Grid {
        let dims = [0, 1, 2].map(|i| (((b.max[i] - b.min[i]) / dx).ceil() as usize).max(1));
        Grid { origin: b.min, dx, dims }
    }
    pub fn len(&self) -> usize { self.dims[0] * self.dims[1] * self.dims[2] }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    pub fn idx(&self, x: usize, y: usize, z: usize) -> usize { x + self.dims[0] * (y + self.dims[1] * z) }
    pub fn xyz(&self, i: usize) -> [usize; 3] {
        let x = i % self.dims[0];
        let y = (i / self.dims[0]) % self.dims[1];
        [x, y, i / (self.dims[0] * self.dims[1])]
    }
    pub fn center(&self, x: usize, y: usize, z: usize) -> V3 {
        [
            self.origin[0] + (x as f64 + 0.5) * self.dx,
            self.origin[1] + (y as f64 + 0.5) * self.dx,
            self.origin[2] + (z as f64 + 0.5) * self.dx,
        ]
    }
    /// Rough memory (bytes) of a D3Q19 f32 solver: single-buffer GPU layout and double-buffer CPU layout.
    pub fn memory_estimate(&self) -> (u64, u64) {
        let n = self.len() as u64;
        (n * 19 * 4 + n, n * 19 * 4 * 2 + n * 17)
    }
}

/// Inside/outside by +X ray parity. Returns (inside, number of columns with odd crossing count).
pub fn voxelize_inside(mesh: &TriMesh, grid: &Grid) -> (Vec<bool>, usize) {
    let [nx, ny, nz] = grid.dims;
    // Irrational jitter keeps ray positions off mesh vertices/edges that sit on the grid.
    let jy = grid.dx * 1.234_567e-4;
    let jz = grid.dx * 2.718_281e-4;
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); ny * nz];
    let ycoord = |j: usize| grid.origin[1] + (j as f64 + 0.5) * grid.dx + jy;
    let zcoord = |k: usize| grid.origin[2] + (k as f64 + 0.5) * grid.dx + jz;
    for t in 0..mesh.triangles.len() {
        let [a, b, c] = mesh.tri(t);
        let area2 = (b[1] - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (b[2] - a[2]);
        if area2.abs() < 1e-14 {
            continue;
        }
        let ymin = a[1].min(b[1]).min(c[1]);
        let ymax = a[1].max(b[1]).max(c[1]);
        let zmin = a[2].min(b[2]).min(c[2]);
        let zmax = a[2].max(b[2]).max(c[2]);
        let j0 = (((ymin - jy - grid.origin[1]) / grid.dx - 0.5).floor().max(0.0)) as usize;
        let k0 = (((zmin - jz - grid.origin[2]) / grid.dx - 0.5).floor().max(0.0)) as usize;
        let j1 = ((((ymax - jy - grid.origin[1]) / grid.dx + 0.5).ceil()).max(0.0) as usize).min(ny);
        let k1 = ((((zmax - jz - grid.origin[2]) / grid.dx + 0.5).ceil()).max(0.0) as usize).min(nz);
        for k in k0..k1 {
            let z = zcoord(k);
            for j in j0..j1 {
                let y = ycoord(j);
                // barycentric in (y,z)
                let w1 = ((y - a[1]) * (c[2] - a[2]) - (c[1] - a[1]) * (z - a[2])) / area2;
                let w2 = ((b[1] - a[1]) * (z - a[2]) - (y - a[1]) * (b[2] - a[2])) / area2;
                let w0 = 1.0 - w1 - w2;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    cols[j + ny * k].push(w0 * a[0] + w1 * b[0] + w2 * c[0]);
                }
            }
        }
    }
    let odd = std::sync::atomic::AtomicUsize::new(0);
    let mut inside = vec![false; nx * ny * nz];
    // fill per (j,k) column; rows of x are contiguous so chunk by nx
    inside.par_chunks_mut(nx).enumerate().for_each(|(row, out)| {
        let xs = &cols[row];
        if xs.is_empty() {
            return;
        }
        let mut xs = xs.clone();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if xs.len() % 2 == 1 {
            odd.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            xs.pop();
        }
        for pair in xs.chunks(2) {
            let (a, b) = (pair[0], pair[1]);
            let i0 = (((a - grid.origin[0]) / grid.dx - 0.5).ceil().max(0.0)) as usize;
            let i1 = ((((b - grid.origin[0]) / grid.dx - 0.5).ceil()).max(0.0) as usize).min(nx);
            for slot in out.iter_mut().take(i1).skip(i0) {
                *slot = true;
            }
        }
    });
    (inside, odd.into_inner())
}

thread_local! {
    /// Set while a connectivity probe (see [`behind_volumes`]) builds its domain: the probe must see the geometry as it
    /// is, without the wall-thickening sealers that the simulation domain uses.
    static PROBE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

fn probing() -> bool {
    PROBE.with(|p| p.get())
}

#[derive(Clone, Debug)]
pub struct PortSpec {
    pub cap: PortCap,
    /// Close the volume behind the cap (inlets, where a fan or wall sits).
    pub plug: bool,
    /// Fan mount seal (mm): also close fluid cells in a collar of this width around the rim of the cap, as the
    /// housing of a mounted fan would. 0 = only the thin sliver at the rim.
    pub collar_mm: f64,
}

#[derive(Debug)]
pub struct Domain {
    pub grid: Grid,
    pub cells: Vec<u8>,
    /// Number of cells assigned to each port.
    pub port_cell_counts: Vec<usize>,
    /// Ray columns with odd crossing counts (non-watertight mesh indicator).
    pub odd_columns: usize,
    /// Fluid cells not connected to any port that were turned solid.
    pub sealed_cells_removed: usize,
    /// Places (mm) where [`auto_seal_leaks`] closed a small leak.
    pub repaired: Vec<V3>,
    /// Every cell the gap sealer and the leak repair closed (for reporting regions).
    pub sealed_idx: Vec<u32>,
    /// Thin gaps that were NOT closed because they are long or wide enough to be an intentional opening.
    pub review_idx: Vec<u32>,
}

/// A cluster of cells that were closed automatically.
#[derive(Clone, Debug)]
pub struct SealedRegion {
    pub center: V3,
    pub cells: usize,
    pub volume_mm3: f64,
    pub radius_mm: f64,
}

impl Domain {
    /// The automatically closed cells grouped into regions (nearby cells merge; at most `max_regions` are returned).
    pub fn sealed_regions(&self, max_regions: usize) -> Vec<SealedRegion> {
        self.cluster(&self.sealed_idx, max_regions)
    }

    /// Gaps left open for review, grouped the same way.
    pub fn review_regions(&self, max_regions: usize) -> Vec<SealedRegion> {
        self.cluster(&self.review_idx, max_regions)
    }

    fn cluster(&self, cells_idx: &[u32], max_regions: usize) -> Vec<SealedRegion> {
        use std::collections::HashMap;
        if cells_idx.is_empty() {
            return vec![];
        }
        let g = &self.grid;
        let mut bin = (3.0 / g.dx).ceil().max(2.0) as i64; // cells per bin
        loop {
            let mut bins: HashMap<(i64, i64, i64), (V3, usize)> = HashMap::new();
            for &i in cells_idx {
                let [x, y, z] = g.xyz(i as usize);
                let key = (x as i64 / bin, y as i64 / bin, z as i64 / bin);
                let c = g.center(x, y, z);
                let e = bins.entry(key).or_insert(([0.0; 3], 0));
                e.0 = add(e.0, c);
                e.1 += 1;
            }
            // merge touching bins
            let keys: Vec<(i64, i64, i64)> = bins.keys().copied().collect();
            let idx_of: HashMap<(i64, i64, i64), usize> = keys.iter().enumerate().map(|(i, k)| (*k, i)).collect();
            let mut parent: Vec<usize> = (0..keys.len()).collect();
            fn find(p: &mut Vec<usize>, a: usize) -> usize { let mut a = a; while p[a] != a { p[a] = p[p[a]]; a = p[a]; } a }
            for (i, k) in keys.iter().enumerate() {
                for dz in -1..=1 { for dy in -1..=1 { for dx in -1..=1 {
                    if let Some(&j) = idx_of.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                        if a != b { parent[a] = b; }
                    }
                } } }
            }
            let mut groups: HashMap<usize, (V3, usize, Vec<V3>)> = HashMap::new();
            for (i, k) in keys.iter().enumerate() {
                let r = find(&mut parent, i);
                let (sum, n) = bins[k];
                let e = groups.entry(r).or_insert(([0.0; 3], 0, vec![]));
                e.0 = add(e.0, sum);
                e.1 += n;
                e.2.push(scale(sum, 1.0 / n as f64));
            }
            if groups.len() <= max_regions || bin > 64 {
                let cell_vol = g.dx * g.dx * g.dx;
                let mut out: Vec<SealedRegion> = groups
                    .into_values()
                    .map(|(sum, n, bcs)| {
                        let center = scale(sum, 1.0 / n as f64);
                        let radius = bcs.iter().map(|b| len(sub(*b, center))).fold(0.0, f64::max) + 0.5 * bin as f64 * g.dx;
                        SealedRegion { center, cells: n, volume_mm3: n as f64 * cell_vol, radius_mm: radius }
                    })
                    .collect();
                out.sort_by(|a, b| b.cells.cmp(&a.cells));
                return out;
            }
            bin *= 2;
        }
    }
}

pub fn build_domain(solids: &[&TriMesh], ports: &[PortSpec], bounds: &Aabb, dx: f64) -> Domain {
    build_domain_ext(solids, &[], 0.0, ports, bounds, dx)
}

/// Like [`build_domain`], with seam sealing. `groups` lists, per duct, the indices (into `solids`) of the parts
/// that form that duct. A fluid cell squeezed between two DIFFERENT parts of the same duct, at most
/// `seam_tol_mm` apart along an axis, is filled, so small gaps between parts do not leak.
pub fn build_domain_ext(solids: &[&TriMesh], groups: &[Vec<usize>], seam_tol_mm: f64, ports: &[PortSpec], bounds: &Aabb, dx: f64) -> Domain {
    build_domain_full(solids, groups, seam_tol_mm, 0.0, ports, &[], &[], bounds, dx)
}

/// Like [`build_domain_ext`], and every cap in `blockers` is filled solid (a shut opening: screw hole, vent...).
pub fn build_domain_full(solids: &[&TriMesh], groups: &[Vec<usize>], seam_tol_mm: f64, gap_all_mm: f64, ports: &[PortSpec], blockers: &[PortCap], plugs: &[(V3, f64)], bounds: &Aabb, dx: f64) -> Domain {
    assert!(ports.len() <= (255 - PORT_BASE as usize), "too many ports");
    let grid = Grid::from_bounds(bounds, dx);
    let mut cells = vec![FLUID; grid.len()];
    let mut label = vec![u16::MAX; grid.len()]; // which solid first claimed each solid cell
    let mut odd_columns = 0;
    for (si, m) in solids.iter().enumerate() {
        let (inside, odd) = voxelize_inside(m, &grid);
        odd_columns += odd;
        for (k, i) in inside.into_iter().enumerate() {
            if i {
                cells[k] = SOLID;
                if label[k] == u16::MAX {
                    label[k] = si as u16;
                }
            }
        }
    }
    restore_thin_walls(&grid, &mut cells, &label, solids);
    if seam_tol_mm > 0.0 && groups.iter().any(|g| g.len() >= 2) {
        let reach = ((seam_tol_mm / dx).ceil() as usize).max(1);
        let mut group_of = vec![usize::MAX; solids.len()];
        for (gi, g) in groups.iter().enumerate() {
            if g.len() >= 2 {
                for &si in g {
                    if si < group_of.len() { group_of[si] = gi; }
                }
            }
        }
        let _ = seal_seams(&grid, &mut cells, &label, &group_of, reach, false);
    }
    let mut sealed_idx: Vec<u32> = vec![];
    let mut review_idx: Vec<u32> = vec![];
    if gap_all_mm > 0.0 {
        // aggressive: close any gap up to this size between two different parts, duct or not
        let reach = ((gap_all_mm / dx + 1e-6).floor() as usize).max(1);
        let (filled, review) = seal_seams(&grid, &mut cells, &label, &[], reach, true);
        sealed_idx = filled.into_iter().map(|i| i as u32).collect();
        review_idx = review.into_iter().map(|i| i as u32).collect();
    }
    close_diagonal_gaps(&grid, &mut cells);
    let [nx, ny, nz] = grid.dims;
    // sphere plugs (a leak the user sealed at its narrowest point)
    for (c, r) in plugs {
        let lo = [0, 1, 2].map(|i| (((c[i] - r - grid.origin[i]) / dx).floor().max(0.0)) as usize);
        let hi = [0, 1, 2].map(|i| ((((c[i] + r - grid.origin[i]) / dx).ceil()).max(0.0) as usize).min(grid.dims[i]));
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let i = grid.idx(x, y, z);
                    if cells[i] == FLUID && len(sub(grid.center(x, y, z), *c)) <= *r {
                        cells[i] = SOLID;
                    }
                }
            }
        }
    }
    // shut openings first, so nothing can flow through them
    for cap in blockers {
        mark_cap(&grid, &mut cells, cap, SOLID, true, 0.0);
    }
    let mut counts = vec![0usize; ports.len()];
    for (k, p) in ports.iter().enumerate() {
        counts[k] = mark_cap(&grid, &mut cells, &p.cap, PORT_BASE + k as u8, p.plug, if p.plug { p.collar_mm } else { 0.0 });
    }
    // Flood fill from port cells through fluid.
    let mut reach = vec![false; grid.len()];
    let mut stack: Vec<usize> = (0..grid.len()).filter(|&i| cells[i] >= PORT_BASE).collect();
    for &i in &stack {
        reach[i] = true;
    }
    while let Some(i) = stack.pop() {
        let [x, y, z] = grid.xyz(i);
        let mut visit = |xx: usize, yy: usize, zz: usize| {
            let j = grid.idx(xx, yy, zz);
            if !reach[j] && cells[j] == FLUID {
                reach[j] = true;
                stack.push(j);
            }
        };
        if x > 0 { visit(x - 1, y, z); }
        if x + 1 < nx { visit(x + 1, y, z); }
        if y > 0 { visit(x, y - 1, z); }
        if y + 1 < ny { visit(x, y + 1, z); }
        if z > 0 { visit(x, y, z - 1); }
        if z + 1 < nz { visit(x, y, z + 1); }
    }
    let mut removed = 0;
    for i in 0..grid.len() {
        if cells[i] == FLUID && !reach[i] {
            cells[i] = SOLID;
            removed += 1;
        }
    }
    Domain { grid, cells, port_cell_counts: counts, odd_columns, sealed_cells_removed: removed, repaired: vec![], sealed_idx, review_idx }
}

/// Mark the cells just inside a cap (planar or curved) with `code` (a port code, or SOLID to shut it);
/// with `plug`, the cells just in front are closed. Returns the number of cells marked.
fn mark_cap(grid: &Grid, cells: &mut [u8], cap: &PortCap, code: u8, plug: bool, collar_mm: f64) -> usize {
    if cap.is_surface() {
        return mark_surface_cap(grid, cells, cap, code, plug);
    }
    let dx = grid.dx;
    let mut count = 0;
    let poly = cap.polygon_2d();
    let holes = cap.holes_2d();
    let mut bb = Aabb::empty();
    for v in &cap.polygon {
        bb.grow(*v);
    }
    let bb = bb.expanded(2.0 * dx);
    let lo = [0, 1, 2].map(|i| (((bb.min[i] - grid.origin[i]) / dx).floor().max(0.0)) as usize);
    let hi = [0, 1, 2].map(|i| ((((bb.max[i] - grid.origin[i]) / dx).ceil()).max(0.0) as usize).min(grid.dims[i]));
    for z in lo[2]..hi[2] {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                let c = grid.center(x, y, z);
                if !cap.contains_in_opening(&poly, &holes, c) {
                    continue;
                }
                let d = cap.height(c);
                let i = grid.idx(x, y, z);
                if d >= -dx && d < 0.0 && cells[i] == FLUID {
                    cells[i] = code;
                    count += 1;
                } else if plug && d >= 0.0 && d < 2.0 * dx && cells[i] == FLUID {
                    cells[i] = SOLID;
                }
            }
        }
    }
    seal_rim(grid, cells, cap, &poly, plug || code == SOLID, 1.2 * dx, dx);
    if collar_mm > 0.0 {
        seal_rim(grid, cells, cap, &poly, true, collar_mm, collar_mm);
    }
    count
}

/// Bring back walls thinner than a cell. The inside/outside test samples only cell centres, so a sheet thinner than
/// one cell is missed wherever no centre falls inside it, and the duct is then open there. Wherever a surface passes
/// through a cell and that cell and both of its neighbours across the surface are fluid, the sheet was lost: that one
/// cell is closed. Walls that were captured (a solid cell on at least one side) are left exactly as they are.
fn restore_thin_walls(grid: &Grid, cells: &mut [u8], label: &[u16], solids: &[&TriMesh]) {
    if probing() || std::env::var_os("PG_NO_THIN_WALLS").is_some() {
        return;
    }
    let dx = grid.dx;
    let [nx, ny, nz] = grid.dims;
    let cell_of = |p: V3| -> Option<[usize; 3]> {
        let f = [(p[0] - grid.origin[0]) / dx, (p[1] - grid.origin[1]) / dx, (p[2] - grid.origin[2]) / dx];
        if f.iter().any(|v| *v < 0.0) { return None; }
        let c = [f[0] as usize, f[1] as usize, f[2] as usize];
        (c[0] < nx && c[1] < ny && c[2] < nz).then_some(c)
    };
    let mut add: Vec<usize> = vec![];
    for m in solids {
        for t in 0..m.triangles.len() {
            let [a, b, c] = m.tri(t);
            let n = cross(sub(b, a), sub(c, a));
            let area2 = len(n);
            if area2 < 1e-12 { continue; }
            let nn = scale(n, 1.0 / area2);
            let axis = (0..3).max_by(|&i, &j| nn[i].abs().partial_cmp(&nn[j].abs()).unwrap()).unwrap();
            // samples across the triangle, about two per cell
            let longest = len(sub(b, a)).max(len(sub(c, b))).max(len(sub(a, c)));
            let steps = ((longest / (0.5 * dx)).ceil() as usize).clamp(1, 400);
            // interior samples only: a sample on a triangle's edge can land in the cell beside the wall (outside the
            // part) and wrongly look like a lost sheet
            let (i0, n_div) = if steps >= 3 { (1usize, steps) } else { (0usize, 0usize) };
            let samples: Vec<(f64, f64)> = if n_div == 0 {
                vec![(1.0 / 3.0, 1.0 / 3.0)]
            } else {
                let mut v = vec![];
                for i in i0..n_div {
                    for j in i0..(n_div - i) {
                        v.push((i as f64 / n_div as f64, j as f64 / n_div as f64));
                    }
                }
                if v.is_empty() { v.push((1.0 / 3.0, 1.0 / 3.0)); }
                v
            };
            for (u, v) in samples {
                {
                    let w = 1.0 - u - v;
                    let p = [a[0] * w + b[0] * u + c[0] * v, a[1] * w + b[1] * u + c[1] * v, a[2] * w + b[2] * u + c[2] * v];
                    let Some(cc) = cell_of(p) else { continue };
                    let idx = grid.idx(cc[0], cc[1], cc[2]);
                    if cells[idx] != FLUID { continue; }
                    let mut lo = cc;
                    let mut hi = cc;
                    if cc[axis] == 0 || cc[axis] + 1 >= grid.dims[axis] { continue; }
                    lo[axis] -= 1;
                    hi[axis] += 1;
                    if cells[grid.idx(lo[0], lo[1], lo[2])] == FLUID && cells[grid.idx(hi[0], hi[1], hi[2])] == FLUID {
                        add.push(idx);
                    }
                }
            }
        }
    }
    let _ = label;
    if std::env::var_os("PG_DEBUG").is_some() {
        let mut ex: Vec<[usize; 3]> = add.iter().take(12).map(|&i| grid.xyz(i)).collect();
        ex.sort();
        eprintln!("thin-wall restore adds {} cells, e.g. {:?}", add.len(), ex);
    }
    for i in add {
        if cells[i] == FLUID { cells[i] = SOLID; }
    }
}

/// Make every thin wall face-connected. A wall one cell thick at a slant becomes a staircase of cells that touch
/// only at their corners: the air on either side then sees each other along a diagonal lattice link and flows
/// straight through. Wherever two solid cells touch diagonally across a pair of fluid cells, one of the fluid
/// cells is closed (the one that already touches more solid), which turns the corner contact into a face contact.
fn close_diagonal_gaps(grid: &Grid, cells: &mut [u8]) {
    if probing() || std::env::var_os("PG_NO_DIAG_CLOSE").is_some() {
        return;
    }
    let [nx, ny, nz] = grid.dims;
    let solid_neighbors = |cells: &[u8], q: [usize; 3]| -> usize {
        let mut n = 0;
        for (dx, dy, dz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (xx, yy, zz) = (q[0] as i32 + dx, q[1] as i32 + dy, q[2] as i32 + dz);
            if xx >= 0 && yy >= 0 && zz >= 0 && (xx as usize) < nx && (yy as usize) < ny && (zz as usize) < nz && cells[grid.idx(xx as usize, yy as usize, zz as usize)] == SOLID {
                n += 1;
            }
        }
        n
    };
    for _pass in 0..2 {
        let mut fill: Vec<usize> = vec![];
        for (u, v) in [(0usize, 1usize), (0, 2), (1, 2)] {
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let c = [x, y, z];
                        if c[u] + 1 >= grid.dims[u] || c[v] + 1 >= grid.dims[v] {
                            continue;
                        }
                        let at = |du: usize, dv: usize| -> ([usize; 3], usize) {
                            let mut q = c;
                            q[u] += du;
                            q[v] += dv;
                            (q, grid.idx(q[0], q[1], q[2]))
                        };
                        let (q00, i00) = at(0, 0);
                        let (q11, i11) = at(1, 1);
                        let (q10, i10) = at(1, 0);
                        let (q01, i01) = at(0, 1);
                        // solid on one diagonal and fluid on the other, either way round
                        let pairs = [((i00, i11), (i10, q10, i01, q01)), ((i10, i01), (i00, q00, i11, q11))];
                        for ((s1, s2), (f1, qf1, f2, qf2)) in pairs {
                            if cells[s1] == SOLID && cells[s2] == SOLID && cells[f1] == FLUID && cells[f2] == FLUID {
                                fill.push(if solid_neighbors(cells, qf1) >= solid_neighbors(cells, qf2) { f1 } else { f2 });
                            }
                        }
                    }
                }
            }
        }
        if fill.is_empty() {
            break;
        }
        for i in fill {
            if cells[i] == FLUID {
                cells[i] = SOLID;
            }
        }
    }
}

/// Close the sliver of fluid cells just outside the rim of a planar cap. The cap polygon is the rim of the
/// opening and a voxel grid never matches it exactly: without this, air slips between the port and the wall
/// (around a fan inlet, or beside an outlet) and leaves through a path that is not a port.
fn seal_rim(grid: &Grid, cells: &mut [u8], cap: &PortCap, poly: &[[f64; 2]], both_sides: bool, ring: f64, depth: f64) {
    if probing() || std::env::var_os("PG_NO_RIM_SEAL").is_some() {
        return;
    }
    let dx = grid.dx;
    let mut bb = Aabb::empty();
    for v in &cap.polygon {
        bb.grow(*v);
    }
    let bb = bb.expanded(ring + 2.0 * dx);
    let lo = [0, 1, 2].map(|i| (((bb.min[i] - grid.origin[i]) / dx).floor().max(0.0)) as usize);
    let hi = [0, 1, 2].map(|i| ((((bb.max[i] - grid.origin[i]) / dx).ceil()).max(0.0) as usize).min(grid.dims[i]));
    let front = if both_sides { depth.max(2.0 * dx) } else { 0.0 };
    for z in lo[2]..hi[2] {
        for y in lo[1]..hi[1] {
            for x in lo[0]..hi[0] {
                let i = grid.idx(x, y, z);
                if cells[i] != FLUID {
                    continue;
                }
                let c = grid.center(x, y, z);
                let d = cap.height(c);
                if d < -depth || d >= front {
                    continue;
                }
                let q = cap.project2d(c);
                if point_in_polygon(poly, q) {
                    continue;
                }
                // distance to the rim
                let mut best = f64::MAX;
                for k in 0..poly.len() {
                    let (a, b) = (poly[k], poly[(k + 1) % poly.len()]);
                    let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
                    let l2 = ex * ex + ey * ey;
                    let t = if l2 > 0.0 { (((q[0] - a[0]) * ex + (q[1] - a[1]) * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
                    let (px, py) = (a[0] + t * ex - q[0], a[1] + t * ey - q[1]);
                    best = best.min((px * px + py * py).sqrt());
                }
                if best <= ring {
                    cells[i] = SOLID;
                }
            }
        }
    }
}

/// Mark the cells just inside a non-planar cap: fluid cells within one cell behind each cap triangle
/// become port cells; for inlets the cells just in front are closed. Returns the number of port cells.
fn mark_surface_cap(grid: &Grid, cells: &mut [u8], cap: &PortCap, code: u8, plug: bool) -> usize {
    let dx = grid.dx;
    let mut count = 0;
    for pass in 0..2 {
        if pass == 1 && !plug {
            break;
        }
        for t in &cap.triangles {
            let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
            let cr = cross(e1, e2);
            let area2 = len(cr);
            if area2 < 1e-9 {
                continue;
            }
            let n = scale(cr, 1.0 / area2);
            let mut bb = Aabb::empty();
            for v in t {
                bb.grow(*v);
            }
            let bb = bb.expanded(2.0 * dx);
            let lo = [0, 1, 2].map(|i| (((bb.min[i] - grid.origin[i]) / dx).floor().max(0.0)) as usize);
            let hi = [0, 1, 2].map(|i| ((((bb.max[i] - grid.origin[i]) / dx).ceil()).max(0.0) as usize).min(grid.dims[i]));
            for z in lo[2]..hi[2] {
                for y in lo[1]..hi[1] {
                    for x in lo[0]..hi[0] {
                        let c = grid.center(x, y, z);
                        let d = dot(sub(c, t[0]), n);
                        if d < -dx || d >= 2.0 * dx {
                            continue;
                        }
                        // barycentric test of the projection onto the triangle's plane
                        let q = sub(sub(c, t[0]), scale(n, d));
                        let (d11, d12, d22) = (dot(e1, e1), dot(e1, e2), dot(e2, e2));
                        let (d1, d2) = (dot(q, e1), dot(q, e2));
                        let det = d11 * d22 - d12 * d12;
                        if det.abs() < 1e-18 {
                            continue;
                        }
                        let u = (d22 * d1 - d12 * d2) / det;
                        let v = (d11 * d2 - d12 * d1) / det;
                        if u < -0.02 || v < -0.02 || u + v > 1.02 {
                            continue;
                        }
                        let i = grid.idx(x, y, z);
                        if pass == 0 && d < 0.0 && cells[i] == FLUID {
                            cells[i] = code;
                            count += 1;
                        } else if pass == 1 && d >= 0.0 && cells[i] == FLUID {
                            cells[i] = SOLID;
                        }
                    }
                }
            }
        }
    }
    count
}

/// Result of [`leak_check`].
#[derive(Clone, Debug)]
pub struct Leak {
    /// Domain-boundary cells reached from the inlets without passing through any port.
    pub boundary_cells: usize,
    /// Approximate place (mm) where the escaping air path leaves the duct region.
    pub exit_point: V3,
    /// Narrowest point (mm) of the escape path, and the width (mm) of the gap there.
    pub throat: V3,
    pub throat_width: f64,
}

/// Is the duct sealed except at its ports? Flood the air from the inlet cells, treating every port cell as a wall.
/// If that air reaches the edge of the domain, it can leave through something that is not a port: a gap between
/// parts, or an opening that was not defined as a port.
pub fn leak_check(domain: &Domain, inlet_codes: &[u8]) -> Option<Leak> {
    leak_check_ignoring(domain, inlet_codes, &[])
}

/// Like [`leak_check`], but air that escapes inside one of the `ignore` spheres (centre mm, radius mm) does not
/// count: those are the surroundings of the outlets, where air is meant to leave.
pub fn leak_check_ignoring(domain: &Domain, inlet_codes: &[u8], ignore: &[(V3, f64)]) -> Option<Leak> {
    let g = &domain.grid;
    let [nx, ny, nz] = g.dims;
    let n = g.len();
    let mut seen = vec![false; n];
    for (c, r) in ignore {
        let lo = [0, 1, 2].map(|i| (((c[i] - r - g.origin[i]) / g.dx).floor().max(0.0)) as usize);
        let hi = [0, 1, 2].map(|i| ((((c[i] + r - g.origin[i]) / g.dx).ceil()).max(0.0) as usize).min(g.dims[i]));
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    if len(sub(g.center(x, y, z), *c)) <= *r { seen[g.idx(x, y, z)] = true; } // never entered
                }
            }
        }
    }
    let mut parent = vec![u32::MAX; n];
    let mut stack: std::collections::VecDeque<usize> = std::collections::VecDeque::new();
    // start from the fluid cells next to an inlet port cell
    for i in 0..n {
        if inlet_codes.contains(&domain.cells[i]) {
            let [x, y, z] = g.xyz(i);
            let mut push = |xx: usize, yy: usize, zz: usize| {
                let j = g.idx(xx, yy, zz);
                if !seen[j] && domain.cells[j] == FLUID {
                    seen[j] = true;
                    stack.push_back(j);
                }
            };
            if x > 0 { push(x - 1, y, z); }
            if x + 1 < nx { push(x + 1, y, z); }
            if y > 0 { push(x, y - 1, z); }
            if y + 1 < ny { push(x, y + 1, z); }
            if z > 0 { push(x, y, z - 1); }
            if z + 1 < nz { push(x, y, z + 1); }
        }
    }
    let on_edge = |x: usize, y: usize, z: usize| x == 0 || y == 0 || z == 0 || x + 1 == nx || y + 1 == ny || z + 1 == nz;
    let mut edge_hits = vec![];
    while let Some(i) = stack.pop_front() {
        let [x, y, z] = g.xyz(i);
        if on_edge(x, y, z) {
            edge_hits.push(i);
            continue;
        }
        for (dx, dy, dz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (xx, yy, zz) = ((x as i32 + dx) as usize, (y as i32 + dy) as usize, (z as i32 + dz) as usize);
            let j = g.idx(xx, yy, zz);
            if !seen[j] && domain.cells[j] == FLUID {
                seen[j] = true;
                parent[j] = i as u32;
                stack.push_back(j);
            }
        }
    }
    if edge_hits.is_empty() {
        return None;
    }
    // walk back from one escape cell to the first cell that lies inside the solid bodies' extent
    let mut solid_lo = [usize::MAX; 3];
    let mut solid_hi = [0usize; 3];
    for i in 0..n {
        if domain.cells[i] == SOLID {
            let c = g.xyz(i);
            for k in 0..3 { solid_lo[k] = solid_lo[k].min(c[k]); solid_hi[k] = solid_hi[k].max(c[k]); }
        }
    }
    let mut cur = edge_hits[0];
    let mut exit = cur;
    loop {
        let c = g.xyz(cur);
        if (0..3).all(|k| c[k] >= solid_lo[k] && c[k] <= solid_hi[k]) {
            exit = cur;
            break;
        }
        if parent[cur] == u32::MAX { break; }
        cur = parent[cur] as usize;
    }
    let [x, y, z] = g.xyz(exit);
    // narrowest point of the whole escape path (skipping the first stretch next to the inlet)
    let mut path = vec![edge_hits[0]];
    let mut cur = edge_hits[0];
    while parent[cur] != u32::MAX {
        cur = parent[cur] as usize;
        path.push(cur);
    }
    path.reverse();
    let skip = (path.len() / 8).max(3).min(path.len().saturating_sub(1));
    let clearance = |i: usize| -> f64 {
        let [cx, cy, cz] = g.xyz(i);
        let r = 8i32;
        let mut best = f64::MAX;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let (xx, yy, zz) = (cx as i32 + dx, cy as i32 + dy, cz as i32 + dz);
                    if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 { continue; }
                    if domain.cells[g.idx(xx as usize, yy as usize, zz as usize)] != FLUID {
                        best = best.min(((dx * dx + dy * dy + dz * dz) as f64).sqrt());
                    }
                }
            }
        }
        if best == f64::MAX { r as f64 } else { best }
    };
    let mut throat = path[path.len() - 1];
    let mut tw = f64::MAX;
    for &i in &path[skip..] {
        let c = g.xyz(i);
        if !(0..3).all(|k| c[k] >= solid_lo[k] && c[k] <= solid_hi[k]) { continue; }
        let w = clearance(i);
        if w <= tw { tw = w; throat = i; }
    }
    let [tx, ty, tz] = g.xyz(throat);
    Some(Leak { boundary_cells: edge_hits.len(), exit_point: g.center(x, y, z), throat: g.center(tx, ty, tz), throat_width: if tw == f64::MAX { 2.0 * g.dx } else { 2.0 * tw * g.dx } })
}

/// Repair small leaks: while air from the inlets can reach the edge of the domain without passing a port, close
/// the narrowest point of its escape path with a small ball of solid. Stops when sealed, after `max_repairs`, or
/// when the narrowest point is wider than `max_width_mm` (that is a real opening, not a gap: it is left for the
/// user). Returns the places that were closed.
pub fn auto_seal_leaks(domain: &mut Domain, inlet_codes: &[u8], ignore: &[(V3, f64)], max_repairs: usize, max_width_mm: f64) -> Vec<V3> {
    let mut done = vec![];
    for _ in 0..max_repairs {
        let Some(leak) = leak_check_ignoring(domain, inlet_codes, ignore) else { break };
        if leak.throat_width > max_width_mm {
            break;
        }
        let g = &domain.grid;
        let r = 0.5 * leak.throat_width + 0.75 * g.dx;
        let c = leak.throat;
        let lo = [0, 1, 2].map(|i| (((c[i] - r - g.origin[i]) / g.dx).floor().max(0.0)) as usize);
        let hi = [0, 1, 2].map(|i| ((((c[i] + r - g.origin[i]) / g.dx).ceil()).max(0.0) as usize).min(g.dims[i]));
        let mut filled = 0;
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let i = g.idx(x, y, z);
                    if domain.cells[i] == FLUID && len(sub(g.center(x, y, z), c)) <= r {
                        domain.cells[i] = SOLID;
                        domain.sealed_idx.push(i as u32);
                        filled += 1;
                    }
                }
            }
        }
        if filled == 0 {
            break;
        }
        done.push(c);
    }
    done
}

/// Keep only the sealed cells that matter to the duct: those connected (through other sealed cells) to the air
/// the inlets can reach. Thin gaps between unrelated parts (a motor, a rail) stay closed but are not reported.
pub fn keep_relevant_seals(domain: &mut Domain, inlet_codes: &[u8]) {
    if domain.sealed_idx.is_empty() {
        return;
    }
    let g = &domain.grid;
    let [nx, ny, nz] = g.dims;
    let n = g.len();
    // air reachable from the inlets (ports are walls)
    let mut reach = vec![false; n];
    let mut stack: Vec<usize> = vec![];
    for i in 0..n {
        if inlet_codes.contains(&domain.cells[i]) {
            let [x, y, z] = g.xyz(i);
            for (dx, dy, dz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
                let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz { continue; }
                let j = g.idx(xx as usize, yy as usize, zz as usize);
                if !reach[j] && domain.cells[j] == FLUID { reach[j] = true; stack.push(j); }
            }
        }
    }
    while let Some(i) = stack.pop() {
        let [x, y, z] = g.xyz(i);
        for (dx, dy, dz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz { continue; }
            let j = g.idx(xx as usize, yy as usize, zz as usize);
            if !reach[j] && domain.cells[j] == FLUID { reach[j] = true; stack.push(j); }
        }
    }
    let mut in_s = vec![false; n];
    for &i in &domain.sealed_idx { in_s[i as usize] = true; }
    let mut keep = vec![false; n];
    let mut q: Vec<usize> = vec![];
    let touches_reach = |i: usize| -> bool {
        let [x, y, z] = g.xyz(i);
        for (dx, dy, dz) in [(-1i32, 0i32, 0i32), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)] {
            let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz { continue; }
            if reach[g.idx(xx as usize, yy as usize, zz as usize)] { return true; }
        }
        false
    };
    for &i in &domain.sealed_idx {
        if touches_reach(i as usize) { keep[i as usize] = true; q.push(i as usize); }
    }
    while let Some(i) = q.pop() {
        let [x, y, z] = g.xyz(i);
        for dz in -1i32..=1 { for dy in -1i32..=1 { for dx in -1i32..=1 {
            let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
            if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz { continue; }
            let j = g.idx(xx as usize, yy as usize, zz as usize);
            if in_s[j] && !keep[j] { keep[j] = true; q.push(j); }
        } } }
    }
    domain.sealed_idx.retain(|&i| keep[i as usize]);
}

/// Fill fluid cells that sit between two different parts of the same duct (see [`build_domain_ext`]).
/// A thin gap longer than this (mm), or covering more than [`INTENTIONAL_AREA_MM2`], is taken to be intentional.
const INTENTIONAL_LENGTH_MM: f64 = 12.0;
const INTENTIONAL_AREA_MM2: f64 = 60.0;

fn seal_seams(grid: &Grid, cells: &mut [u8], label: &[u16], group_of: &[usize], reach: usize, any_pair: bool) -> (Vec<usize>, Vec<usize>) {
    let [nx, ny, nz] = grid.dims;
    let mut fill: Vec<usize> = vec![];
    let step_dirs: [(i32, i32, i32); 3] = [(1, 0, 0), (0, 1, 0), (0, 0, 1)];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let i = grid.idx(x, y, z);
                if cells[i] != FLUID {
                    continue;
                }
                'axes: for (dx_, dy_, dz_) in step_dirs {
                    let find = |sign: i32| -> Option<(u16, usize)> {
                        for k in 1..=reach as i32 {
                            let (xx, yy, zz) = (x as i32 + sign * k * dx_, y as i32 + sign * k * dy_, z as i32 + sign * k * dz_);
                            if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 {
                                return None;
                            }
                            let j = grid.idx(xx as usize, yy as usize, zz as usize);
                            if cells[j] == SOLID {
                                return Some((label[j], k as usize));
                            }
                            if cells[j] != FLUID {
                                return None;
                            }
                        }
                        None
                    };
                    if let (Some((a, da)), Some((b, db))) = (find(-1), find(1)) {
                        // the whole gap between the two surfaces must be thin, not just this cell's distance to each side
                        if da + db - 1 > reach {
                            continue;
                        }
                        if any_pair || (a != b && a != u16::MAX && b != u16::MAX && group_of[a as usize] != usize::MAX && group_of[a as usize] == group_of[b as usize]) {
                            fill.push(i);
                            break 'axes;
                        }
                    }
                }
            }
        }
    }
    // A gap this thin that runs over a long distance or a large area is an intentional opening (a slot, a vent),
    // not a CAD clearance: leave it open and hand it back for review.
    let mut review: Vec<usize> = vec![];
    if any_pair && !fill.is_empty() {
        let n = grid.len();
        let mut in_fill = vec![false; n];
        for &i in &fill { in_fill[i] = true; }
        let mut done = vec![false; n];
        let mut keep: Vec<usize> = vec![];
        for &s in &fill {
            if done[s] { continue; }
            let mut comp = vec![s];
            done[s] = true;
            let mut head = 0;
            let (mut lo, mut hi) = (grid.xyz(s), grid.xyz(s));
            while head < comp.len() {
                let [x, y, z] = grid.xyz(comp[head]);
                head += 1;
                for k in 0..3 { let c = [x, y, z][k]; lo[k] = lo[k].min(c); hi[k] = hi[k].max(c); }
                for dz in -1i32..=1 { for dy in -1i32..=1 { for dx in -1i32..=1 {
                    let (xx, yy, zz) = (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                    if xx < 0 || yy < 0 || zz < 0 || xx as usize >= nx || yy as usize >= ny || zz as usize >= nz { continue; }
                    let j = grid.idx(xx as usize, yy as usize, zz as usize);
                    if in_fill[j] && !done[j] { done[j] = true; comp.push(j); }
                } } }
            }
            let mut ext: Vec<f64> = (0..3).map(|k| (hi[k] - lo[k] + 1) as f64 * grid.dx).collect();
            ext.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let (longest, second) = (ext[0], ext[1]);
            if longest > INTENTIONAL_LENGTH_MM || longest * second > INTENTIONAL_AREA_MM2 {
                review.extend(comp);
            } else {
                keep.extend(comp);
            }
        }
        fill = keep;
    }
    for &i in &fill {
        cells[i] = SOLID;
    }
    (fill, review)
}

/// For each cap, the volume (mm^3) of the connected air just behind it, with every cap plugged. Air behind a
/// real opening is the whole duct interior (shared by all its openings); air behind a pocket or recess is a small
/// closed cavity. Compare the values: a pocket scores a tiny fraction of the largest.
pub fn behind_volumes(solids: &[&TriMesh], caps: &[PortCap], bounds: &Aabb, dx: f64) -> Vec<f64> {
    let n = caps.len().min(250);
    let ports: Vec<PortSpec> = caps.iter().take(n).map(|c| PortSpec { cap: c.clone(), plug: true, collar_mm: 0.0 }).collect();
    PROBE.with(|p| p.set(true));
    let d = build_domain(solids, &ports, bounds, dx);
    PROBE.with(|p| p.set(false));
    let g = &d.grid;
    let [nx, ny, nz] = g.dims;
    // label connected components of all non-solid cells (port cells belong to the air behind their cap)
    let mut comp = vec![u32::MAX; g.len()];
    let mut sizes: Vec<u64> = vec![];
    for s in 0..g.len() {
        if d.cells[s] == SOLID || comp[s] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        let mut count = 0u64;
        let mut stack = vec![s];
        comp[s] = id;
        while let Some(i) = stack.pop() {
            count += 1;
            let [x, y, z] = g.xyz(i);
            let mut visit = |xx: usize, yy: usize, zz: usize| {
                let j = g.idx(xx, yy, zz);
                if comp[j] == u32::MAX && d.cells[j] != SOLID {
                    comp[j] = id;
                    stack.push(j);
                }
            };
            if x > 0 { visit(x - 1, y, z); }
            if x + 1 < nx { visit(x + 1, y, z); }
            if y > 0 { visit(x, y - 1, z); }
            if y + 1 < ny { visit(x, y + 1, z); }
            if z > 0 { visit(x, y, z - 1); }
            if z + 1 < nz { visit(x, y, z + 1); }
        }
        sizes.push(count);
    }
    let cell_vol = dx * dx * dx;
    // NaN = the cap could not be resolved on this grid (no cells behind it): unknown, not zero
    let mut out = vec![f64::NAN; caps.len()];
    for k in 0..n {
        let code = PORT_BASE + k as u8;
        if let Some(i) = (0..g.len()).find(|&i| d.cells[i] == code) {
            out[k] = sizes[comp[i] as usize] as f64 * cell_vol;
        }
    }
    out
}

#[cfg(test)]
mod diag_tests {
    use super::*;

    /// A slanted wall one cell thick (corner contacts only) must become face-connected.
    #[test]
    fn diagonal_wall_is_closed() {
        let grid = Grid { origin: [0.0; 3], dx: 1.0, dims: [8, 8, 1] };
        let mut cells = vec![FLUID; grid.len()];
        for i in 0..8 {
            cells[grid.idx(i, i, 0)] = SOLID; // the diagonal
        }
        // before: cells (3,4) and (4,3) are fluid, diagonal neighbours on opposite sides of the wall
        close_diagonal_gaps(&grid, &mut cells);
        // after: no pair of solid cells is left touching only at corners across two fluid cells
        for y in 0..7 {
            for x in 0..7 {
                let (a, b, c, d) = (cells[grid.idx(x, y, 0)], cells[grid.idx(x + 1, y, 0)], cells[grid.idx(x, y + 1, 0)], cells[grid.idx(x + 1, y + 1, 0)]);
                assert!(!(a == SOLID && d == SOLID && b == FLUID && c == FLUID), "open corner at {x},{y}");
                assert!(!(b == SOLID && c == SOLID && a == FLUID && d == FLUID), "open corner at {x},{y}");
            }
        }
        // and it only added a few cells
        assert!(cells.iter().filter(|&&c| c == SOLID).count() < 8 + 8);
    }

    /// A gap is closed only when the WHOLE gap is thin: a channel a few cells wide must stay open, not get a plug
    /// down its middle.
    #[test]
    fn only_thin_gaps_are_closed() {
        for (gap, closed) in [(1usize, true), (2, true), (3, true), (4, false), (5, false)] {
            let grid = Grid { origin: [0.0; 3], dx: 1.0, dims: [gap + 2, 3, 3] };
            let mut cells = vec![FLUID; grid.len()];
            for y in 0..3 {
                for z in 0..3 {
                    cells[grid.idx(0, y, z)] = SOLID;
                    cells[grid.idx(gap + 1, y, z)] = SOLID;
                }
            }
            let label = vec![0u16; grid.len()];
            let _ = seal_seams(&grid, &mut cells, &label, &[], 3, true);
            let filled = (1..=gap).all(|x| cells[grid.idx(x, 1, 1)] == SOLID);
            let untouched = (1..=gap).all(|x| cells[grid.idx(x, 1, 1)] == FLUID);
            assert!(if closed { filled } else { untouched }, "gap of {gap} cells: closed = {closed}");
        }
    }

    /// A thin gap that runs a long way is an intentional opening: it must be left open and handed back for review.
    #[test]
    fn long_thin_gaps_are_left_for_review() {
        let make = |len: usize| {
            let grid = Grid { origin: [0.0; 3], dx: 1.0, dims: [len, len.min(8), 6] };
            let mut cells = vec![FLUID; grid.len()];
            for x in 0..grid.dims[0] { for y in 0..grid.dims[1] {
                cells[grid.idx(x, y, 0)] = SOLID; cells[grid.idx(x, y, 1)] = SOLID;
                cells[grid.idx(x, y, 4)] = SOLID; cells[grid.idx(x, y, 5)] = SOLID;
            } }
            (grid, cells)
        };
        // 6 x 6 mm patch, 2 mm gap: a clearance, sealed
        let (grid, mut cells) = make(6);
        let label = vec![0u16; grid.len()];
        let (filled, review) = seal_seams(&grid, &mut cells, &label, &[], 3, true);
        assert!(!filled.is_empty() && review.is_empty());
        // 30 mm long: intentional, left open
        let (grid, mut cells) = make(30);
        let label = vec![0u16; grid.len()];
        let (filled, review) = seal_seams(&grid, &mut cells, &label, &[], 3, true);
        assert!(filled.is_empty() && !review.is_empty());
        assert!(cells[grid.idx(15, 3, 2)] == FLUID, "the long gap must stay open");
    }

    /// A sheet thinner than a cell must never vanish, wherever it sits relative to the cell centres.
    #[test]
    fn thin_sheets_survive_voxelization() {
        use peregrine_core::shapes;
        let mut lost_without = 0;
        for k in 0..10 {
            let off = 0.05 + 0.1 * k as f64;
            let sheet = shapes::solid_box([off, -5.0, -5.0], [off + 0.25, 5.0, 5.0]);
            let mut bb = sheet.bounds();
            bb.grow([-3.0, -6.0, -6.0]);
            bb.grow([4.0, 6.0, 6.0]);
            let grid = Grid::from_bounds(&bb, 1.0);
            let (inside, _) = voxelize_inside(&sheet, &grid);
            let mut cells: Vec<u8> = inside.iter().map(|&i| if i { SOLID } else { FLUID }).collect();
            let row = |cells: &[u8]| (0..grid.dims[0]).any(|x| cells[grid.idx(x, grid.dims[1] / 2, grid.dims[2] / 2)] == SOLID);
            if !row(&cells) { lost_without += 1; }
            let label = vec![0u16; grid.len()];
            restore_thin_walls(&grid, &mut cells, &label, &[&sheet]);
            assert!(row(&cells), "sheet at offset {off} vanished");
        }
        assert!(lost_without > 0, "the test must include sheets that the plain test loses");
    }
}
