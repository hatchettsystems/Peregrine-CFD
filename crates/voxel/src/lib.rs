//! Cartesian voxelisation: solids by ray parity, port caps, fluid-region flood fill.
use peregrine_core::port_cap::PortCap;
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

#[derive(Clone, Debug)]
pub struct PortSpec {
    pub cap: PortCap,
    /// Close the volume behind the cap (inlets, where a fan or wall sits).
    pub plug: bool,
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
}

pub fn build_domain(solids: &[&TriMesh], ports: &[PortSpec], bounds: &Aabb, dx: f64) -> Domain {
    build_domain_ext(solids, &[], 0.0, ports, bounds, dx)
}

/// Like [`build_domain`], with seam sealing. `groups` lists, per duct, the indices (into `solids`) of the parts
/// that form that duct. A fluid cell squeezed between two DIFFERENT parts of the same duct, at most
/// `seam_tol_mm` apart along an axis, is filled, so small gaps between parts do not leak.
pub fn build_domain_ext(solids: &[&TriMesh], groups: &[Vec<usize>], seam_tol_mm: f64, ports: &[PortSpec], bounds: &Aabb, dx: f64) -> Domain {
    build_domain_full(solids, groups, seam_tol_mm, ports, &[], bounds, dx)
}

/// Like [`build_domain_ext`], and every cap in `blockers` is filled solid (a shut opening: screw hole, vent...).
pub fn build_domain_full(solids: &[&TriMesh], groups: &[Vec<usize>], seam_tol_mm: f64, ports: &[PortSpec], blockers: &[PortCap], bounds: &Aabb, dx: f64) -> Domain {
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
        seal_seams(&grid, &mut cells, &label, &group_of, reach);
    }
    let [nx, ny, nz] = grid.dims;
    // shut openings first, so nothing can flow through them
    for cap in blockers {
        mark_cap(&grid, &mut cells, cap, SOLID, true);
    }
    let mut counts = vec![0usize; ports.len()];
    for (k, p) in ports.iter().enumerate() {
        counts[k] = mark_cap(&grid, &mut cells, &p.cap, PORT_BASE + k as u8, p.plug);
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
    Domain { grid, cells, port_cell_counts: counts, odd_columns, sealed_cells_removed: removed }
}

/// Mark the cells just inside a cap (planar or curved) with `code` (a port code, or SOLID to shut it);
/// with `plug`, the cells just in front are closed. Returns the number of cells marked.
fn mark_cap(grid: &Grid, cells: &mut [u8], cap: &PortCap, code: u8, plug: bool) -> usize {
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
    count
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
}

/// Is the duct sealed except at its ports? Flood the air from the inlet cells, treating every port cell as a wall.
/// If that air reaches the edge of the domain, it can leave through something that is not a port: a gap between
/// parts, or an opening that was not defined as a port.
pub fn leak_check(domain: &Domain, inlet_codes: &[u8]) -> Option<Leak> {
    let g = &domain.grid;
    let [nx, ny, nz] = g.dims;
    let n = g.len();
    let mut seen = vec![false; n];
    let mut parent = vec![u32::MAX; n];
    let mut stack: Vec<usize> = vec![];
    // start from the fluid cells next to an inlet port cell
    for i in 0..n {
        if inlet_codes.contains(&domain.cells[i]) {
            let [x, y, z] = g.xyz(i);
            let mut push = |xx: usize, yy: usize, zz: usize| {
                let j = g.idx(xx, yy, zz);
                if !seen[j] && domain.cells[j] == FLUID {
                    seen[j] = true;
                    stack.push(j);
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
    while let Some(i) = stack.pop() {
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
                stack.push(j);
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
    Some(Leak { boundary_cells: edge_hits.len(), exit_point: g.center(x, y, z) })
}

/// Fill fluid cells that sit between two different parts of the same duct (see [`build_domain_ext`]).
fn seal_seams(grid: &Grid, cells: &mut [u8], label: &[u16], group_of: &[usize], reach: usize) {
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
                    let find = |sign: i32| -> Option<u16> {
                        for k in 1..=reach as i32 {
                            let (xx, yy, zz) = (x as i32 + sign * k * dx_, y as i32 + sign * k * dy_, z as i32 + sign * k * dz_);
                            if xx < 0 || yy < 0 || zz < 0 || xx >= nx as i32 || yy >= ny as i32 || zz >= nz as i32 {
                                return None;
                            }
                            let j = grid.idx(xx as usize, yy as usize, zz as usize);
                            if cells[j] == SOLID {
                                return Some(label[j]);
                            }
                            if cells[j] != FLUID {
                                return None;
                            }
                        }
                        None
                    };
                    if let (Some(a), Some(b)) = (find(-1), find(1)) {
                        if a != b && a != u16::MAX && b != u16::MAX && group_of[a as usize] != usize::MAX && group_of[a as usize] == group_of[b as usize] {
                            fill.push(i);
                            break 'axes;
                        }
                    }
                }
            }
        }
    }
    for i in fill {
        cells[i] = SOLID;
    }
}

/// For each cap, the volume (mm^3) of the connected air just behind it, with every cap plugged. Air behind a
/// real opening is the whole duct interior (shared by all its openings); air behind a pocket or recess is a small
/// closed cavity. Compare the values: a pocket scores a tiny fraction of the largest.
pub fn behind_volumes(solids: &[&TriMesh], caps: &[PortCap], bounds: &Aabb, dx: f64) -> Vec<f64> {
    let n = caps.len().min(250);
    let ports: Vec<PortSpec> = caps.iter().take(n).map(|c| PortSpec { cap: c.clone(), plug: true }).collect();
    let d = build_domain(solids, &ports, bounds, dx);
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
