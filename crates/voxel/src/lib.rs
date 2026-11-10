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
    assert!(ports.len() <= (255 - PORT_BASE as usize), "too many ports");
    let grid = Grid::from_bounds(bounds, dx);
    let mut cells = vec![FLUID; grid.len()];
    let mut odd_columns = 0;
    for m in solids {
        let (inside, odd) = voxelize_inside(m, &grid);
        odd_columns += odd;
        for (c, i) in cells.iter_mut().zip(inside) {
            if i {
                *c = SOLID;
            }
        }
    }
    let [nx, ny, nz] = grid.dims;
    let mut counts = vec![0usize; ports.len()];
    for (k, p) in ports.iter().enumerate() {
        let poly = p.cap.polygon_2d();
        let mut bb = Aabb::empty();
        for v in &p.cap.polygon {
            bb.grow(*v);
        }
        let bb = bb.expanded(2.0 * dx);
        let lo = [0, 1, 2].map(|i| (((bb.min[i] - grid.origin[i]) / dx).floor().max(0.0)) as usize);
        let hi = [0, 1, 2].map(|i| ((((bb.max[i] - grid.origin[i]) / dx).ceil()).max(0.0) as usize).min(grid.dims[i]));
        for z in lo[2]..hi[2] {
            for y in lo[1]..hi[1] {
                for x in lo[0]..hi[0] {
                    let c = grid.center(x, y, z);
                    if !p.cap.contains_projected(&poly, c) {
                        continue;
                    }
                    let d = p.cap.height(c);
                    let i = grid.idx(x, y, z);
                    if d >= -dx && d < 0.0 && cells[i] == FLUID {
                        cells[i] = PORT_BASE + k as u8;
                        counts[k] += 1;
                    } else if p.plug && d >= 0.0 && d < 2.0 * dx && cells[i] == FLUID {
                        cells[i] = SOLID;
                    }
                }
            }
        }
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
