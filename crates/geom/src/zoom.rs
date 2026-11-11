//! Zoom runs: re-simulate a small box of the duct (for example the bottom plate with its slots) at a much finer
//! cell size, fed by the flow a coarse run of the whole duct measured where its channels cross the top of the box.
use anyhow::{bail, Context, Result};
use peregrine_core::PortCap;
use peregrine_project::{Port, PortKind, Project};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// A finished coarse run: the solver grid with the velocity (m/s) in every cell.
pub struct CoarseField {
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub dx: f64,
    /// 1 = solid, 0 = fluid, 2.. = port cells.
    pub cells: Vec<u8>,
    pub vel: Vec<[f32; 3]>,
}

/// Where and how finely to zoom. Lengths in mm.
#[derive(Clone, Debug)]
pub struct ZoomSpec {
    /// The top of the box: channels are cut here and fed with the coarse flow.
    pub plane_z: f64,
    pub z_min: f64,
    pub x_range: [f64; 2],
    pub y_range: [f64; 2],
    pub fine_dx: f64,
}

/// Read the `result.vtk` written by a run.
pub fn read_vtk_field(path: &Path) -> Result<CoarseField> {
    let b = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let head = String::from_utf8_lossy(&b[..b.len().min(800)]).to_string();
    let grab = |key: &str| -> Result<Vec<f64>> {
        let line = head.lines().find(|l| l.starts_with(key)).with_context(|| format!("no {key} in the vtk header"))?;
        Ok(line.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect())
    };
    let d = grab("DIMENSIONS")?;
    let o = grab("ORIGIN")?;
    let sp = grab("SPACING")?;
    if d.len() < 3 || o.len() < 3 || sp.is_empty() { bail!("unreadable vtk header"); }
    let (nx, ny, nz) = (d[0] as usize - 1, d[1] as usize - 1, d[2] as usize - 1);
    let n = nx * ny * nz;
    let find = |from: usize, pat: &[u8]| -> Result<usize> {
        b[from..].windows(pat.len()).position(|w| w == pat).map(|p| p + from).with_context(|| format!("missing {}", String::from_utf8_lossy(pat)))
    };
    let mut pos = find(0, b"CELL_DATA")?;
    pos = find(pos, b"\n")? + 1;
    let mut scalar = |pos: &mut usize| -> Result<Vec<f32>> {
        *pos = find(*pos, b"\n")? + 1; // SCALARS
        *pos = find(*pos, b"\n")? + 1; // LOOKUP_TABLE
        let v: Vec<f32> = (0..n).map(|i| f32::from_be_bytes([b[*pos + 4 * i], b[*pos + 4 * i + 1], b[*pos + 4 * i + 2], b[*pos + 4 * i + 3]])).collect();
        *pos += 4 * n + 1;
        Ok(v)
    };
    let cell = scalar(&mut pos)?;
    let _speed = scalar(&mut pos)?;
    let _pressure = scalar(&mut pos)?;
    pos = find(pos, b"\n")? + 1;
    let vel: Vec<[f32; 3]> = (0..n)
        .map(|i| {
            let g = |a: usize| { let q = pos + 12 * i + 4 * a; f32::from_be_bytes([b[q], b[q + 1], b[q + 2], b[q + 3]]) };
            [g(0), g(1), g(2)]
        })
        .collect();
    Ok(CoarseField { dims: [nx, ny, nz], origin: [o[0], o[1], o[2]], dx: sp[0], cells: cell.iter().map(|&c| c as u8).collect(), vel })
}

/// Boundary of a set of cells as one polygon (the longest loop), in mm.
fn outline(cells: &HashSet<(i64, i64)>, origin: [f64; 2], dx: f64) -> Vec<[f64; 2]> {
    let mut next: HashMap<(i64, i64), Vec<(i64, i64)>> = HashMap::new();
    for &(x, y) in cells {
        let has = |a: i64, b: i64| cells.contains(&(a, b));
        if !has(x, y - 1) { next.entry((x, y)).or_default().push((x + 1, y)); }
        if !has(x + 1, y) { next.entry((x + 1, y)).or_default().push((x + 1, y + 1)); }
        if !has(x, y + 1) { next.entry((x + 1, y + 1)).or_default().push((x, y + 1)); }
        if !has(x - 1, y) { next.entry((x, y + 1)).or_default().push((x, y)); }
    }
    let mut best: Vec<(i64, i64)> = vec![];
    let mut starts: Vec<(i64, i64)> = next.keys().copied().collect();
    starts.sort();
    for s in starts {
        while next.get(&s).map_or(false, |v| !v.is_empty()) {
            let mut lp = vec![s];
            let mut cur = s;
            loop {
                let Some(nx) = next.get_mut(&cur).and_then(|v| v.pop()) else { break };
                if nx == s { break; }
                lp.push(nx);
                cur = nx;
                if lp.len() > 100_000 { break; }
            }
            if lp.len() > best.len() { best = lp; }
        }
    }
    // drop collinear vertices
    let n = best.len();
    let mut out: Vec<(i64, i64)> = vec![];
    for i in 0..n {
        let (a, b, c) = (best[(i + n - 1) % n], best[i], best[(i + 1) % n]);
        let cross = (b.0 - a.0) * (c.1 - b.1) - (b.1 - a.1) * (c.0 - b.0);
        if cross != 0 { out.push(b); }
    }
    out.iter().map(|&(i, j)| [origin[0] + i as f64 * dx, origin[1] + j as f64 * dx]).collect()
}

/// Build the zoom project: the same duct, ports and blockers inside the box, the box cut out of the model, and the
/// coarse flow through the box top fed in as flow inlets. Returns the project and a short description of each inlet.
pub fn zoom_project(project: &Project, coarse: &CoarseField, spec: &ZoomSpec) -> Result<(Project, Vec<String>)> {
    let [nx, ny, nz] = coarse.dims;
    let dx = coarse.dx;
    let k = ((spec.plane_z - coarse.origin[2]) / dx).floor();
    if k < 1.0 || k as usize + 1 >= nz { bail!("the plane z = {} mm is outside the coarse run", spec.plane_z); }
    let k = k as usize;
    let idx = |x: usize, y: usize| x + nx * (y + ny * k);
    // downward speed of the air crossing the plane, through the duct's channels (not the slow air around it)
    let down = |x: usize, y: usize| -> f64 { if coarse.cells[idx(x, y)] == 1 { 0.0 } else { -(coarse.vel[idx(x, y)][2] as f64) } };
    let mut vmax = 0.0f64;
    for y in 0..ny { for x in 0..nx { vmax = vmax.max(down(x, y)); } }
    if vmax < 0.5 { bail!("no air flows down through z = {} mm in the coarse run (peak {:.2} m/s)", spec.plane_z, vmax); }
    let thr = 0.2 * vmax;
    let (xlo, xhi) = (spec.x_range[0], spec.x_range[1]);
    let (ylo, yhi) = (spec.y_range[0], spec.y_range[1]);
    // channels = connected groups of fast downward cells inside the box
    let mut label = vec![u32::MAX; nx * ny];
    let mut groups: Vec<Vec<(usize, usize)>> = vec![];
    for y in 0..ny {
        for x in 0..nx {
            let (px, py) = (coarse.origin[0] + (x as f64 + 0.5) * dx, coarse.origin[1] + (y as f64 + 0.5) * dx);
            if label[x + nx * y] != u32::MAX || down(x, y) < thr || px < xlo || px > xhi || py < ylo || py > yhi { continue; }
            let id = groups.len() as u32;
            let mut stack = vec![(x, y)];
            label[x + nx * y] = id;
            let mut g = vec![];
            while let Some((cx, cy)) = stack.pop() {
                g.push((cx, cy));
                for (ddx, ddy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (qx, qy) = (cx as i64 + ddx, cy as i64 + ddy);
                    if qx < 0 || qy < 0 || qx >= nx as i64 || qy >= ny as i64 { continue; }
                    let (qx, qy) = (qx as usize, qy as usize);
                    let (px, py) = (coarse.origin[0] + (qx as f64 + 0.5) * dx, coarse.origin[1] + (qy as f64 + 0.5) * dx);
                    if label[qx + nx * qy] == u32::MAX && down(qx, qy) >= thr && px >= xlo && px <= xhi && py >= ylo && py <= yhi {
                        label[qx + nx * qy] = id;
                        stack.push((qx, qy));
                    }
                }
            }
            if g.len() >= 4 { groups.push(g); }
        }
    }
    if groups.is_empty() { bail!("no channel crosses z = {} mm inside the box", spec.plane_z); }
    let mut out = project.clone();
    let inside = |p: [f64; 3]| p[0] >= xlo && p[0] <= xhi && p[1] >= ylo && p[1] <= yhi && p[2] >= spec.z_min && p[2] <= spec.plane_z;
    // keep what lies in the box: outlets and blockers; the fan and its source are outside
    out.ports.retain(|p| !p.kind.is_inlet() && inside(p.cap.centroid()));
    out.blockers.retain(|b| inside(b.cap.centroid()));
    out.plugs.retain(|p| inside(p.center));
    out.sources.clear();
    let body = project.ducts.first().map(|d| d.name.clone()).unwrap_or_else(|| "duct".into());
    let mut notes = vec![];
    let cell_area = (dx * 1e-3) * (dx * 1e-3);
    for (gi, g) in groups.iter().enumerate() {
        let q: f64 = g.iter().map(|&(x, y)| down(x, y).max(0.0)).sum::<f64>() * cell_area * 60000.0; // L/min
        if q < 0.05 { continue; }
        let set: HashSet<(i64, i64)> = g.iter().map(|&(x, y)| (x as i64, y as i64)).collect();
        let poly2 = outline(&set, [coarse.origin[0], coarse.origin[1]], dx);
        if poly2.len() < 3 { continue; }
        let polygon: Vec<[f64; 3]> = poly2.iter().map(|p| [p[0], p[1], spec.plane_z]).collect();
        let c = [poly2.iter().map(|p| p[0]).sum::<f64>() / poly2.len() as f64, poly2.iter().map(|p| p[1]).sum::<f64>() / poly2.len() as f64, spec.plane_z];
        let cap = PortCap { origin: c, normal: [0.0, 0.0, 1.0], polygon, triangles: vec![], holes: vec![] };
        let name = format!("zoom_in_{}", gi + 1);
        notes.push(format!("{name}: {:.1} L/min through {:.0} mm² at ({:.1}, {:.1}, {:.1})", q, cap.area(), c[0], c[1], c[2]));
        out.ports.push(Port { name, body: body.clone(), kind: PortKind::FlowInlet { l_per_min: q }, cap });
    }
    if notes.is_empty() { bail!("the channels at z = {} mm carry too little flow", spec.plane_z); }
    out.name = format!("{} zoom", project.name);
    out.solver.cell_size_mm = spec.fine_dx;
    out.solver.reference_speed = 0.0;
    out.domain.padding_mm = 3.0;
    out.domain.outlet_clearance_mm = 15.0;
    out.domain.crop = Some([xlo, ylo, spec.z_min, xhi, yhi, spec.plane_z]);
    Ok((out, notes))
}
