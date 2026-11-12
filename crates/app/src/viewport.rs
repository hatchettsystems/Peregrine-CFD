//! Orbit camera and software-projected mesh drawing (painter's algorithm).
use eframe::egui;
use eframe::egui::{epaint::Vertex, Color32, Mesh, Pos2, Rect, Shape, Stroke};
use peregrine_core::vec3::*;
use peregrine_core::TriMesh;

#[derive(Clone)]
pub struct Camera {
    pub target: V3,
    pub dist: f64,
    pub yaw: f64,
    pub pitch: f64,
}

impl Camera {
    pub fn new() -> Self {
        Camera { target: [0.0; 3], dist: 100.0, yaw: -0.6, pitch: 0.45 }
    }
    pub fn frame(&mut self, min: V3, max: V3) {
        self.target = scale(add(min, max), 0.5);
        self.dist = len(sub(max, min)).max(1.0) * 1.4;
    }
    fn axes(&self) -> (V3, V3, V3) {
        // Z-up orbit: eye direction from target
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let eye_dir = [cp * cy, cp * sy, sp];
        let fwd = scale(eye_dir, -1.0);
        let right = normalize(cross(fwd, [0.0, 0.0, 1.0]));
        let up = cross(right, fwd);
        (right, up, fwd)
    }
    pub fn eye(&self) -> V3 {
        let (_, _, f) = self.axes();
        sub(self.target, scale(f, self.dist))
    }
    /// Project to screen; returns (pos, depth) or None if behind the eye.
    pub fn project(&self, rect: Rect, p: V3) -> Option<(Pos2, f64)> {
        let (r, u, f) = self.axes();
        let d = sub(p, self.eye());
        let (x, y, z) = (dot(d, r), dot(d, u), dot(d, f));
        if z < 1e-3 {
            return None;
        }
        let focal = rect.height() as f64 * 0.5 / (35f64.to_radians() * 0.5).tan();
        Some((Pos2::new(rect.center().x + (x / z * focal) as f32, rect.center().y - (y / z * focal) as f32), z))
    }
    /// Precomputed projection for projecting many points in one frame (picking).
    pub fn projector(&self, rect: Rect) -> Projector {
        let (r, u, f) = self.axes();
        Projector { r, u, f, eye: self.eye(), focal: rect.height() as f64 * 0.5 / (35f64.to_radians() * 0.5).tan(), center: rect.center() }
    }
    /// Camera data for the graphics-card renderer (same projection as [`Camera::project`]).
    pub fn gl_uniform(&self, rect: Rect) -> crate::viewport_gl::CamUniform {
        let (r, u, f) = self.axes();
        let eye = self.eye();
        let l = self.light();
        let focal = rect.height() as f64 * 0.5 / (35f64.to_radians() * 0.5).tan();
        let c = |v: V3| [v[0] as f32, v[1] as f32, v[2] as f32];
        crate::viewport_gl::CamUniform { eye: c(eye), r: c(r), u: c(u), f: c(f), focal: focal as f32, half: [rect.width() * 0.5, rect.height() * 0.5], near: (self.dist * 0.01).max(0.05) as f32, far: (self.dist * 30.0 + 1000.0) as f32, light: c(l) }
    }
    /// Eye ray through a screen point: (origin, unit direction).
    pub fn ray(&self, rect: Rect, p: Pos2) -> (V3, V3) {
        let (r, u, f) = self.axes();
        let focal = rect.height() as f64 * 0.5 / (35f64.to_radians() * 0.5).tan();
        let (x, y) = ((p.x - rect.center().x) as f64 / focal, (rect.center().y - p.y) as f64 / focal);
        (self.eye(), normalize(add(add(f, scale(r, x)), scale(u, y))))
    }
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx as f64 * 0.008;
        self.pitch = (self.pitch + dy as f64 * 0.008).clamp(-1.5, 1.5);
    }
    pub fn pan(&mut self, rect: Rect, dx: f32, dy: f32) {
        let (r, u, _) = self.axes();
        let k = self.dist * 2.0 * (35f64.to_radians() * 0.5).tan() / rect.height() as f64;
        self.target = add(self.target, add(scale(r, -dx as f64 * k), scale(u, dy as f64 * k)));
    }
    pub fn zoom(&mut self, scroll: f32) {
        self.dist = (self.dist * (1.0 - scroll as f64 * 0.0015)).clamp(1.0, 5000.0);
    }
    pub fn light(&self) -> V3 {
        let (r, u, f) = self.axes();
        normalize(add(add(scale(f, -1.0), scale(u, 0.6)), scale(r, -0.4)))
    }
}

/// Distance along the ray `o + t d` to triangle `abc` (Moller-Trumbore), if it hits it in front of the eye.
pub fn ray_triangle(o: V3, d: V3, a: V3, b: V3, c: V3) -> Option<f64> {
    let (e1, e2) = (sub(b, a), sub(c, a));
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-14 { return None; }
    let inv = 1.0 / det;
    let s = sub(o, a);
    let u = dot(s, p) * inv;
    if !(0.0..=1.0).contains(&u) { return None; }
    let q = cross(s, e1);
    let v = dot(d, q) * inv;
    if v < 0.0 || u + v > 1.0 { return None; }
    let t = dot(e2, q) * inv;
    (t > 1e-9).then_some(t)
}

/// Draw a shaded closed mesh with back-face culling, depth-sorted triangles.
/// Smooth vertex normals (area-weighted average of the faces around each vertex), cached per mesh.
/// Faces of a STEP part keep their own vertices, so a curved face is smoothed while an edge between two faces stays sharp.
fn vertex_normals(mesh: &TriMesh) -> std::sync::Arc<Vec<V3>> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, usize), Arc<Vec<V3>>>>> = OnceLock::new();
    let key = (mesh.vertices.as_ptr() as usize, mesh.vertices.len(), mesh.triangles.len());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(v) = cache.lock().unwrap().get(&key) {
        return v.clone();
    }
    let mut acc = vec![[0.0f64; 3]; mesh.vertices.len()];
    for i in 0..mesh.triangles.len() {
        let [a, b, c] = mesh.tri(i);
        let n = cross(sub(b, a), sub(c, a)); // length = 2 x area
        for &v in &mesh.triangles[i] {
            let v = v as usize;
            acc[v] = add(acc[v], n);
        }
    }
    let out: Vec<V3> = acc.into_iter().map(|n| { let l = len(n); if l > 1e-12 { scale(n, 1.0 / l) } else { [0.0, 0.0, 1.0] } }).collect();
    let arc = Arc::new(out);
    let mut c = cache.lock().unwrap();
    if c.len() > 256 { c.clear(); }
    c.insert(key, arc.clone());
    arc
}

pub fn draw_mesh(painter: &egui::Painter, rect: Rect, cam: &Camera, mesh: &TriMesh, base: Color32, alpha: u8, outline: bool) {
    let light = cam.light();
    let eye = cam.eye();
    let normals = vertex_normals(mesh);
    let shade_of = |n: V3| 0.35 + 0.65 * dot(n, light).max(0.0) as f32;
    let tint = |sh: f32| Color32::from_rgba_unmultiplied((base.r() as f32 * sh) as u8, (base.g() as f32 * sh) as u8, (base.b() as f32 * sh) as u8, alpha);
    let mut items: Vec<(f64, [Pos2; 3], [Color32; 3])> = Vec::with_capacity(mesh.triangles.len());
    for i in 0..mesh.triangles.len() {
        let [a, b, c] = mesh.tri(i);
        let n = normalize(cross(sub(b, a), sub(c, a)));
        let centroid = scale(add(add(a, b), c), 1.0 / 3.0);
        if dot(n, sub(eye, centroid)) <= 0.0 {
            continue;
        }
        let (Some((pa, da)), Some((pb, db)), Some((pc, dc))) = (cam.project(rect, a), cam.project(rect, b), cam.project(rect, c)) else { continue };
        let t = mesh.triangles[i];
        // smooth shading unless the vertex normal strays far from the face (a crease inside one face)
        let col = |v: u32| { let vn = normals[v as usize]; tint(shade_of(if dot(vn, n) > 0.7 { vn } else { n })) };
        items.push(((da + db + dc) / 3.0, [pa, pb, pc], [col(t[0]), col(t[1]), col(t[2])]));
    }
    items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut m = Mesh::default();
    for (_, p, c) in &items {
        let base_i = m.vertices.len() as u32;
        for (q, col) in p.iter().zip(c) {
            m.vertices.push(Vertex { pos: *q, uv: egui::epaint::WHITE_UV, color: *col });
        }
        m.indices.extend([base_i, base_i + 1, base_i + 2]);
        if m.vertices.len() > 60_000 {
            painter.add(Shape::mesh(std::mem::take(&mut m)));
        }
    }
    painter.add(Shape::mesh(m));
    if outline {
        for (_, p, _) in items.iter().step_by(1) {
            painter.add(Shape::closed_line(p.to_vec(), Stroke::new(0.5, Color32::from_black_alpha(60))));
        }
    }
}

pub fn draw_polygon(painter: &egui::Painter, rect: Rect, cam: &Camera, pts: &[V3], stroke: Stroke) {
    let pp: Vec<Pos2> = pts.iter().filter_map(|p| cam.project(rect, *p).map(|x| x.0)).collect();
    if pp.len() == pts.len() && pp.len() > 2 {
        painter.add(Shape::closed_line(pp, stroke));
    }
}

pub struct Projector {
    r: V3,
    u: V3,
    f: V3,
    pub eye: V3,
    focal: f64,
    center: Pos2,
}

impl Projector {
    pub fn project(&self, p: V3) -> Option<(Pos2, f64)> {
        let d = sub(p, self.eye);
        let (x, y, z) = (dot(d, self.r), dot(d, self.u), dot(d, self.f));
        if z < 1e-3 {
            return None;
        }
        Some((Pos2::new(self.center.x + (x / z * self.focal) as f32, self.center.y - (y / z * self.focal) as f32), z))
    }
}
