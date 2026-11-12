//! Hardware-accelerated mesh drawing. The triangles of every body are uploaded to the graphics card once; each frame
//! only the camera changes, and the card does the projection, depth test and shading (flat, one normal per triangle).
//! This replaces the old per-frame CPU projection and depth sort, which got slow with detailed STEP meshes.
use eframe::egui_glow::{self, CallbackFn};
use eframe::glow::{self, HasContext};
use peregrine_core::vec3::*;
use peregrine_core::TriMesh;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static DEAD: OnceLock<Mutex<Vec<u64>>> = OnceLock::new();
fn dead() -> &'static Mutex<Vec<u64>> { DEAD.get_or_init(|| Mutex::new(vec![])) }

/// One body's triangles, as interleaved position + flat normal floats, waiting to be (or already) on the card.
pub struct GlBody {
    id: u64,
    data: Mutex<Option<Vec<f32>>>,
    tris: usize,
}

impl GlBody {
    pub fn new(mesh: &TriMesh) -> GlBody {
        let mut v: Vec<f32> = Vec::with_capacity(mesh.triangles.len() * 18);
        for i in 0..mesh.triangles.len() {
            let [a, b, c] = mesh.tri(i);
            let n = normalize(cross(sub(b, a), sub(c, a)));
            for p in [a, b, c] {
                v.extend_from_slice(&[p[0] as f32, p[1] as f32, p[2] as f32, n[0] as f32, n[1] as f32, n[2] as f32]);
            }
        }
        GlBody { id: NEXT_ID.fetch_add(1, Ordering::Relaxed), data: Mutex::new(Some(v)), tris: mesh.triangles.len() }
    }
}

impl Drop for GlBody {
    fn drop(&mut self) {
        dead().lock().unwrap().push(self.id);
    }
}

#[derive(Clone)]
pub struct DrawItem {
    pub body: Arc<GlBody>,
    pub color: [f32; 4],
}

/// Camera for the vertex shader (lengths in mm, screen sizes in points).
#[derive(Clone, Copy)]
pub struct CamUniform {
    pub eye: [f32; 3],
    pub r: [f32; 3],
    pub u: [f32; 3],
    pub f: [f32; 3],
    pub focal: f32,
    pub half: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub light: [f32; 3],
}

struct Buf {
    vao: glow::VertexArray,
    vbo: glow::Buffer,
    tris: i32,
}

struct Inner {
    program: glow::Program,
    bufs: HashMap<u64, Buf>,
}

pub struct GlRenderer {
    inner: Mutex<Option<Inner>>,
}

pub fn shared() -> Arc<GlRenderer> {
    static R: OnceLock<Arc<GlRenderer>> = OnceLock::new();
    R.get_or_init(|| Arc::new(GlRenderer { inner: Mutex::new(None) })).clone()
}

const VS: &str = r#"#version 150
in vec3 a_pos;
in vec3 a_nrm;
uniform vec3 u_eye; uniform vec3 u_r; uniform vec3 u_u; uniform vec3 u_f;
uniform vec4 u_view; // focal, half width, half height, near
uniform float u_far;
out vec3 v_nrm;
out vec3 v_pos;
void main() {
    vec3 d = a_pos - u_eye;
    float x = dot(d, u_r); float y = dot(d, u_u); float z = dot(d, u_f);
    float n = u_view.w; float f = u_far;
    float cz = (f + n) / (f - n) * z - 2.0 * f * n / (f - n);
    gl_Position = vec4(x * u_view.x / u_view.y, y * u_view.x / u_view.z, cz, z);
    v_nrm = a_nrm; v_pos = a_pos;
}
"#;

const FS: &str = r#"#version 150
in vec3 v_nrm;
in vec3 v_pos;
uniform vec3 u_eye; uniform vec3 u_light; uniform vec4 u_color;
out vec4 frag;
void main() {
    vec3 n = normalize(v_nrm);
    if (dot(n, u_eye - v_pos) <= 0.0) discard;   // back faces, as before
    float sh = 0.35 + 0.65 * max(dot(n, u_light), 0.0);
    frag = vec4(u_color.rgb * sh, u_color.a);
}
"#;

impl Inner {
    unsafe fn new(gl: &glow::Context) -> Result<Inner, String> {
        let program = gl.create_program()?;
        let mut shaders = vec![];
        for (kind, src) in [(glow::VERTEX_SHADER, VS), (glow::FRAGMENT_SHADER, FS)] {
            let sh = gl.create_shader(kind)?;
            gl.shader_source(sh, src);
            gl.compile_shader(sh);
            if !gl.get_shader_compile_status(sh) { return Err(gl.get_shader_info_log(sh)); }
            gl.attach_shader(program, sh);
            shaders.push(sh);
        }
        gl.bind_attrib_location(program, 0, "a_pos");
        gl.bind_attrib_location(program, 1, "a_nrm");
        gl.link_program(program);
        if !gl.get_program_link_status(program) { return Err(gl.get_program_info_log(program)); }
        for sh in shaders { gl.detach_shader(program, sh); gl.delete_shader(sh); }
        Ok(Inner { program, bufs: HashMap::new() })
    }
}

impl GlRenderer {
    fn draw(&self, gl: &glow::Context, cam: &CamUniform, items: &[DrawItem]) {
        let mut guard = self.inner.lock().unwrap();
        unsafe {
            if guard.is_none() {
                match Inner::new(gl) {
                    Ok(i) => *guard = Some(i),
                    Err(e) => { eprintln!("viewport shader failed: {e}"); return; }
                }
            }
            let inner = guard.as_mut().unwrap();
            // free the buffers of bodies that no longer exist
            for id in dead().lock().unwrap().drain(..) {
                if let Some(b) = inner.bufs.remove(&id) { gl.delete_vertex_array(b.vao); gl.delete_buffer(b.vbo); }
            }
            // upload new bodies
            for it in items {
                if inner.bufs.contains_key(&it.body.id) { continue; }
                let Some(data) = it.body.data.lock().unwrap().take() else { continue };
                let vao = match gl.create_vertex_array() { Ok(v) => v, Err(_) => continue };
                let vbo = match gl.create_buffer() { Ok(v) => v, Err(_) => continue };
                gl.bind_vertex_array(Some(vao));
                gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck_cast(&data), glow::STATIC_DRAW);
                gl.enable_vertex_attrib_array(0);
                gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 24, 0);
                gl.enable_vertex_attrib_array(1);
                gl.vertex_attrib_pointer_f32(1, 3, glow::FLOAT, false, 24, 12);
                inner.bufs.insert(it.body.id, Buf { vao, vbo, tris: it.body.tris as i32 });
            }
            gl.depth_mask(true);
            gl.clear_depth_f32(1.0);
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_func(glow::LEQUAL);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(inner.program));
            let loc = |n: &str| gl.get_uniform_location(inner.program, n);
            gl.uniform_3_f32(loc("u_eye").as_ref(), cam.eye[0], cam.eye[1], cam.eye[2]);
            gl.uniform_3_f32(loc("u_r").as_ref(), cam.r[0], cam.r[1], cam.r[2]);
            gl.uniform_3_f32(loc("u_u").as_ref(), cam.u[0], cam.u[1], cam.u[2]);
            gl.uniform_3_f32(loc("u_f").as_ref(), cam.f[0], cam.f[1], cam.f[2]);
            gl.uniform_4_f32(loc("u_view").as_ref(), cam.focal, cam.half[0], cam.half[1], cam.near);
            gl.uniform_1_f32(loc("u_far").as_ref(), cam.far);
            gl.uniform_3_f32(loc("u_light").as_ref(), cam.light[0], cam.light[1], cam.light[2]);
            let color_loc = loc("u_color");
            for pass in 0..2 {
                if pass == 1 {
                    gl.enable(glow::BLEND);
                    gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
                    gl.depth_mask(false);
                }
                for it in items.iter().filter(|i| (i.color[3] >= 0.999) == (pass == 0)) {
                    let Some(b) = inner.bufs.get(&it.body.id) else { continue };
                    gl.uniform_4_f32(color_loc.as_ref(), it.color[0], it.color[1], it.color[2], it.color[3]);
                    gl.bind_vertex_array(Some(b.vao));
                    gl.draw_arrays(glow::TRIANGLES, 0, b.tris * 3);
                }
            }
            gl.depth_mask(true);
            gl.disable(glow::BLEND);
            gl.disable(glow::DEPTH_TEST);
            gl.bind_vertex_array(None);
            gl.use_program(None);
        }
    }
}

fn bytemuck_cast(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any bit pattern is a valid byte sequence
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// A paint callback that draws `items` with the card inside `rect` (the whole 3D canvas).
pub fn callback(rect: eframe::egui::Rect, cam: CamUniform, items: Vec<DrawItem>) -> eframe::egui::PaintCallback {
    let r = shared();
    eframe::egui::PaintCallback {
        rect,
        callback: Arc::new(CallbackFn::new(move |_info, painter: &egui_glow::Painter| {
            r.draw(painter.gl(), &cam, &items);
        })),
    }
}
