// Release builds on Windows are GUI-subsystem apps (no console window).
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod theme;
mod tutorial;
mod viewport;
mod viewport_gl;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use peregrine_core::port_cap::point_in_polygon;
use peregrine_core::vec3::{dot, len, sub};
use peregrine_core::{cap_from_loop_with_grid, external_visibility, inward_clearance, shapes, smooth_region_loops, EdgeGraph, PortCandidate, PortCap, RayGrid, TriMesh};
use peregrine_geom::{apply_roles, build_project_domain, domain_bounds, duct_port_candidates, load_bodies, project_bounds, Body};
use peregrine_project::presets::{self, FanPreset};
use peregrine_project::{BodyRole, Port, PortKind, Project, Source, SourceKind, SplitMode};
use peregrine_post::streamlines::{self as sl, SeedSpec, Streamline, VelocityField};
use peregrine_solver::{Backend, PortReport, Simulation, SourceReport};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use viewport::Camera;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool { Orbit, Pan, PickPort, Slice, Rim, Draw, Block }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab { Bodies, Ports, Results, Exceptions }
/// What the active tab asks the tab shell to do after the frame.
enum ShellCmd { New, Duplicate, Close(usize), Switch(usize), Tutorial, Zoom(Box<Project>) }

/// An edge loop found by the Rim tool with its quality scores.
#[derive(Clone)]
struct RimInfo { l: peregrine_core::RimLoop, vis: f64, clear: f64 }

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Sev { Info, Warning, Error }

/// One entry of the Exceptions tab: a short title, a severity and the full explanation.
#[derive(Clone)]
struct Issue { sev: Sev, title: String, detail: String }

/// Turn a free-text solver / setup message into a titled issue (titles are fixed so the list reads the same every run).
fn classify_issue(msg: &str) -> Issue {
    let quoted = msg.split('\'').nth(1).map(|x| x.to_string());
    let (sev, title) = if msg.contains("ESTIMATED") {
        (Sev::Warning, format!("Estimated fan curve{}", quoted.map(|q| format!(": {q}")).unwrap_or_default()))
    } else if msg.starts_with("Air can leave") {
        (Sev::Warning, "Air leaks out of the duct".to_string())
    } else if msg.starts_with("Reference speed raised") {
        (Sev::Info, "Reference speed raised".to_string())
    } else if msg.starts_with("The graphics card was NOT used") {
        (Sev::Error, "Running on the CPU: the GPU was not used".to_string())
    } else if msg.contains("coarser than some walls") {
        (Sev::Warning, "Cells coarser than the thinnest walls".to_string())
    } else if msg.starts_with("Possible intentional") {
        (Sev::Warning, "Possible intentional openings left unsealed".to_string())
    } else if msg.starts_with("Auto-sealed") {
        (Sev::Info, "Small gaps sealed automatically".to_string())
    } else if msg.contains("too coarse") {
        (Sev::Warning, "Cells too coarse for the physical viscosity".to_string())
    } else if msg.starts_with("Viscosity is") {
        (Sev::Warning, "Effective viscosity inflated".to_string())
    } else if msg.contains("ray columns") {
        (Sev::Warning, "Geometry may not be watertight".to_string())
    } else if msg.contains("not accounted for") {
        (Sev::Warning, "Flow does not balance".to_string())
    } else {
        (Sev::Warning, "Notice".to_string())
    };
    Issue { sev, title, detail: msg.to_string() }
}

/// Everything worth flagging about a run, ordered errors first.
fn issues_of(s: &Snapshot, project: &Project) -> Vec<Issue> {
    let mut v: Vec<Issue> = vec![];
    if let Some(e) = &s.error { v.push(Issue { sev: Sev::Error, title: "Simulation failed".into(), detail: e.clone() }); }
    for w in &s.warnings { v.push(classify_issue(w)); }
    if s.inflation > 3.0 { v.push(classify_issue(&format!("Viscosity is {:.0}x physical at this cell size: patterns are qualitative, pressures overestimated.", s.inflation))); }
    let q_in: f64 = s.report.iter().filter(|r| r.inlet).map(|r| r.flow_l_min).sum();
    let net: f64 = s.report.iter().map(|r| r.flow_l_min).sum();
    if q_in > 1e-6 && net.abs() > 0.08 * q_in && s.step > 2000 {
        let msg = format!("{:.2} CFM of the {:.2} CFM going in ({:.0}%) is not accounted for by your outlets. Air is leaving through something that is not a defined port (a gap, or an opening that is not a port), or the run has not settled yet. Check the leak warning, define the missing openings as ports, or block them.", cfm(net.abs()), cfm(q_in), 100.0 * net.abs() / q_in);
        let severe = net.abs() > 0.25 * q_in;
        v.push(Issue { sev: if severe { Sev::Error } else { Sev::Warning }, title: if severe { "Major leak: most of the air is not leaving through the outlets".into() } else { "Flow does not balance".into() }, detail: msg });
    }
    if s.step > 1500 {
        // outlets that carry far less than their share of the opening area
        let outs: Vec<(&peregrine_solver::PortReport, f64)> = s.report.iter().filter(|r| !r.inlet).map(|r| (r, project.ports.iter().find(|p| p.name == r.name).map_or(0.0, |p| p.cap.area()))).collect();
        let total_out: f64 = outs.iter().map(|(r, _)| (-r.flow_l_min).max(0.0)).sum();
        let total_area: f64 = outs.iter().map(|(_, a)| *a).sum();
        if outs.len() >= 2 && total_area > 0.0 && q_in > 1e-6 {
            for (r, a) in &outs {
                let (ash, fsh) = (a / total_area, (-r.flow_l_min).max(0.0) / total_out.max(1e-9));
                if ash > 0.2 && fsh < 0.25 * ash {
                    v.push(Issue { sev: Sev::Warning, title: format!("Outlet '{}' is starved", r.name), detail: format!("'{}' is {:.0}% of the outlet area ({:.0} mm²) but carries only {:.1}% of the outflow ({:.2} CFM). A large opening that passes almost no air is usually cut off from the duct (a pocket, a closed-off passage or a part in the way), or air is escaping elsewhere before it gets there. Check the flow path in the streamlines and the leak warning.", r.name, 100.0 * ash, a, 100.0 * fsh, cfm(-r.flow_l_min)) });
                }
            }
        }
        // how hard the fan is working
        for g in &s.sources {
            if let Some(Source { kind: SourceKind::Fan { curve }, .. }) = project.sources.iter().find(|x| x.name == g.name) {
                let qmax = curve.iter().map(|c| c.0).fold(0.0, f64::max);
                let pmax = curve.iter().map(|c| c.1).fold(0.0, f64::max);
                if qmax > 0.0 && pmax > 0.0 {
                    let (fq, fp) = (g.total_flow_l_min / qmax, g.pressure_pa / pmax);
                    if fq < 0.35 {
                        v.push(Issue { sev: if fq < 0.1 { Sev::Error } else { Sev::Warning }, title: format!("Major flow restriction at '{}'", g.name), detail: format!("The duct lets '{}' deliver only {:.0}% of its free-air flow ({:.2} of {:.1} CFM) while working at {:.0}% of its maximum pressure ({:.3} of {:.3} psi). The duct is a severe restriction for this fan.", g.name, 100.0 * fq, cfm(g.total_flow_l_min), cfm(qmax), 100.0 * fp, psi(g.pressure_pa), psi(pmax)) });
                    } else if fq > 0.85 && fp < 0.15 {
                        v.push(Issue { sev: Sev::Warning, title: format!("'{}' is barely loaded", g.name), detail: format!("'{}' delivers {:.0}% of its free-air flow at only {:.0}% of its maximum pressure. A real duct usually pressurises noticeably; this suggests air is escaping (a leak) or the run has not settled.", g.name, 100.0 * fq, 100.0 * fp) });
                    }
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    v.retain(|i| seen.insert(i.title.clone()));
    v.sort_by(|a, b| b.sev.cmp(&a.sev));
    v
}

fn chip(ui: &mut egui::Ui, text: &str, col: Color32) {
    egui::Frame::new().fill(col).corner_radius(egui::CornerRadius::same(7)).inner_margin(egui::Margin::symmetric(6, 0)).show(ui, |ui| {
        ui.label(egui::RichText::new(text).size(10.5).strong().color(Color32::BLACK));
    });
}

fn sev_color(s: Sev) -> Color32 {
    match s { Sev::Error => Color32::from_rgb(0xE0, 0x5A, 0x4F), Sev::Warning => Color32::from_rgb(0xF2, 0xD0, 0x3A), Sev::Info => Color32::from_rgb(0x6C, 0xB4, 0xF0) }
}

/// First "(x, y, z)" triple of numbers in a message, as a point in mm.
fn parse_coords(msg: &str) -> Option<[f64; 3]> {
    let mut rest = msg;
    while let Some(i) = rest.find('(') {
        let after = &rest[i + 1..];
        let end = after.find(')')?;
        let parts: Vec<&str> = after[..end].split(',').collect();
        if parts.len() == 3 {
            let v: Vec<f64> = parts.iter().filter_map(|p| p.trim().parse().ok()).collect();
            if v.len() == 3 { return Some([v[0], v[1], v[2]]); }
        }
        rest = &after[end..];
    }
    None
}

/// Where in the model an issue is about, when it says so (shown as a clickable marker in the viewport).
fn issue_point(i: &Issue, all: &[Issue], project: &Project) -> Option<[f64; 3]> {
    if let Some(p) = parse_coords(&i.detail) { return Some(p); }
    if let Some(name) = i.title.strip_prefix("Outlet '").and_then(|t| t.split('\'').next()) {
        return project.ports.iter().find(|p| p.name == name).map(|p| p.cap.centroid());
    }
    if i.title.starts_with("Major leak") { return all.iter().find_map(|o| parse_coords(&o.detail)); }
    None
}

#[derive(Clone)]
struct TabInfo { title: String, running: bool }

#[derive(Clone, Copy, PartialEq, Eq)]
enum MeshMode { Shaded, Ghost, Hidden }
#[derive(Clone, Copy, PartialEq, Eq)]
enum LineColor { Speed, Source, Pressure }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field { Speed, Pressure }
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Target { Cand(usize), Port(usize) }

#[derive(Default)]
struct Snapshot {
    step: usize,
    total: usize,
    engine: String,
    dims: [usize; 3],
    origin: [f64; 3],
    dx: f64,
    cells: Arc<Vec<u8>>,
    vel: Arc<Vec<[f32; 3]>>,
    ref_speed: f64,
    speed: Vec<f32>,
    pressure: Vec<f32>,
    report: Vec<PortReport>,
    sources: Vec<SourceReport>,
    inflation: f64,
    warnings: Vec<String>,
    sealed: Vec<[f64; 5]>,
    finished: bool,
    error: Option<String>,
    version: u64,
}

struct RunHandle {
    stop: Arc<AtomicBool>,
    snap: Arc<Mutex<Snapshot>>,
}

struct App {
    project: Project,
    project_path: Option<PathBuf>,
    bodies: Vec<Body>,
    visible: Vec<bool>,
    gl_bodies: Vec<Option<std::sync::Arc<viewport_gl::GlBody>>>,
    pick_cache: std::cell::RefCell<Option<([i32; 2], [u64; 4], Option<(usize, usize)>)>>,
    candidates: Vec<(String, PortCandidate)>,
    cam: Camera,
    tool: Tool,
    tab: Tab,
    sel_cand: Option<usize>,
    sel_port: Option<usize>,
    backend: Backend,
    run: Option<RunHandle>,
    last: Option<Arc<Mutex<Snapshot>>>,
    field: Field,
    slice_axis: usize,
    slice_pos: f32,
    status: String,
    ctx_target: Option<Target>,
    hide_screws: bool,
    screw_max_d: f64,
    detect_curved: bool,
    show_hidden: bool,
    tut: Option<usize>,
    hidden_caps: std::collections::HashSet<[i64; 3]>,
    zoom_plane: f64,
    zoom_zmin: f64,
    zoom_box: [f64; 4],
    zoom_dx: f64,
    leak_notes: Vec<String>,
    sealed_check: Vec<[f64; 5]>,
    show_sealed: bool,
    plug_radius: f64,
    show_markers: bool,
    focus_issue: Option<String>,
    show_ports: bool,
    line_cutoff: f32,
    undo: Vec<Project>,
    redo: Vec<Project>,
    last_project: Option<(String, Project)>,
    block_cands: Vec<(String, PortCandidate)>,
    block_max_d: f64,
    block_stale: bool,
    sel_block: Option<usize>,
    draw_close_requested: bool,
    draw_undo_requested: bool,
    draw_clear_requested: bool,
    // hand-drawn edge loop
    draw_pts: Vec<u32>,
    draw_body: Option<usize>,
    draw_path: Vec<u32>,
    draw_marks: Vec<usize>,
    draw_graph: Option<(usize, EdgeGraph)>,
    rim_loops: Vec<RimInfo>,
    rim_sel: Option<usize>,
    rim_body: Option<usize>,
    manual_openings: Vec<(String, PortCandidate)>,
    sel_bodies: Vec<usize>,
    duct_name_buf: String,
    scroll_to_body: Option<usize>,
    ctx_body: Option<usize>,
    mesh_mode: MeshMode,
    show_lines: bool,
    show_slice: bool,
    line_color: LineColor,
    line_width: f32,
    lines: Vec<Streamline>,
    lines_vmax: f32,
    lines_prange: (f32, f32),
    probe_on: bool,
    probe_pins: Vec<[f64; 3]>,
    lines_request_sig: u64,
    lines_job: Option<std::sync::mpsc::Receiver<(u64, Vec<Streamline>)>>,
    // tab shell plumbing
    tab_infos: Vec<TabInfo>,
    tab_active: usize,
    cmds: Vec<ShellCmd>,
}

// Ask hybrid-graphics laptops (NVIDIA Optimus, AMD PowerXpress) to run this program on the discrete graphics card.
#[cfg(windows)]
#[no_mangle]
pub static NvOptimusEnablement: u32 = 1;
#[cfg(windows)]
#[no_mangle]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;

/// Display units: the solver works in L/min and Pa; the interface shows CFM and PSI.
const CFM_LPM: f64 = 28.3168;
const PSI_PA: f64 = 6894.757;
fn cfm(lpm: f64) -> f64 { lpm / CFM_LPM }
fn psi(pa: f64) -> f64 { pa / PSI_PA }

/// A number box that stores L/min but shows and accepts CFM.
fn flow_drag(ui: &mut egui::Ui, v: &mut f64, speed: f64) -> egui::Response {
    ui.add(egui::DragValue::new(v).speed(speed)
        .custom_formatter(|x, _| format!("{:.2} CFM", cfm(x)))
        .custom_parser(|t| t.trim().to_lowercase().trim_end_matches("cfm").trim().parse::<f64>().ok().map(|c| c * CFM_LPM)))
}

/// A number box that stores Pa but shows and accepts PSI.
fn pressure_drag(ui: &mut egui::Ui, v: &mut f64, speed: f64) -> egui::Response {
    ui.add(egui::DragValue::new(v).speed(speed)
        .custom_formatter(|x, _| format!("{:.4} psi", psi(x)))
        .custom_parser(|t| t.trim().to_lowercase().trim_end_matches("psi").trim().parse::<f64>().ok().map(|c| c * PSI_PA)))
}

/// The graphics card the solver will use, found once in the background at start-up.
static GPU: std::sync::OnceLock<Option<peregrine_solver::gpu::GpuInfo>> = std::sync::OnceLock::new();

/// Usable graphics memory in GB: 90 % of the card's dedicated memory when it is known, otherwise a safe 7 GB.
fn gpu_budget_gb() -> f64 {
    match GPU.get() {
        Some(Some(g)) => g.vram_bytes.map_or(7.0, |b| b as f64 * 0.9 / 1e9),
        _ => 7.0,
    }
}

fn main() -> eframe::Result {
    std::thread::spawn(|| { let _ = GPU.set(peregrine_solver::gpu::query_gpu()); });
    let mut viewport = egui::ViewportBuilder::default().with_inner_size([1360.0, 820.0]).with_title("Peregrine CFD");
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon256.png")) {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }
    // depth buffer for the graphics-card mesh renderer, and smooth edges
    let opts = eframe::NativeOptions { viewport, depth_buffer: 24, ..Default::default() };
    eframe::run_native("Peregrine CFD", opts, Box::new(|cc| {
        theme::apply(&cc.egui_ctx);
        let mut app = App::new();
        let args: Vec<String> = std::env::args().collect();
        if let Some(pos) = args.iter().position(|a| a == "--mesh") {
            match args.get(pos + 1).map(String::as_str) { Some("hidden") => app.mesh_mode = MeshMode::Hidden, Some("ghost") => app.mesh_mode = MeshMode::Ghost, _ => {} }
        }
        if let Some(pos) = args.iter().position(|a| a == "--project") {
            if let Some(path) = args.get(pos + 1) {
                app.load_project_file(PathBuf::from(path));
                if args.iter().any(|a| a == "--run") {
                    if let Some(sp) = args.iter().position(|a| a == "--steps") { if let Some(n) = args.get(sp + 1).and_then(|v| v.parse().ok()) { app.project.solver.max_steps = n; } }
                    app.backend = Backend::Cpu;
                    app.start_run();
                }
            }
        }
        // debugging aids: --hide <body>, --frame <body>, --yaw <deg>, --pitch <deg>
        for (i, a) in args.iter().enumerate() {
            if a == "--hide" { if let Some(n) = args.get(i + 1) { if let Some(k) = app.bodies.iter().position(|b| &b.name == n) { app.visible[k] = false; } } }
            if a == "--frame" { if let Some(n) = args.get(i + 1) { if let Some(b) = app.bodies.iter().find(|b| &b.name == n) { let bb = b.mesh.bounds(); app.cam.frame(bb.min, bb.max); } } }
            if a == "--show-hidden" { app.show_hidden = true; app.refresh_candidates(); }
            if a == "--check-leaks" {
                app.check_leaks();
                app.tab = Tab::Exceptions;
                if let Some(c) = app.leak_notes.first().and_then(|m| parse_coords(m)) { app.cam.target = c; app.cam.dist = 75.0; }
                app.mesh_mode = MeshMode::Ghost;
                app.focus_issue = Some("Air leaks out of the duct".into());
            }
            if a == "--show-sealed" { app.show_sealed = true; }
            if a == "--look" { if let Some(v) = args.get(i + 1) { let n: Vec<f64> = v.split(',').filter_map(|x| x.parse().ok()).collect(); if n.len() >= 3 { app.cam.target = [n[0], n[1], n[2]]; if let Some(d) = n.get(3) { app.cam.dist = *d; } } } }
            if a == "--only-ducts" { for (k, b) in app.bodies.iter().enumerate() { if app.project.duct_of(&b.name).is_none() { app.visible[k] = false; } } }
            if a == "--tab" { match args.get(i + 1).map(|v| v.as_str()) { Some("ports") => app.tab = Tab::Ports, Some("results") => app.tab = Tab::Results, Some("exceptions") => app.tab = Tab::Exceptions, _ => {} } }
            if a == "--tutorial" { app.start_tutorial(); }
            if a == "--tutorial-selftest" {
                app.start_tutorial();
                for act in [tutorial::Act::DefineDuct, tutorial::Act::AddFan, tutorial::Act::AssignInlet, tutorial::Act::MakeOutlet, tutorial::Act::BlockHole] { app.tutorial_act(act); }
                eprintln!("selftest: ducts={} candidates={} ports={:?} blockers={} sources={}", app.project.ducts.len(), app.candidates.len(), app.project.ports.iter().map(|p| (p.name.clone(), p.kind.is_inlet())).collect::<Vec<_>>(), app.project.blockers.len(), app.project.sources.len());
                eprintln!("can_run: {:?}", app.can_run());
                if std::env::args().any(|x| x == "--run") { app.backend = Backend::Cpu; app.tutorial_act(tutorial::Act::Run); app.tab = Tab::Results; } else { std::process::exit(0); }
            }
            if a == "--tool" { match args.get(i + 1).map(|v| v.as_str()) { Some("block") => app.tool = Tool::Block, Some("draw") => app.tool = Tool::Draw, _ => {} } }
            if a == "--yaw" { if let Some(v) = args.get(i + 1).and_then(|v| v.parse::<f64>().ok()) { app.cam.yaw = v.to_radians(); } }
            if a == "--pitch" { if let Some(v) = args.get(i + 1).and_then(|v| v.parse::<f64>().ok()) { app.cam.pitch = v.to_radians(); } }
        }
        if std::env::args().any(|a| a == "--demo") {
            app.load_demo();
            if std::env::args().any(|a| a == "--run") {
                app.project.solver.max_steps = 1750;
                app.backend = Backend::Cpu;
                app.start_run();
            }
        }
        // first launch with nothing to open: offer the tutorial
        if std::env::args().len() == 1 && !tutorial::tutorial_done() { app.start_tutorial(); }
        Ok(Box::new(Shell { docs: vec![app], active: 0, counter: 1 }))
    }))
}

impl App {
    fn new() -> App {
        App {
            project: Project::new("untitled 1"),
            project_path: None,
            bodies: vec![],
            visible: vec![],
            gl_bodies: vec![],
            pick_cache: std::cell::RefCell::new(None),
            candidates: vec![],
            cam: Camera::new(),
            tool: Tool::Orbit,
            tab: Tab::Bodies,
            sel_cand: None,
            sel_port: None,
            backend: Backend::Auto,
            run: None,
            last: None,
            field: Field::Speed,
            slice_axis: 1,
            slice_pos: 0.5,
            status: "Open a geometry file (STL) or load the demo to begin.".into(),
            ctx_target: None,
            hide_screws: true,
            screw_max_d: 6.0,
            detect_curved: true,
            show_hidden: false,
            tut: None,
            hidden_caps: Default::default(),
            zoom_plane: -38.0,
            zoom_zmin: -66.0,
            zoom_box: [-36.0, -36.0, 36.0, 18.0],
            zoom_dx: 0.3,
            leak_notes: vec![],
            sealed_check: vec![],
            show_sealed: false,
            plug_radius: 3.0,
            show_markers: true,
            focus_issue: None,
            show_ports: true,
            line_cutoff: 0.0,
            undo: vec![],
            redo: vec![],
            last_project: None,
            block_cands: vec![],
            block_max_d: 12.0,
            block_stale: true,
            sel_block: None,
            draw_close_requested: false,
            draw_undo_requested: false,
            draw_clear_requested: false,
            draw_pts: vec![],
            draw_body: None,
            draw_path: vec![],
            draw_marks: vec![],
            draw_graph: None,
            rim_loops: vec![],
            rim_sel: None,
            rim_body: None,
            manual_openings: vec![],
            sel_bodies: vec![],
            duct_name_buf: String::new(),
            scroll_to_body: None,
            ctx_body: None,
            mesh_mode: MeshMode::Shaded,
            show_lines: true,
            show_slice: false,
            line_color: LineColor::Speed,
            line_width: 1.6,
            lines: vec![],
            lines_vmax: 1.0,
            lines_prange: (0.0, 1.0),
            probe_on: false,
            probe_pins: vec![],
            lines_request_sig: 0,
            lines_job: None,
            tab_infos: vec![],
            tab_active: 0,
            cmds: vec![],
        }
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.visible = vec![true; bodies.len()];
        self.gl_bodies.clear();
        self.pick_cache.replace(None);
        self.bodies = bodies;
        self.sel_bodies.clear();
        self.manual_openings.clear();
        self.rim_loops.clear();
        self.reclassify();
        let b = peregrine_geom::domain_bounds(&self.bodies, 0.0);
        if b.is_valid() {
            self.cam.frame(b.min, b.max);
        }
    }

    fn reclassify(&mut self) {
        apply_roles(&self.project, &mut self.bodies, false);
        self.refresh_candidates();
    }

    fn refresh_candidates(&mut self) {
        self.candidates = duct_port_candidates(&self.bodies, &self.project.ducts, 0.5, if self.hide_screws { Some(self.screw_max_d) } else { None }, self.detect_curved);
        // an opening that admits air is visible from outside and leads into a channel, not a pocket
        // only openings that fail BOTH tests (hard to see from outside and no air behind) are hidden as pockets; the
        // air-behind test depends on thin necks that a grid can close, so it must not hide an opening on its own
        if !self.show_hidden { self.candidates.retain(|c| !c.1.inline && (c.1.clearance >= 0.5 || c.1.visibility >= 0.5)); }
        self.candidates.extend(self.manual_openings.iter().cloned());
        self.sel_cand = None;
        self.block_stale = true;
    }

    /// Mesh of the duct that `body` belongs to (all its parts), or just the body itself.
    fn duct_mesh_and_name(&self, body: usize) -> (String, TriMesh) {
        let name = self.bodies[body].name.clone();
        if let Some(d) = self.project.duct_of(&name) {
            let mut m = TriMesh::default();
            for p in &d.parts {
                if let Some(b) = self.bodies.iter().find(|b| &b.name == p) { m.append(&b.mesh); }
            }
            (d.name.clone(), m)
        } else {
            (name, self.bodies[body].mesh.clone())
        }
    }

    /// The user clicked a surface with the Rim tool: list the closed edge loops that bound the smooth patch there.
    fn rim_pick(&mut self, body: usize, tri: usize) {
        let mut loops = smooth_region_loops(&self.bodies[body].mesh, tri, 25.0);
        loops.retain(|l| l.perimeter >= 5.0);
        loops.truncate(12);
        self.rim_body = Some(body);
        // score every loop: is it visible from outside, and does air continue behind it?
        let (_, duct_mesh) = self.duct_mesh_and_name(body);
        let grid = RayGrid::new(&duct_mesh);
        let infos: Vec<RimInfo> = loops
            .into_iter()
            .map(|l| {
                let (vis, clear) = match cap_from_loop_with_grid(&l.points, &grid) {
                    Some(cap) => (external_visibility(&cap, &grid), inward_clearance(&cap, &grid, 5.0)),
                    None => (0.0, 0.0),
                };
                RimInfo { l, vis, clear }
            })
            .collect();
        // prefer the loop that is both visible and leads into a channel; otherwise a rim band's inner (shorter) edge
        let best = infos.iter().enumerate().max_by(|a, b| (a.1.vis * a.1.clear).partial_cmp(&(b.1.vis * b.1.clear)).unwrap());
        let pick = match best {
            Some((i, r)) if r.vis * r.clear >= 0.25 => Some(i),
            _ if infos.len() == 2 => infos.iter().enumerate().min_by(|a, b| a.1.l.perimeter.partial_cmp(&b.1.l.perimeter).unwrap()).map(|x| x.0),
            _ if infos.is_empty() => None,
            _ => Some(0),
        };
        self.rim_sel = pick;
        self.status = if infos.is_empty() {
            "No closed edge loop found at that surface. Click the rim of the opening or the wall just inside it.".into()
        } else {
            format!("{} edge loop(s) found; the one that is visible from outside and leads into the duct is preselected. Press 'Add as opening'.", infos.len())
        };
        self.rim_loops = infos;
        self.tab = Tab::Ports;
    }

    fn add_rim_opening(&mut self, li: usize) {
        let (Some(info), Some(bi)) = (self.rim_loops.get(li).cloned(), self.rim_body) else { return };
        let l = info.l.clone();
        let (duct, mesh) = self.duct_mesh_and_name(bi);
        match cap_from_loop_with_grid(&l.points, &RayGrid::new(&mesh)) {
            Some(cap) => {
                let area = cap.area();
                let perim = l.perimeter;
                let cand = PortCandidate { area, parent_area: 0.0, circularity: (4.0 * std::f64::consts::PI * area / (perim * perim)).min(1.0), equiv_diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), visibility: info.vis, clearance: info.clear, inline: false, cap };
                self.manual_openings.push((duct, cand));
                self.refresh_candidates();
                self.sel_cand = Some(self.candidates.len() - 1);
                self.rim_loops.clear();
                self.rim_sel = None;
                self.status = format!("Added a {} opening ({:.1} mm²{}). Assign it to a source or make it an outlet.", if l.planarity_dev > 0.05 { "curved" } else { "flat" }, area, if l.planarity_dev > 0.05 { format!(", {:.1} mm off-plane", l.planarity_dev) } else { String::new() });
            }
            None => self.status = "Could not build a cap for that loop.".into(),
        }
    }

    // ---------- blockers (shut openings) ----------

    fn blocker_at(&self, cap: &PortCap) -> Option<usize> {
        let ctr = cap.centroid();
        self.project.blockers.iter().position(|b| len(sub(b.cap.centroid(), ctr)) < 0.05 && dot(b.cap.normal, cap.normal) > 0.99)
    }

    /// Find the small openings (screw holes and the like) that can be shut.
    fn refresh_block_cands(&mut self) {
        let max_d = self.block_max_d;
        let mut c = duct_port_candidates(&self.bodies, &self.project.ducts, 0.05, None, false);
        c.retain(|x| x.1.equiv_diameter <= max_d);
        self.block_cands = c;
        self.block_stale = false;
        self.sel_block = None;
    }

    fn add_blocker(&mut self, body: String, cap: PortCap) {
        if self.blocker_at(&cap).is_some() { return; }
        // a port that sat on this opening is replaced by the blocker
        let ctr = cap.centroid();
        self.project.ports.retain(|p| !(len(sub(p.cap.centroid(), ctr)) < 0.05 && dot(p.cap.normal, cap.normal) > 0.99));
        let name = Self::unique_name(|n| self.project.blockers.iter().any(|b| b.name == n), "block");
        self.project.blockers.push(peregrine_project::Blocker { name, body, cap });
        self.sel_port = None;
    }

    fn remove_blocker(&mut self, i: usize) {
        if i < self.project.blockers.len() { self.project.blockers.remove(i); }
        self.sel_block = None;
    }

    fn add_rim_blocker(&mut self, li: usize) {
        let (Some(info), Some(bi)) = (self.rim_loops.get(li).cloned(), self.rim_body) else { return };
        let (duct, mesh) = self.duct_mesh_and_name(bi);
        match cap_from_loop_with_grid(&info.l.points, &RayGrid::new(&mesh)) {
            Some(cap) => {
                let area = cap.area();
                self.add_blocker(duct, cap);
                self.rim_loops.clear();
                self.rim_sel = None;
                self.status = format!("Blocked a {:.1} mm² opening: it is now closed in the simulation.", area);
            }
            None => self.status = "Could not build a cap for that loop.".into(),
        }
    }

    // ---------- undo / redo of the project (ports, sources, blockers, ducts, settings) ----------

    /// Called every frame after the UI: when something in the project changed, remember the state before it.
    fn track_changes(&mut self, ctx: &egui::Context) {
        let active = ctx.input(|i| i.pointer.any_released() || i.events.iter().any(|e| matches!(e, egui::Event::Key { pressed: true, .. } | egui::Event::Text(_))));
        if !active && self.last_project.is_some() { return; }
        let now = format!("{:?}", self.project);
        match &self.last_project {
            None => self.last_project = Some((now, self.project.clone())),
            Some((old, prev)) if *old != now => {
                self.undo.push(prev.clone());
                if self.undo.len() > 100 { self.undo.remove(0); }
                self.redo.clear();
                self.last_project = Some((now, self.project.clone()));
            }
            _ => {}
        }
    }

    fn reset_history(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.last_project = None;
    }

    fn apply_project(&mut self, p: Project) {
        let reclass = format!("{:?}{:?}", p.ducts, p.geometry) != format!("{:?}{:?}", self.project.ducts, self.project.geometry);
        self.project = p;
        self.sel_port = None;
        self.sel_block = None;
        self.last_project = Some((format!("{:?}", self.project), self.project.clone()));
        if reclass { self.reclassify(); }
    }

    fn undo_project(&mut self) {
        if self.tool == Tool::Draw && !self.draw_pts.is_empty() { self.draw_undo(); return; }
        self.track_changes_now();
        if let Some(prev) = self.undo.pop() {
            self.redo.push(self.project.clone());
            self.apply_project(prev);
            self.status = "Undo".into();
        }
    }

    fn redo_project(&mut self) {
        self.track_changes_now();
        if let Some(next) = self.redo.pop() {
            self.undo.push(self.project.clone());
            self.apply_project(next);
            self.status = "Redo".into();
        }
    }

    /// Record a pending change right away (so undo never skips an edit that has not been sampled yet).
    fn track_changes_now(&mut self) {
        let now = format!("{:?}", self.project);
        if let Some((old, prev)) = &self.last_project {
            if *old != now { self.undo.push(prev.clone()); self.redo.clear(); }
        }
        self.last_project = Some((now, self.project.clone()));
    }

    // ---------- hand-drawn edge loop ----------

    fn draw_clear(&mut self) {
        self.draw_pts.clear();
        self.draw_path.clear();
        self.draw_marks.clear();
        self.draw_body = None;
    }

    fn draw_undo(&mut self) {
        if self.draw_pts.pop().is_some() {
            self.draw_marks.pop();
            let keep = self.draw_marks.last().copied().unwrap_or(0);
            self.draw_path.truncate(keep.max(if self.draw_pts.is_empty() { 0 } else { 1 }));
            if self.draw_pts.is_empty() { self.draw_clear(); }
        }
    }

    /// Add a clicked mesh vertex to the hand-drawn loop, joined to the previous point along the mesh edges.
    fn draw_add_vertex(&mut self, body: usize, v: u32) {
        if self.draw_body != Some(body) {
            self.draw_clear();
            self.draw_body = Some(body);
        }
        if self.draw_graph.as_ref().map_or(true, |g| g.0 != body) {
            self.draw_graph = Some((body, EdgeGraph::new(&self.bodies[body].mesh, 25.0)));
        }
        if let Some(&last) = self.draw_pts.last() {
            if last == v { return; }
            if let Some(path) = self.draw_graph.as_ref().and_then(|g| g.1.path(last, v, 4.0)) {
                self.draw_path.extend(path.into_iter().skip(1));
            } else {
                self.status = "Those two points are not connected on the surface.".into();
                return;
            }
        } else {
            self.draw_path.push(v);
        }
        self.draw_pts.push(v);
        self.draw_marks.push(self.draw_path.len());
        self.status = format!("{} point(s). Keep clicking along the rim; click the first point again or press Enter to close the loop.", self.draw_pts.len());
    }

    /// Close the hand-drawn loop (last point back to the first) and turn it into a scored, addable loop.
    fn draw_close(&mut self) {
        let (Some(body), Some(&first), Some(&last)) = (self.draw_body, self.draw_pts.first(), self.draw_pts.last()) else { return };
        if self.draw_pts.len() < 3 {
            self.status = "Click at least 3 points around the opening first.".into();
            return;
        }
        if let Some(path) = self.draw_graph.as_ref().and_then(|g| g.1.path(last, first, 4.0)) {
            self.draw_path.extend(path.into_iter().skip(1));
        }
        // drop the repeated first vertex at the end
        if self.draw_path.len() > 1 && self.draw_path.last() == self.draw_path.first() { self.draw_path.pop(); }
        let mesh = &self.bodies[body].mesh;
        let points: Vec<[f64; 3]> = self.draw_path.iter().map(|&v| mesh.vertices[v as usize]).collect();
        let perimeter: f64 = (0..points.len()).map(|i| { let (a, b) = (points[i], points[(i + 1) % points.len()]); ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() }).sum();
        let (c, n) = peregrine_core::rim::loop_frame(&points);
        let planarity_dev = points.iter().map(|p| ((p[0] - c[0]) * n[0] + (p[1] - c[1]) * n[1] + (p[2] - c[2]) * n[2]).abs()).fold(0.0, f64::max);
        let l = peregrine_core::RimLoop { points, perimeter, planarity_dev };
        let (_, duct_mesh) = self.duct_mesh_and_name(body);
        let grid = RayGrid::new(&duct_mesh);
        let (vis, clear) = match cap_from_loop_with_grid(&l.points, &grid) {
            Some(cap) => (external_visibility(&cap, &grid), inward_clearance(&cap, &grid, 5.0)),
            None => (0.0, 0.0),
        };
        self.rim_body = Some(body);
        self.rim_loops = vec![RimInfo { l, vis, clear }];
        self.rim_sel = Some(0);
        self.draw_clear();
        self.tab = Tab::Ports;
        self.status = "Loop closed. Press 'Add as opening' to use it as an opening.".into();
    }

    fn suggested_bodies(&self) -> Vec<usize> {
        (0..self.bodies.len()).filter(|&i| self.project.is_suggested(&self.bodies[i].name)).collect()
    }

    /// Select body `i` (replacing the selection, or toggling it when `add`), and reveal it in the Bodies list.
    fn select_body(&mut self, i: usize, add: bool) {
        if add {
            if let Some(p) = self.sel_bodies.iter().position(|&x| x == i) { self.sel_bodies.remove(p); } else { self.sel_bodies.push(i); }
        } else {
            self.sel_bodies = vec![i];
        }
        self.scroll_to_body = Some(i);
        self.tab = Tab::Bodies;
        if let Some(&first) = self.sel_bodies.first() {
            let n = &self.bodies[first].name;
            self.duct_name_buf = n.rsplit_once(':').filter(|(_, t)| t.chars().all(|c| c.is_ascii_digit())).map_or(n.clone(), |(h, _)| h.to_string());
        }
    }

    fn define_duct_from(&mut self, idx: &[usize], name: &str) {
        let parts: Vec<String> = idx.iter().map(|&i| self.bodies[i].name.clone()).collect();
        if parts.is_empty() { return; }
        let n = self.project.define_duct(name, &parts);
        self.status = format!("Defined duct '{n}' from {} part(s). Its openings are in the Ports tab.", parts.len());
        self.reclassify();
    }

    fn remove_from_ducts(&mut self, idx: &[usize]) {
        for &i in idx {
            let n = self.bodies[i].name.clone();
            for d in &mut self.project.ducts { d.parts.retain(|p| p != &n); }
            self.project.geometry.ignored.remove(&n);
        }
        self.project.ducts.retain(|d| !d.parts.is_empty());
        self.reclassify();
    }

    fn ignore_bodies(&mut self, idx: &[usize]) {
        self.remove_from_ducts(idx);
        for &i in idx { self.project.geometry.ignored.insert(self.bodies[i].name.clone()); }
        self.reclassify();
    }

    fn add_to_duct(&mut self, idx: &[usize], duct: &str) {
        let add: Vec<String> = idx.iter().map(|&i| self.bodies[i].name.clone()).collect();
        for d in &mut self.project.ducts { d.parts.retain(|p| !add.contains(p)); }
        if let Some(d) = self.project.ducts.iter_mut().find(|d| d.name == duct) { d.parts.extend(add); }
        self.project.ducts.retain(|d| !d.parts.is_empty());
        self.reclassify();
    }

    fn load_demo(&mut self) {
        self.reset_history();
        self.project = Project::new("demo");
        self.project.solver.cell_size_mm = 0.5;
        self.project.solver.reference_speed = 12.0;
        self.project.solver.max_steps = 4000;
        self.project.domain.padding_mm = 4.0;
        let duct = shapes::hollow_tube(10.0, 6.0, 40.0);
        let nozzle = shapes::solid_box([46.0, -5.0, -5.0], [56.0, 5.0, 5.0]);
        let bodies = vec![
            Body { name: "duct".into(), role: BodyRole::Duct, mesh: duct },
            Body { name: "nozzle".into(), role: BodyRole::Solid, mesh: nozzle },
        ];
        self.set_bodies(bodies);
        self.define_duct_from(&[0], "duct");
        self.project.sources.push(Source { name: "fan".into(), kind: SourceKind::Velocity { speed: 8.0 }, split: SplitMode::ByArea, preset: None, streamlines: 40 });
        self.add_port_from(0, "inlet_1", PortKind::SourceInlet { source: "fan".into(), weight: 1.0 });
        self.add_port_from(1, "outlet_1", PortKind::PressureOutlet { pressure: 0.0 });
        self.status = "Demo loaded: straight duct with a nozzle block. Press Simulation > Run.".into();
    }

    /// A duct whose exit (4 mm) is much smaller than its inlet (10 mm), fed by a small blower: the restriction chokes the
    /// source, so the pressure inside rises, the exit jet is fast, and the delivered flow drops.
    fn load_restriction_sample(&mut self) {
        self.reset_history();
        self.project = Project::new("restriction sample");
        self.project.solver.cell_size_mm = 0.4;
        self.project.solver.reference_speed = 0.0;
        self.project.solver.max_steps = 5000;
        self.project.domain.padding_mm = 4.0;
        let main = shapes::hollow_tube(14.0, 10.0, 30.0);
        let mut tip = shapes::hollow_tube(14.0, 4.0, 10.0);
        tip.translate([30.0, 0.0, 0.0]);
        self.set_bodies(vec![
            Body { name: "duct_main (10 mm bore)".into(), role: BodyRole::Duct, mesh: main },
            Body { name: "exit_tip (4 mm bore)".into(), role: BodyRole::Duct, mesh: tip },
        ]);
        self.define_duct_from(&[0, 1], "duct");
        let curve = vec![(0.0, 300.0), (5.0, 280.0), (10.0, 230.0), (15.0, 150.0), (20.0, 0.0)];
        self.project.sources.push(Source { name: "blower".into(), kind: SourceKind::Fan { curve }, split: SplitMode::ByArea, preset: None, streamlines: 60 });
        // the end openings: the far-left facing one and the far-right facing one (not the joint between the parts)
        let ends = |sign: f64| self.candidates.iter().enumerate().filter(|(_, (_, c))| c.cap.normal[0] * sign > 0.5).min_by(|a, b| (sign * -a.1 .1.cap.origin[0]).total_cmp(&(sign * -b.1 .1.cap.origin[0]))).map(|(i, _)| i);
        let inlet = ends(-1.0);
        let outlet = ends(1.0);
        if let Some(i) = inlet { self.add_port_from(i, "inlet_1", PortKind::SourceInlet { source: "blower".into(), weight: 1.0 }); }
        if let Some(o) = outlet { self.add_port_from(o, "outlet_1", PortKind::PressureOutlet { pressure: 0.0 }); }
        self.mesh_mode = MeshMode::Ghost;
        self.status = "Restriction sample loaded: 10 mm duct into a 4 mm exit. Run it, then colour the streamlines by Pressure and compare with a wider exit.".into();
    }

    fn open_geometry(&mut self, path: PathBuf) {
        match load_bodies(&path, &self.project) {
            Ok(b) => {
                self.reset_history();
                let n = b.len();
                self.project.geometry_file = Some(path.clone());
                if self.project.name.starts_with("untitled") {
                    if let Some(stem) = path.file_stem().and_then(|x| x.to_str()) { self.project.name = stem.to_string(); }
                }
                self.set_bodies(b);
                self.tab = Tab::Bodies;
                let hints = self.suggested_bodies().len();
                self.status = format!("Loaded {n} bodies. Next: pick the duct part(s) in the Bodies tab or click them in the 3D view, then 'Define as duct'{}.", if hints > 0 { format!(" ({hints} likely duct part(s) are highlighted)") } else { String::new() });
            }
            Err(e) => self.status = format!("Import failed: {e:#}"),
        }
    }

    fn add_port_from(&mut self, cand: usize, name: &str, kind: PortKind) {
        if let Some((body, c)) = self.candidates.get(cand).cloned() {
            self.project.ports.retain(|p| p.name != name);
            self.project.ports.push(Port { name: name.into(), body, kind, cap: c.cap });
        }
    }

    fn save_project(&mut self) {
        let path = self.project_path.clone().or_else(|| rfd::FileDialog::new().add_filter("Peregrine project", &["toml"]).set_file_name("project.toml").save_file());
        if let Some(p) = path {
            match self.project.save(&p) {
                Ok(_) => { self.status = format!("Saved {}", p.display()); self.project_path = Some(p); }
                Err(e) => self.status = format!("Save failed: {e:#}"),
            }
        }
    }

    fn open_project(&mut self) {
        if let Some(p) = rfd::FileDialog::new().add_filter("Peregrine project", &["toml"]).pick_file() {
            self.load_project_file(p);
        }
    }

    fn load_project_file(&mut self, p: PathBuf) {
        match Project::load(&p) {
            Ok(pr) => {
                self.reset_history();
                self.project = pr;
                self.project_path = Some(p.clone());
                if let Some(g) = self.project.geometry_file.clone() {
                    let full = if g.is_absolute() { g } else { p.parent().unwrap_or(std::path::Path::new(".")).join(g) };
                    match load_bodies(&full, &self.project) {
                        Ok(b) => {
                            // projects saved before ducts were user-defined relied on name rules: keep them working
                            if self.project.ducts.is_empty() && !self.project.ports.is_empty() {
                                let names: Vec<String> = b.iter().map(|x| x.name.clone()).collect();
                                self.project.ducts = self.project.effective_ducts(&names);
                            }
                            self.set_bodies(b);
                        }
                        Err(e) => self.status = format!("Geometry: {e:#}"),
                    }
                }
                self.status = format!("Opened {}", p.display());
            }
            Err(e) => self.status = format!("Open failed: {e:#}"),
        }
    }

    fn can_run(&self) -> Result<(), String> {
        if self.bodies.is_empty() { return Err("no geometry loaded".into()); }
        let problems = self.project.validate();
        if !problems.is_empty() { return Err(problems.join("; ")); }
        // the grid must fit on the graphics card
        if self.backend != Backend::Cpu {
            let b = project_bounds(&self.project, &self.bodies);
            let dx = self.project.solver.cell_size_mm;
            let n: f64 = (0..3).map(|i| ((b.max[i] - b.min[i]) / dx).ceil().max(1.0)).product();
            let gpu_gb = n * 172.0 / 1e9;
            let budget = gpu_budget_gb();
            if gpu_gb > budget {
                let ok_dx = dx * (gpu_gb / (budget * 0.93)).cbrt();
                return Err(format!("a {dx:.2} mm grid is {:.0} M cells and needs about {gpu_gb:.1} GB of graphics memory (your card allows about {budget:.1} GB). Use a cell size of {ok_dx:.2} mm or larger, or crop the model.", n / 1e6));
            }
        }
        Ok(())
    }

    fn start_run(&mut self) {
        if let Err(e) = self.can_run() { self.status = format!("Cannot run: {e}"); return; }
        let project = self.project.clone();
        let bodies = self.bodies.clone();
        let backend = self.backend;
        let stop = Arc::new(AtomicBool::new(false));
        let snap = Arc::new(Mutex::new(Snapshot { total: project.solver.max_steps, ..Default::default() }));
        let (stop2, snap2) = (stop.clone(), snap.clone());
        std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<()> {
                let domain = build_project_domain(&project, &bodies)?;
                let cells = Arc::new(domain.cells.clone());
                let mut base_warnings = presets::preset_warnings(&project);
                if let Some(w) = peregrine_geom::leak_warning(&project, &domain) { base_warnings.push(w); }
                if let Some(w) = peregrine_geom::repair_note(&domain) { base_warnings.push(w); }
                if let Some(w) = peregrine_geom::review_note(&domain) { base_warnings.push(w); }
                if let Some(w) = peregrine_geom::thin_wall_note(&domain) { base_warnings.push(w); }
                let sealed_regions = peregrine_geom::sealed_regions(&domain);
                if domain.odd_columns > 0 {
                    base_warnings.push(format!("{} ray columns crossed a surface an odd number of times; a body may not be watertight.", domain.odd_columns));
                }
                {
                    let mut s = snap2.lock().unwrap();
                    s.dims = domain.grid.dims;
                    s.origin = domain.grid.origin;
                    s.dx = domain.grid.dx;
                    s.cells = cells.clone();
                }
                let total = project.solver.max_steps;
                let mut ref_speed = project.solver.reference_speed;
                let mut attempt = 0;
                let mut notes: Vec<String> = vec![];
                'attempts: loop {
                    let mut sim = Simulation::new_with_ref(&project, &domain, backend, ref_speed)?;
                    {
                        let mut s = snap2.lock().unwrap();
                        s.engine = sim.engine_name();
                        s.inflation = sim.units.viscosity_inflation();
                        s.ref_speed = sim.ref_speed;
                        s.step = 0;
                        s.sealed = sealed_regions.clone();
                        s.warnings = base_warnings.clone();
                        s.warnings.extend(notes.clone());
                        if sim.ref_speed > ref_speed * 1.01 {
                            s.warnings.push(format!("Reference speed raised to {:.1} m/s: the estimated peak flow speed is higher than the project setting.", sim.ref_speed));
                        }
                        s.warnings.extend(sim.units.warnings());
                        if let Some(why) = &sim.gpu_fallback {
                            s.warnings.push(format!("The graphics card was NOT used: {why}. This run is on the CPU, which is far slower and needs system memory (about 170 bytes per cell). Use a larger cell size or a smaller box so the grid fits on the card."));
                        }
                    }
                    let mut done = 0;
                    // Steps between display updates. Each update reads the field back and rebuilds the speed / pressure /
                    // velocity arrays on the CPU, which leaves the graphics card idle: when that takes a big share of the
                    // time, update less often (and more often again if the updates become too sparse).
                    let mut per = 250usize;
                    while done < total && !stop2.load(Ordering::Relaxed) {
                        let t_iter = std::time::Instant::now();
                        let n = per.min(total - done);
                        sim.advance(n);
                        done += n;
                        if sim.unstable() {
                            let spot = sim.hotspot();
                            attempt += 1;
                            if attempt > 2 {
                                anyhow::bail!("solution became unstable at step {done} even with reference speed {:.0} m/s. Fastest cell: {spot}. Check the geometry for tiny gaps, reduce the flow, or refine the cell size.", sim.ref_speed);
                            }
                            ref_speed = sim.ref_speed * 1.6;
                            notes.push(format!("Run became unstable at step {done} (fastest cell {spot}); restarted with reference speed {ref_speed:.1} m/s (attempt {attempt})."));
                            continue 'attempts;
                        }
                        let u = sim.units;
                        let report = sim.report();
                        let sources = sim.source_reports();
                        // use the read-back field in place: a copy of it would be another full-size array in memory
                        let mac = sim.mac();
                        let t_post = std::time::Instant::now();
                        let solid = |i: usize| cells[i] == 1;
                        let speed: Vec<f32> = mac.iter().enumerate().map(|(i, m)| if solid(i) { 0.0 } else { u.vel_to_phys((m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() as f64) as f32 }).collect();
                        let pressure: Vec<f32> = mac.iter().enumerate().map(|(i, m)| if solid(i) { 0.0 } else { u.pressure(m[0] as f64) as f32 }).collect();
                        let vel: Vec<[f32; 3]> = mac.iter().enumerate().map(|(i, m)| if solid(i) { [0.0; 3] } else { [u.vel_to_phys(m[1] as f64) as f32, u.vel_to_phys(m[2] as f64) as f32, u.vel_to_phys(m[3] as f64) as f32] }).collect();
                        let mut s = snap2.lock().unwrap();
                        s.sources = sources;
                        s.step = done;
                        s.speed = speed;
                        s.pressure = pressure;
                        s.vel = Arc::new(vel);
                        s.report = report;
                        s.version += 1;
                        drop(s);
                        let total_t = t_iter.elapsed().as_secs_f64();
                        let post_t = t_post.elapsed().as_secs_f64();
                        if post_t > 0.15 * total_t && per < 4000 { per *= 2; } else if total_t > 8.0 && per > 250 { per /= 2; }
                    }
                    break;
                }
                Ok(())
            })();
            let mut s = snap2.lock().unwrap();
            s.finished = true;
            if let Err(e) = result { s.error = Some(format!("{e:#}")); }
        });
        self.run = Some(RunHandle { stop, snap: snap.clone() });
        self.last = Some(snap);
        self.tab = Tab::Results;
        self.lines.clear();
        self.lines_request_sig = 0;
        if self.mesh_mode == MeshMode::Shaded { self.mesh_mode = MeshMode::Ghost; }
        self.status = "Running...".into();
    }

    fn lines_signature(&self, version: u64) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        version.hash(&mut h);
        for s in &self.project.sources { (&s.name, s.streamlines).hash(&mut h); }
        for p in &self.project.ports {
            p.name.hash(&mut h);
            p.kind.source().hash(&mut h);
            for c in p.cap.centroid() { c.to_bits().hash(&mut h); }
        }
        h.finish().max(1)
    }

    /// Collect a finished streamline job and start a new one when the data or line counts changed.
    fn update_lines(&mut self) {
        if let Some(rx) = &self.lines_job {
            if let Ok((_, lines)) = rx.try_recv() {
                let mut speeds: Vec<f32> = lines.iter().flat_map(|l| l.speed.iter().copied()).collect();
                speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());
                self.lines_vmax = speeds.get(speeds.len().saturating_sub(1) * 98 / 100).copied().unwrap_or(1.0).max(0.1);
                let mut pr: Vec<f32> = lines.iter().flat_map(|l| l.pressure.iter().copied()).collect();
                pr.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if pr.len() > 10 {
                    let n = pr.len() - 1;
                    self.lines_prange = (pr[n * 2 / 100], pr[n * 98 / 100].max(pr[n * 2 / 100] + 1e-3));
                }
                self.lines = lines;
                self.lines_job = None;
            }
        }
        let Some(snap) = self.last.clone() else { return };
        let (version, ready) = { let s = snap.lock().unwrap(); (s.version, !s.vel.is_empty()) };
        if !ready { return; }
        let sig = self.lines_signature(version);
        if self.lines_job.is_some() || sig == self.lines_request_sig { return; }
        self.lines_request_sig = sig;
        // seeds: each source spreads its line count over its inlet openings in proportion to area
        let mut specs: Vec<SeedSpec> = vec![];
        for (si, src) in self.project.sources.iter().enumerate() {
            let members = self.project.members(&src.name);
            if src.streamlines == 0 || members.is_empty() { continue; }
            let areas: Vec<f64> = members.iter().map(|&k| self.project.ports[k].cap.area()).collect();
            let total: f64 = areas.iter().sum::<f64>().max(1e-9);
            let caps = members.iter().zip(&areas).map(|(&k, a)| (self.project.ports[k].cap.clone(), ((src.streamlines as f64 * a / total).round() as usize).max(1))).collect();
            specs.push(SeedSpec { source: si, caps });
        }
        let n_src = self.project.sources.len();
        for (j, p) in self.project.ports.iter().enumerate() {
            if matches!(p.kind, PortKind::VelocityInlet { .. } | PortKind::FlowInlet { .. } | PortKind::FanInlet { .. }) {
                specs.push(SeedSpec { source: n_src + j, caps: vec![(p.cap.clone(), 12)] });
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.lines_job = Some(rx);
        std::thread::spawn(move || {
            let (dims, origin, dx, cells, vel, pressure) = { let s = snap.lock().unwrap(); (s.dims, s.origin, s.dx, s.cells.clone(), s.vel.clone(), s.pressure.clone()) };
            let field = VelocityField { dims, origin, dx, cells: &cells, vel: &vel };
            let mut lines = sl::streamlines(&field, &specs, 600.0);
            // pressure (gauge, Pa) in the cell each point sits in
            for l in &mut lines {
                l.pressure = l.pts.iter().map(|p| {
                    let c: Vec<usize> = (0..3).map(|k| (((p[k] as f64 - origin[k]) / dx).floor().max(0.0) as usize).min(dims[k] - 1)).collect();
                    pressure.get(c[0] + dims[0] * (c[1] + dims[1] * c[2])).copied().unwrap_or(0.0)
                }).collect();
            }
            let _ = tx.send((sig, lines));
        });
    }

    fn running(&self) -> bool {
        self.run.as_ref().map_or(false, |r| !r.snap.lock().unwrap().finished)
    }

    // ---------- UI pieces ----------

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("New Simulation Tab   Ctrl+T").clicked() { self.cmds.push(ShellCmd::New); ui.close_menu(); }
                if ui.button("Duplicate This Tab").on_hover_text("Copy geometry, ports and settings into a new tab to try a variation").clicked() { self.cmds.push(ShellCmd::Duplicate); ui.close_menu(); }
                if ui.button("Close Tab   Ctrl+W").clicked() { self.cmds.push(ShellCmd::Close(self.tab_active)); ui.close_menu(); }
                ui.separator();
                if ui.button("Open Geometry (STL file)...").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Geometry", &["stl", "step", "stp"]).pick_file() { self.open_geometry(p); }
                    ui.close_menu();
                }
                if ui.button("Open Geometry Folder (one STL per body)...").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() { self.open_geometry(p); }
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Open Project...").clicked() { self.open_project(); ui.close_menu(); }
                if ui.button("Save Project").clicked() { self.save_project(); ui.close_menu(); }
                ui.separator();
                if ui.button("Load Demo").clicked() { self.load_demo(); ui.close_menu(); }
                if ui.button("Load Restriction Sample").on_hover_text("A duct that narrows to a small exit, to see pressure build up and flow drop").clicked() { self.load_restriction_sample(); ui.close_menu(); }
                if ui.button("Quit").clicked() { ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close); }
            });
            ui.menu_button("Edit", |ui| {
                if ui.button("Clear all ports").clicked() { self.project.ports.clear(); ui.close_menu(); }
            });
            ui.menu_button("View", |ui| {
                ui.label("Mesh");
                ui.radio_value(&mut self.mesh_mode, MeshMode::Shaded, "Shaded");
                ui.radio_value(&mut self.mesh_mode, MeshMode::Ghost, "Ghost (see-through)");
                ui.radio_value(&mut self.mesh_mode, MeshMode::Hidden, "Hidden (flow only)");
                ui.checkbox(&mut self.show_lines, "Streamlines");
                ui.separator();
                if ui.button("Frame all").clicked() {
                    let b = domain_bounds(&self.bodies, 0.0);
                    if b.is_valid() { self.cam.frame(b.min, b.max); }
                    ui.close_menu();
                }
            });
            ui.menu_button("Geometry", |ui| {
                if ui.button("Select suggested parts").clicked() { self.sel_bodies = self.suggested_bodies(); self.tab = Tab::Bodies; ui.close_menu(); }
                if ui.button("Re-detect port candidates").clicked() { self.refresh_candidates(); ui.close_menu(); }
            });
            ui.menu_button("Simulation", |ui| {
                if ui.button("Check for leaks").clicked() { self.check_leaks(); self.tab = Tab::Exceptions; ui.close_menu(); }
                let running = self.running();
                if ui.add_enabled(!running, egui::Button::new("Run")).clicked() { self.start_run(); ui.close_menu(); }
                if ui.add_enabled(running, egui::Button::new("Stop")).clicked() {
                    if let Some(r) = &self.run { r.stop.store(true, Ordering::Relaxed); }
                    ui.close_menu();
                }
            });
            ui.menu_button("Window", |ui| {
                for (t, n) in [(Tab::Bodies, "Bodies"), (Tab::Ports, "Ports"), (Tab::Results, "Results")] {
                    if ui.button(n).clicked() { self.tab = t; ui.close_menu(); }
                }
            });
            ui.menu_button("Help", |ui| {
                ui.label("Peregrine CFD - duct airflow (LBM)");
                ui.label("Left-drag orbit, right/middle-drag pan, wheel zoom.");
                ui.separator();
                if ui.button("Run tutorial").on_hover_text("Opens a sample duct in a new tab and walks through every major feature").clicked() { self.cmds.push(ShellCmd::Tutorial); ui.close_menu(); }
                ui.label("Shortcuts: O orbit, H pan, R curved opening, L draw loop, B blocker, M mesh mode, Ctrl+Z undo.");
            });
        });
    }

    fn options_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.label(egui::RichText::new(match self.tool { Tool::Orbit => "Orbit", Tool::Pan => "Pan", Tool::PickPort => "Port Pick", Tool::Slice => "Slice", Tool::Rim => "Curved Opening", Tool::Draw => "Draw Loop", Tool::Block => "Blocker" }).strong());
            ui.separator();
            ui.label("Cell size");
            ui.add(egui::DragValue::new(&mut self.project.solver.cell_size_mm).speed(0.01).range(0.05..=2.0).suffix(" mm"));
            ui.label("Ref. speed");
            ui.add(egui::DragValue::new(&mut self.project.solver.reference_speed).speed(0.1).range(0.0..=100.0).suffix(" m/s").custom_formatter(|v, _| if v <= 0.0 { "auto".to_string() } else { format!("{v:.1}") }).custom_parser(|t| if t.trim().eq_ignore_ascii_case("auto") { Some(0.0) } else { t.trim().trim_end_matches("m/s").trim().parse().ok() })).on_hover_text("Largest expected air speed. 'auto' picks it from your sources; lower values reduce artificial viscosity but need a stable run.");
            ui.label("Steps");
            ui.add(egui::DragValue::new(&mut self.project.solver.max_steps).speed(50.0).range(250..=500_000));
            ui.label("Padding");
            ui.add(egui::DragValue::new(&mut self.project.domain.padding_mm).speed(0.2).range(0.0..=100.0).suffix(" mm"));
            ui.label("Outlet room");
            ui.add(egui::DragValue::new(&mut self.project.domain.outlet_clearance_mm).speed(0.5).range(0.0..=200.0).suffix(" mm")).on_hover_text("The simulation box extends this far beyond every outlet so jets can develop and converge.");
            ui.label("Min tau");
            ui.add(egui::DragValue::new(&mut self.project.solver.min_tau).speed(0.001).range(0.5005..=0.7).fixed_decimals(3)).on_hover_text("Lowest relaxation time. Lower = less artificial viscosity (more realistic jets) but less stable. The solver retries automatically if a run becomes unstable.");
            ui.separator();
            egui::ComboBox::from_id_salt("backend").selected_text(match self.backend { Backend::Auto => "Auto (GPU if found)", Backend::Cpu => "CPU", Backend::Gpu => "GPU" }).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.backend, Backend::Auto, "Auto (GPU if found)");
                ui.selectable_value(&mut self.backend, Backend::Gpu, "GPU");
                ui.selectable_value(&mut self.backend, Backend::Cpu, "CPU");
            });
            let running = self.running();
            if running {
                if ui.button("Stop").clicked() { if let Some(r) = &self.run { r.stop.store(true, Ordering::Relaxed); } }
            } else if ui.add(egui::Button::new(egui::RichText::new("Run").color(Color32::WHITE)).fill(theme::ACCENT)).clicked() {
                self.start_run();
            }
        });
    }

    fn tool_palette(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.vertical_centered(|ui| {
            for (t, tip) in [(Tool::Orbit, "Orbit (O)"), (Tool::Pan, "Pan (H)"), (Tool::PickPort, "Port pick (P)"), (Tool::Rim, "Curved opening: click the rim or the wall at a non-flat opening (R)"), (Tool::Draw, "Draw an edge loop by hand: click points along the rim (L)"), (Tool::Block, "Blocker: click a small hole to close it, or draw a loop and add it as a blocker (B)"), (Tool::Slice, "Slice (S)")] {
                let (rect, resp) = ui.allocate_exact_size(Vec2::splat(30.0), Sense::click());
                let selected = self.tool == t;
                let bg = if selected { theme::ACCENT } else if resp.hovered() { theme::WIDGET_HOVER } else { theme::WIDGET };
                ui.painter().rect_filled(rect, 2.0, bg);
                draw_tool_icon(ui.painter(), rect, t);
                if resp.on_hover_text(tip).clicked() { self.tool = t; }
                ui.add_space(2.0);
            }
        });
    }

    fn bodies_panel(&mut self, ui: &mut egui::Ui) {
        let sel = self.sel_bodies.clone();
        theme::section(ui, "Parts", |ui| {
            if self.bodies.is_empty() {
                ui.label(egui::RichText::new("No geometry. File > Open Geometry...").color(theme::TEXT_DIM));
                return;
            }
            if self.project.ducts.is_empty() {
                ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), "No duct defined yet. Select the duct part(s) here or click them in the 3D view (Ctrl-click for several), then press 'Define as duct'.");
            } else {
                ui.label(egui::RichText::new("Click a part in the 3D view to find it here. Ctrl-click selects several.").color(theme::TEXT_DIM));
            }
            let sugg = self.suggested_bodies();
            if !sugg.is_empty() {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("★ {} part(s) look like ducts", sugg.len())).color(Color32::from_rgb(0xE8, 0xB8, 0x4A)));
                    if ui.small_button("Select them").clicked() { self.sel_bodies = sugg.clone(); }
                });
            }
            let mut clicked: Option<(usize, bool)> = None;
            for i in 0..self.bodies.len() {
                let name = self.bodies[i].name.clone();
                let selected = sel.contains(&i);
                let suggested = self.project.is_suggested(&name);
                let role = self.bodies[i].role;
                let duct = self.project.duct_of(&name).map(|d| d.name.clone());
                let row = ui.horizontal(|ui| {
                    let eye = if self.visible[i] { "o" } else { "-" };
                    if ui.small_button(eye).on_hover_text("Visibility").clicked() { self.visible[i] = !self.visible[i]; }
                    let color = if suggested { Color32::from_rgb(0xE8, 0xB8, 0x4A) } else { theme::TEXT };
                    let r = ui.selectable_label(selected, egui::RichText::new(&name).color(color));
                    let badge = match (role, &duct) {
                        (BodyRole::Duct, Some(d)) => egui::RichText::new(format!("duct: {d}")).color(Color32::from_rgb(0x8F, 0xB4, 0xD9)),
                        (BodyRole::Ignore, _) => egui::RichText::new("ignored").color(theme::TEXT_DIM),
                        _ if suggested => egui::RichText::new("★ suggested").color(Color32::from_rgb(0xE8, 0xB8, 0x4A)),
                        _ => egui::RichText::new("solid").color(theme::TEXT_DIM),
                    };
                    ui.label(badge.size(10.5));
                    r
                });
                if self.scroll_to_body == Some(i) { row.response.scroll_to_me(Some(egui::Align::Center)); }
                if row.inner.clicked() { clicked = Some((i, ui.input(|x| x.modifiers.ctrl || x.modifiers.command))); }
            }
            self.scroll_to_body = None;
            if let Some((i, add)) = clicked { self.select_body(i, add); }
        });
        if !self.sel_bodies.is_empty() {
            theme::section(ui, &format!("Selection ({})", self.sel_bodies.len()), |ui| {
                let sel = self.sel_bodies.clone();
                ui.horizontal(|ui| {
                    ui.label("Duct name");
                    ui.add(egui::TextEdit::singleline(&mut self.duct_name_buf).desired_width(120.0));
                });
                ui.horizontal_wrapped(|ui| {
                    if ui.add(egui::Button::new(egui::RichText::new("Define as duct").color(Color32::WHITE)).fill(theme::ACCENT)).on_hover_text("Treat the selected parts as ONE duct (use several parts for a duct made of multiple pieces)").clicked() {
                        let name = self.duct_name_buf.clone();
                        self.define_duct_from(&sel, &name);
                    }
                    if !self.project.ducts.is_empty() {
                        ui.menu_button("Add to duct", |ui| {
                            let names: Vec<String> = self.project.ducts.iter().map(|d| d.name.clone()).collect();
                            for n in names { if ui.button(&n).clicked() { self.add_to_duct(&sel, &n); ui.close_menu(); } }
                        });
                    }
                    if ui.button("Solid").on_hover_text("Obstacle that blocks air (nozzle, heater block, fan housing...)").clicked() { self.remove_from_ducts(&sel); }
                    if ui.button("Ignore").on_hover_text("Leave out of the simulation").clicked() { self.ignore_bodies(&sel); }
                });
            });
        }
        theme::section(ui, "Ducts", |ui| {
            if self.project.ducts.is_empty() {
                ui.label(egui::RichText::new("None yet.").color(theme::TEXT_DIM));
            }
            let mut rename: Option<(String, String)> = None;
            let mut delete: Option<String> = None;
            let mut remove_part: Option<(usize, String)> = None;
            for di in 0..self.project.ducts.len() {
                let name = self.project.ducts[di].name.clone();
                let parts = self.project.ducts[di].parts.clone();
                egui::Frame::group(ui.style()).fill(theme::DEEP).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let id = ui.make_persistent_id(("ductname", di));
                        let mut buf: String = ui.data_mut(|d| d.get_temp(id)).unwrap_or_else(|| name.clone());
                        let r = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(120.0));
                        if r.lost_focus() {
                            let nb = buf.trim().to_string();
                            if !nb.is_empty() && nb != name && !self.project.ducts.iter().any(|d| d.name == nb) { rename = Some((name.clone(), nb)); }
                            buf = name.clone();
                        } else if !r.has_focus() { buf = name.clone(); }
                        ui.data_mut(|d| d.insert_temp(id, buf));
                        ui.label(egui::RichText::new(format!("{} part(s)", parts.len())).color(theme::TEXT_DIM));
                        if ui.small_button("x").on_hover_text("Delete duct (parts become solids)").clicked() { delete = Some(name.clone()); }
                    });
                    for p in &parts {
                        ui.horizontal(|ui| {
                            ui.label(format!("  {p}"));
                            if ui.small_button("-").on_hover_text("Remove this part from the duct").clicked() { remove_part = Some((di, p.clone())); }
                        });
                    }
                });
            }
            if let Some((o, n)) = rename {
                for d in &mut self.project.ducts { if d.name == o { d.name = n.clone(); } }
                for p in &mut self.project.ports { if p.body == o { p.body = n.clone(); } }
                self.refresh_candidates();
            }
            if let Some((di, part)) = remove_part {
                self.project.ducts[di].parts.retain(|p| p != &part);
                self.project.ducts.retain(|d| !d.parts.is_empty());
                self.reclassify();
            }
            if let Some(n) = delete { self.project.ducts.retain(|d| d.name != n); self.reclassify(); }
        });
        theme::section(ui, "Seams between parts", |ui| {
            ui.label(egui::RichText::new("Parts of one duct that do not quite touch: gaps up to this size are sealed. Bigger gaps leak and are reported when you run.").color(theme::TEXT_DIM));
            ui.horizontal(|ui| {
                ui.label("Seal gaps up to");
                ui.add(egui::DragValue::new(&mut self.project.domain.seam_tolerance_mm).speed(0.05).range(0.0..=3.0).suffix(" mm"));
            });
        });
        theme::section(ui, "Name hints (regex)", |ui| {
            ui.label(egui::RichText::new("Parts whose name matches are highlighted as likely ducts. Only a hint; you decide.").color(theme::TEXT_DIM));
            for r in &mut self.project.geometry.rules {
                ui.add(egui::TextEdit::singleline(&mut r.pattern).desired_width(200.0).font(FontId::monospace(11.0)));
            }
        });
    }

    // ---------- sources and port assignment ----------

    fn port_for_cand(&self, i: usize) -> Option<usize> {
        let (_, c) = self.candidates.get(i)?;
        let ctr = c.cap.centroid();
        self.project.ports.iter().position(|p| len(sub(p.cap.centroid(), ctr)) < 0.05 && dot(p.cap.normal, c.cap.normal) > 0.99)
    }

    fn resolve(&self, t: Target) -> Target {
        if let Target::Cand(i) = t {
            if let Some(p) = self.port_for_cand(i) { return Target::Port(p); }
        }
        t
    }

    fn target_cap(&self, t: Target) -> Option<&PortCap> {
        match t {
            Target::Cand(i) => self.candidates.get(i).map(|c| &c.1.cap),
            Target::Port(i) => self.project.ports.get(i).map(|p| &p.cap),
        }
    }

    fn select_target(&mut self, t: Target) {
        match self.resolve(t) {
            Target::Cand(i) => { self.sel_cand = Some(i); self.sel_port = None; }
            Target::Port(i) => { self.sel_port = Some(i); self.sel_cand = (0..self.candidates.len()).find(|&c| self.port_for_cand(c) == Some(i)); }
        }
    }

    fn unique_name(taken: impl Fn(&str) -> bool, prefix: &str) -> String {
        (1..).map(|n| format!("{prefix}_{n}")).find(|n| !taken(n)).unwrap()
    }

    fn ensure_port(&mut self, t: Target, prefix: &str, kind: PortKind) -> Option<usize> {
        match self.resolve(t) {
            Target::Port(i) => { self.project.ports[i].kind = kind; Some(i) }
            Target::Cand(i) => {
                let (body, c) = self.candidates.get(i)?.clone();
                let name = Self::unique_name(|n| self.project.ports.iter().any(|p| p.name == n), prefix);
                self.project.ports.push(Port { name, body, kind, cap: c.cap });
                Some(self.project.ports.len() - 1)
            }
        }
    }

    fn assign_to_source(&mut self, t: Target, source: &str) {
        if let Some(i) = self.ensure_port(t, "inlet", PortKind::SourceInlet { source: source.into(), weight: 1.0 }) {
            self.sel_port = Some(i);
        }
    }

    fn make_outlet(&mut self, t: Target) {
        if let Some(i) = self.ensure_port(t, "outlet", PortKind::PressureOutlet { pressure: 0.0 }) { self.sel_port = Some(i); }
    }

    fn add_source(&mut self, kind: SourceKind, prefix: &str, preset: Option<String>) -> String {
        let name = Self::unique_name(|n| self.project.sources.iter().any(|s| s.name == n), prefix);
        self.project.sources.push(Source { name: name.clone(), kind, split: SplitMode::ByArea, preset, streamlines: 40 });
        name
    }

    fn add_fan_preset(&mut self, p: &FanPreset) -> String {
        self.add_source(SourceKind::Fan { curve: p.curve() }, "fan", Some(p.name.to_string()))
    }

    fn add_default_source(&mut self, which: u8) -> String {
        match which {
            0 => self.add_source(SourceKind::Fan { curve: vec![(0.0, 300.0), (20.0, 200.0), (40.0, 0.0)] }, "fan", None),
            1 => self.add_source(SourceKind::Flow { l_per_min: 10.0 }, "flow", None),
            _ => self.add_source(SourceKind::Velocity { speed: 5.0 }, "velocity", None),
        }
    }

    fn rename_source(&mut self, old: &str, new: &str) {
        for s in &mut self.project.sources { if s.name == old { s.name = new.to_string(); } }
        for p in &mut self.project.ports {
            if let PortKind::SourceInlet { source, .. } = &mut p.kind { if source == old { *source = new.to_string(); } }
        }
    }

    fn remove_source(&mut self, name: &str) {
        self.project.ports.retain(|p| p.kind.source() != Some(name));
        self.project.sources.retain(|s| s.name != name);
        self.sel_port = None;
    }

    /// The visible part under the screen position (front-most triangle).
    fn pick_body(&self, rect: Rect, pos: Pos2) -> Option<usize> {
        self.pick_body_tri(rect, pos).map(|x| x.0)
    }

    /// The body and triangle under a screen point: a ray from the eye against the visible meshes (the nearest hit).
    /// The answer is remembered while neither the pointer nor the camera moves, so hovering costs nothing.
    fn pick_body_tri(&self, rect: Rect, pos: Pos2) -> Option<(usize, usize)> {
        let key_cam = [self.cam.target[0].to_bits() ^ self.cam.target[1].to_bits().rotate_left(13) ^ self.cam.target[2].to_bits().rotate_left(27), self.cam.dist.to_bits(), self.cam.yaw.to_bits(), self.cam.pitch.to_bits() ^ (self.visible.iter().filter(|v| **v).count() as u64)];
        let key_pos = [pos.x.round() as i32, pos.y.round() as i32];
        if let Some((kp, kc, res)) = &*self.pick_cache.borrow() {
            if *kp == key_pos && *kc == key_cam { return *res; }
        }
        let (o, d) = self.cam.ray(rect, pos);
        let inv = [1.0 / if d[0].abs() < 1e-12 { 1e-12 } else { d[0] }, 1.0 / if d[1].abs() < 1e-12 { 1e-12 } else { d[1] }, 1.0 / if d[2].abs() < 1e-12 { 1e-12 } else { d[2] }];
        let mut best: Option<(f64, usize, usize)> = None;
        for (bi, b) in self.bodies.iter().enumerate() {
            if !self.visible[bi] || b.role == BodyRole::Ignore { continue; }
            // ray against the body's bounding box first
            let bb = b.mesh.bounds();
            let (mut t0, mut t1) = (0.0f64, f64::MAX);
            for k in 0..3 {
                let (a, c) = ((bb.min[k] - o[k]) * inv[k], (bb.max[k] - o[k]) * inv[k]);
                t0 = t0.max(a.min(c));
                t1 = t1.min(a.max(c));
            }
            if t0 > t1 || best.map_or(false, |x| t0 > x.0) { continue; }
            for ti in 0..b.mesh.triangles.len() {
                let [a, b2, c] = b.mesh.tri(ti);
                if let Some(t) = viewport::ray_triangle(o, d, a, b2, c) {
                    if best.map_or(true, |x| t < x.0) { best = Some((t, bi, ti)); }
                }
            }
        }
        let res = best.map(|x| (x.1, x.2));
        *self.pick_cache.borrow_mut() = Some((key_pos, key_cam, res));
        res
    }

    fn body_context_menu_ui(&mut self, ui: &mut egui::Ui, bi: usize) {
        let name = self.bodies[bi].name.clone();
        let sel: Vec<usize> = if self.sel_bodies.contains(&bi) { self.sel_bodies.clone() } else { vec![bi] };
        ui.label(egui::RichText::new(format!("Part '{}'{}", name, if sel.len() > 1 { format!("  (+{} selected)", sel.len() - 1) } else { String::new() })).strong());
        if let Some(d) = self.project.duct_of(&name) { ui.label(egui::RichText::new(format!("in duct '{}'", d.name)).color(theme::TEXT_DIM)); }
        ui.separator();
        if ui.button("Define as new duct").clicked() {
            let default_name = self.duct_name_buf.clone();
            self.define_duct_from(&sel, if default_name.is_empty() { &name } else { &default_name });
            ui.close_menu();
        }
        if !self.project.ducts.is_empty() {
            ui.menu_button("Add to duct", |ui| {
                let names: Vec<String> = self.project.ducts.iter().map(|d| d.name.clone()).collect();
                for n in names { if ui.button(&n).clicked() { self.add_to_duct(&sel, &n); ui.close_menu(); } }
            });
        }
        if self.project.duct_of(&name).is_some() && ui.button("Remove from duct").clicked() { self.remove_from_ducts(&sel); ui.close_menu(); }
        if ui.button("Solid obstacle").clicked() { self.remove_from_ducts(&sel); ui.close_menu(); }
        if ui.button("Ignore").clicked() { self.ignore_bodies(&sel); ui.close_menu(); }
        ui.separator();
        if ui.button("Show in Bodies list").clicked() { self.select_body(bi, false); ui.close_menu(); }
    }

    /// Identity of an opening for view state (hidden or not): its centre to 0.05 mm.
    fn cap_key(cap: &PortCap) -> [i64; 3] {
        let c = cap.centroid();
        [(c[0] * 20.0).round() as i64, (c[1] * 20.0).round() as i64, (c[2] * 20.0).round() as i64]
    }

    /// In the opaque (Shaded) view an opening on the far side of the part is behind the surface: do not draw or pick it.
    fn cap_visible_from_eye(&self, cap: &PortCap) -> bool {
        if self.mesh_mode != MeshMode::Shaded { return true; }
        let e = self.cam.eye();
        dot(cap.normal, sub(e, cap.centroid())) > 0.0
    }

    fn cap_hidden(&self, cap: &PortCap) -> bool {
        self.hidden_caps.contains(&Self::cap_key(cap))
    }

    fn toggle_cap_hidden(&mut self, cap: &PortCap) {
        let k = Self::cap_key(cap);
        if !self.hidden_caps.remove(&k) { self.hidden_caps.insert(k); }
    }

    /// The opening whose BOUNDARY is nearest the cursor (within a few pixels). Selecting by outline, not by area,
    /// keeps a small opening inside a big one selectable. Hidden openings are skipped.
    fn hit_test(&self, rect: Rect, pos: Pos2) -> Option<Target> {
        let mut items: Vec<(Target, &PortCap)> = vec![];
        for (i, (_, c)) in self.candidates.iter().enumerate() { items.push((Target::Cand(i), &c.cap)); }
        for (i, p) in self.project.ports.iter().enumerate() { items.push((Target::Port(i), &p.cap)); }
        let seg_dist = |a: Pos2, b: Pos2| -> f32 {
            let ab = b - a;
            let l2 = ab.length_sq();
            let t = if l2 > 0.0 { (((pos - a).x * ab.x + (pos - a).y * ab.y) / l2).clamp(0.0, 1.0) } else { 0.0 };
            (a + ab * t).distance(pos)
        };
        let mut best: Option<(f32, f32, Target)> = None; // distance to outline, screen area, target
        for (t, cap) in items {
            if self.cap_hidden(cap) || !self.cap_visible_from_eye(cap) { continue; }
            let mut loops: Vec<&[[f64; 3]]> = vec![&cap.polygon];
            for h in &cap.holes { loops.push(h); }
            let mut dmin = f32::MAX;
            let mut screen_area = 0.0f32;
            for (li, lp) in loops.iter().enumerate() {
                let pts: Vec<Option<Pos2>> = lp.iter().map(|p| self.cam.project(rect, *p).map(|x| x.0)).collect();
                if pts.len() < 3 || pts.iter().any(|p| p.is_none()) { continue; }
                let pts: Vec<Pos2> = pts.into_iter().flatten().collect();
                for k in 0..pts.len() { dmin = dmin.min(seg_dist(pts[k], pts[(k + 1) % pts.len()])); }
                if li == 0 {
                    let mut a = 0.0f32;
                    for k in 0..pts.len() { let (p, q) = (pts[k], pts[(k + 1) % pts.len()]); a += p.x * q.y - q.x * p.y; }
                    screen_area = a.abs() * 0.5;
                }
            }
            if dmin <= 8.0 {
                let better = match best {
                    None => true,
                    Some((bd, ba, _)) => dmin < bd - 1.5 || (dmin <= bd + 1.5 && screen_area < ba),
                };
                if better { best = Some((dmin, screen_area, t)); }
            }
        }
        best.map(|b| self.resolve(b.2))
    }

    fn context_menu_ui(&mut self, ui: &mut egui::Ui) {
        let Some(t0) = self.ctx_target else {
            if let Some(bi) = self.ctx_body { self.body_context_menu_ui(ui, bi); } else { ui.label("Right-click an opening to assign it, or a part to define a duct."); }
            return;
        };
        let t = self.resolve(t0);
        let title = match t {
            Target::Port(i) => format!("Port '{}'", self.project.ports[i].name),
            Target::Cand(i) => format!("Opening #{i}  ({:.1} mm²)", self.candidates[i].1.area),
        };
        ui.label(egui::RichText::new(title).strong());
        ui.separator();
        ui.menu_button("Assign to source", |ui| {
            let names: Vec<String> = self.project.sources.iter().map(|s| s.name.clone()).collect();
            for n in &names {
                if ui.button(n).clicked() { self.assign_to_source(t, n); ui.close_menu(); }
            }
            if !names.is_empty() { ui.separator(); }
            ui.menu_button("New fan source", |ui| {
                if let Some(p) = fan_library_menu(ui) { let n = self.add_fan_preset(&p); self.assign_to_source(t, &n); ui.close_menu(); }
                ui.separator();
                if ui.button("Custom fan curve").clicked() { let n = self.add_default_source(0); self.assign_to_source(t, &n); ui.close_menu(); }
            });
            if ui.button("New flow source").clicked() { let n = self.add_default_source(1); self.assign_to_source(t, &n); ui.close_menu(); }
            if ui.button("New velocity source").clicked() { let n = self.add_default_source(2); self.assign_to_source(t, &n); ui.close_menu(); }
        });
        if ui.button("Pressure outlet").clicked() { self.make_outlet(t); ui.close_menu(); }
        if ui.button("Block (close this opening)").clicked() {
            match t {
                Target::Cand(i) => { if let Some((b, c)) = self.candidates.get(i).cloned() { self.add_blocker(b, c.cap); } }
                Target::Port(i) => { let (b, c) = (self.project.ports[i].body.clone(), self.project.ports[i].cap.clone()); self.add_blocker(b, c); }
            }
            ui.close_menu();
        }
        if let Target::Port(i) = t {
            ui.separator();
            if ui.button("Show in Ports panel").clicked() { self.sel_port = Some(i); self.tab = Tab::Ports; ui.close_menu(); }
            if ui.button("Remove port").clicked() { self.project.ports.remove(i); self.sel_port = None; ui.close_menu(); }
        }
    }

    fn ports_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.add_enabled(!self.undo.is_empty() || self.last_project.as_ref().map_or(false, |(d, _)| *d != format!("{:?}", self.project)), egui::Button::new("Undo")).on_hover_text("Ctrl+Z").clicked() { self.undo_project(); }
            if ui.add_enabled(!self.redo.is_empty(), egui::Button::new("Redo")).on_hover_text("Ctrl+Shift+Z or Ctrl+Y").clicked() { self.redo_project(); }
            ui.label(egui::RichText::new(format!("{} step(s)", self.undo.len())).color(theme::TEXT_DIM));
        });
        theme::section(ui, "Sources", |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.menu_button("+ Fan", |ui| {
                    if let Some(p) = fan_library_menu(ui) { self.add_fan_preset(&p); ui.close_menu(); }
                    ui.separator();
                    if ui.button("Custom fan curve").clicked() { self.add_default_source(0); ui.close_menu(); }
                });
                if ui.button("+ Flow").clicked() { self.add_default_source(1); }
                if ui.button("+ Velocity").clicked() { self.add_default_source(2); }
            });
            if self.project.sources.is_empty() {
                ui.label(egui::RichText::new("Define a source first (a fan, a flow rate or a speed), then assign its inlet openings to it.").color(theme::TEXT_DIM));
            }
            self.sources_ui(ui);
        });
        if self.project.ducts.is_empty() {
            ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), "Define a duct first (Bodies tab) so its openings can be found.");
        }
        theme::section(ui, "Openings", |ui| self.openings_ui(ui));
        theme::section(ui, "Outlets", |ui| self.outlets_ui(ui));
        theme::section(ui, "Blockers", |ui| self.blockers_ui(ui));
    }

    fn blockers_ui(&mut self, ui: &mut egui::Ui) {
        ui.label(egui::RichText::new("Shut a hole so no air passes (screw holes, unused vents). Use the Blocker tool (B) and click a hole, or right-click any opening > Block. A hole that is not found can be drawn with the Draw Loop tool (L) and added as a blocker.").color(theme::TEXT_DIM));
        ui.horizontal(|ui| {
            ui.label("Find holes up to");
            if ui.add(egui::DragValue::new(&mut self.block_max_d).speed(0.2).range(0.5..=60.0).suffix(" mm")).changed() { self.block_stale = true; }
            if ui.button("Find").clicked() { self.refresh_block_cands(); self.tool = Tool::Block; }
        });
        if self.block_stale && !self.bodies.is_empty() && self.tool == Tool::Block { self.refresh_block_cands(); }
        let open: Vec<usize> = (0..self.block_cands.len()).filter(|&i| self.blocker_at(&self.block_cands[i].1.cap).is_none()).collect();
        if !self.block_cands.is_empty() {
            ui.horizontal(|ui| {
                ui.label(format!("{} small hole(s) open", open.len()));
                if !open.is_empty() && ui.button("Block all").clicked() {
                    for &i in &open { let (b, c) = self.block_cands[i].clone(); self.add_blocker(b, c.cap); }
                }
            });
        }
        let mut remove = None;
        for (i, b) in self.project.blockers.iter().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("x").color(Color32::from_rgb(0xE0, 0x4A, 0x4A)));
                if ui.selectable_label(self.sel_block == Some(i), format!("{}  {:.1} mm²", b.name, b.cap.area())).clicked() { self.sel_block = Some(i); }
                if ui.small_button("Remove").clicked() { remove = Some(i); }
            });
        }
        if let Some(i) = remove { self.remove_blocker(i); }
        let mut rm_plug = None;
        for (i, p) in self.project.plugs.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("o").color(Color32::from_rgb(0xE0, 0x4A, 0x4A)));
                ui.label(format!("{} ({:.1}, {:.1}, {:.1})", p.name, p.center[0], p.center[1], p.center[2]));
                ui.add(egui::DragValue::new(&mut p.radius_mm).speed(0.1).range(0.3..=20.0).prefix("r ").suffix(" mm"));
                if ui.small_button("Remove").clicked() { rm_plug = Some(i); }
            });
        }
        if let Some(i) = rm_plug { self.project.plugs.remove(i); }
    }

    fn sources_ui(&mut self, ui: &mut egui::Ui) {
        let mut remove_src: Option<String> = None;
        let mut rename: Option<(String, String)> = None;
        let mut unassign: Option<usize> = None;
        let mut replace: Option<(usize, FanPreset)> = None;
        let mut assign_sel: Option<(usize, String)> = None;
        for si in 0..self.project.sources.len() {
            let name = self.project.sources[si].name.clone();
            let members = self.project.members(&name);
            let areas: Vec<f64> = members.iter().map(|&k| self.project.ports[k].cap.area()).collect();
            let sum_a: f64 = areas.iter().sum();
            let weights: Vec<f64> = members.iter().map(|&k| match &self.project.ports[k].kind { PortKind::SourceInlet { weight, .. } => *weight, _ => 1.0 }).collect();
            let sum_w: f64 = weights.iter().sum();
            let split = self.project.sources[si].split;
            let sel_free = self.sel_cand.filter(|&c| self.port_for_cand(c).is_none());
            egui::Frame::group(ui.style()).fill(theme::DEEP).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let id = ui.make_persistent_id(("srcname", si));
                    let mut buf: String = ui.data_mut(|d| d.get_temp(id)).unwrap_or_else(|| name.clone());
                    let r = ui.add(egui::TextEdit::singleline(&mut buf).desired_width(110.0).font(FontId::proportional(12.0)));
                    if r.lost_focus() {
                        let nb = buf.trim().to_string();
                        if !nb.is_empty() && nb != name && !self.project.sources.iter().any(|s| s.name == nb) { rename = Some((name.clone(), nb)); }
                        buf = name.clone();
                    } else if !r.has_focus() {
                        buf = name.clone();
                    }
                    ui.data_mut(|d| d.insert_temp(id, buf));
                    if ui.small_button("x").on_hover_text("Delete source and unassign its inlets").clicked() { remove_src = Some(name.clone()); }
                });
                let src = &mut self.project.sources[si];
                let Source { kind, split, preset, streamlines, .. } = src;
                match kind {
                    SourceKind::Flow { l_per_min } => { ui.horizontal(|ui| { ui.label("Total flow"); flow_drag(ui, l_per_min, 0.5); }); }
                    SourceKind::Velocity { speed } => { ui.horizontal(|ui| { ui.label("Mean speed"); ui.add(egui::DragValue::new(speed).speed(0.1).range(0.0..=200.0).suffix(" m/s")); }); }
                    SourceKind::Fan { curve } => {
                        if let Some(pn) = preset.as_ref() {
                            ui.label(egui::RichText::new(format!("Library: {pn}")).color(theme::TEXT_DIM));
                            if let Some(p) = presets::find_preset(pn) {
                                if !p.verified { ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), "Estimated curve - replace with datasheet values"); }
                            }
                        }
                        ui.horizontal(|ui| {
                            ui.menu_button("Library", |ui| { if let Some(p) = fan_library_menu(ui) { replace = Some((si, p)); ui.close_menu(); } });
                            ui.label(egui::RichText::new("Fan curve: flow -> pressure").color(theme::TEXT_DIM));
                        });
                        let mut del = None;
                        let mut edited = false;
                        for (j, (q, dp)) in curve.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                edited |= flow_drag(ui, q, 0.5).changed();
                                edited |= pressure_drag(ui, dp, 5.0).changed();
                                if ui.small_button("-").clicked() { del = Some(j); }
                            });
                        }
                        if let Some(j) = del { if curve.len() > 2 { curve.remove(j); edited = true; } }
                        if ui.small_button("+ point").clicked() { let l = *curve.last().unwrap(); curve.push((l.0 + 10.0, (l.1 - 20.0).max(0.0))); edited = true; }
                        if edited { *preset = None; }
                    }
                }
                ui.horizontal(|ui| {
                    ui.label("Split");
                    egui::ComboBox::from_id_salt(("split", si)).selected_text(match split { SplitMode::ByArea => "By cross-section", SplitMode::Weights => "Manual weights", SplitMode::Plenum => "Shared plenum pressure" }).show_ui(ui, |ui| {
                        ui.selectable_value(split, SplitMode::ByArea, "By cross-section").on_hover_text("Each opening takes flow in proportion to its area (equal speed at every opening).");
                        ui.selectable_value(split, SplitMode::Weights, "Manual weights").on_hover_text("Each opening takes flow in proportion to the weight you set.");
                        ui.selectable_value(split, SplitMode::Plenum, "Shared plenum pressure").on_hover_text("All openings sit at one pressure and the flow divides by each duct's resistance (one fan over a manifold).");
                    });
                });
                ui.horizontal(|ui| {
                    ui.label("Streamlines");
                    ui.add(egui::DragValue::new(streamlines).speed(1.0).range(0..=2000)).on_hover_text("Flow trajectories started from this source's inlets (0 = none). More lines = finer detail.");
                });
                ui.label(egui::RichText::new(format!("Inlet openings ({})", members.len())).color(theme::TEXT_DIM));
                for (j, &k) in members.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(self.sel_port == Some(k), &self.project.ports[k].name).clicked() { self.sel_port = Some(k); }
                        ui.label(format!("{:.1} mm²", areas[j]));
                        match split {
                            SplitMode::ByArea => { ui.label(format!("{:.0}%", 100.0 * areas[j] / sum_a.max(1e-9))); }
                            SplitMode::Weights => {
                                if let PortKind::SourceInlet { weight, .. } = &mut self.project.ports[k].kind { ui.add(egui::DragValue::new(weight).speed(0.05).range(0.0..=1000.0).prefix("w ")); }
                                ui.label(format!("{:.0}%", 100.0 * weights[j] / sum_w.max(1e-9)));
                            }
                            SplitMode::Plenum => { ui.label(egui::RichText::new("by resistance").color(theme::TEXT_DIM)); }
                        }
                        if ui.small_button("x").on_hover_text("Remove this inlet").clicked() { unassign = Some(k); }
                    });
                }
                if let Some(c) = sel_free {
                    if ui.button(format!("Assign opening #{c} here")).clicked() { assign_sel = Some((c, name.clone())); }
                }
            });
        }
        if let Some((si, p)) = replace {
            self.project.sources[si].kind = SourceKind::Fan { curve: p.curve() };
            self.project.sources[si].preset = Some(p.name.to_string());
        }
        if let Some((c, n)) = assign_sel { self.assign_to_source(Target::Cand(c), &n); }
        if let Some(k) = unassign { self.project.ports.remove(k); self.sel_port = None; }
        if let Some((o, n)) = rename { self.rename_source(&o, &n); }
        if let Some(n) = remove_src { self.remove_source(&n); }
    }

    fn openings_ui(&mut self, ui: &mut egui::Ui) {
        if self.candidates.is_empty() {
            ui.label(egui::RichText::new("None. Duct bodies need planar faces with openings.").color(theme::TEXT_DIM));
        }
        ui.label(egui::RichText::new("Tip: right-click an opening in the 3D view to assign it. For a curved (non-flat) opening use the Curved Opening tool (R) and click its rim.").color(theme::TEXT_DIM));
        if self.tool == Tool::Draw || !self.draw_pts.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Draw an edge loop").strong());
            ui.label(egui::RichText::new("Click points in order along the edge of the opening. Each click is joined to the previous one along the mesh edges, following sharp edges. Click the first point (green) again, or press Enter, to close the loop. Backspace undoes a point, Esc clears.").color(theme::TEXT_DIM));
            ui.horizontal(|ui| {
                ui.label(format!("{} point(s)", self.draw_pts.len()));
                if ui.add_enabled(self.draw_pts.len() >= 3, egui::Button::new("Close loop")).clicked() { self.draw_close(); }
                if ui.add_enabled(!self.draw_pts.is_empty(), egui::Button::new("Undo point")).clicked() { self.draw_undo(); }
                if ui.add_enabled(!self.draw_pts.is_empty(), egui::Button::new("Clear")).clicked() { self.draw_clear(); }
            });
        }
        if !self.rim_loops.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Edge loops at your click").strong());
            let mut add: Option<usize> = None;
            let mut block: Option<usize> = None;
            for li in 0..self.rim_loops.len() {
                let info = &self.rim_loops[li];
                let l = &info.l;
                let kind = if l.planarity_dev > 0.05 { format!("curved, {:.1} mm off-plane", l.planarity_dev) } else { "flat".to_string() };
                let text = format!("loop {}  {:.0} mm around  ({kind})  visible {:.0}%, air behind {:.0}%", li + 1, l.perimeter, info.vis * 100.0, info.clear * 100.0);
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.rim_sel == Some(li), text).clicked() { self.rim_sel = Some(li); }
                    if self.rim_sel == Some(li) && ui.button("Add as opening").clicked() { add = Some(li); }
                    if self.rim_sel == Some(li) && ui.button("Add as blocker").clicked() { block = Some(li); }
                });
            }
            if let Some(li) = add { self.add_rim_opening(li); }
            if let Some(li) = block { self.add_rim_blocker(li); }
            ui.separator();
        }
        ui.horizontal(|ui| {
            let mut changed = ui.checkbox(&mut self.hide_screws, "Hide small round holes").changed();
            if self.hide_screws {
                changed |= ui.add(egui::DragValue::new(&mut self.screw_max_d).speed(0.1).range(0.5..=30.0).prefix("up to ").suffix(" mm")).changed();
            }
            changed |= ui.checkbox(&mut self.show_hidden, "Show hidden loops").on_hover_text("Also list loops that are not visible from outside the duct, or that open onto a pocket instead of a channel.").changed();
            changed |= ui.checkbox(&mut self.detect_curved, "Detect curved openings").on_hover_text("Also look for openings that are not flat (scoops, curved lips). Use the Curved Opening tool for any it misses.").changed();
            if changed { self.refresh_candidates(); }
        });
        let mut act: Option<(usize, u8, String)> = None;
        let mut eye_toggle: Option<PortCap> = None;
        for i in 0..self.candidates.len() {
            let status = match self.port_for_cand(i).map(|k| &self.project.ports[k].kind) {
                Some(PortKind::PressureOutlet { .. }) => "outlet".to_string(),
                Some(k) => format!("inlet <- {}", k.source().unwrap_or("?")),
                None => "unassigned".to_string(),
            };
            let (body, c) = &self.candidates[i];
            let hidden = if c.inline { "  inline: another opening lies down this tube" } else if c.clearance < 0.5 && c.visibility < 0.5 { "  pocket" } else if c.visibility < 0.5 { "  not visible from outside" } else if c.clearance < 0.5 { "  air behind not confirmed" } else { "" };
            let text = format!("#{i}  {body}  {:.1} mm²{}{}  [{status}]", c.area, if c.cap.is_surface() { "  curved" } else { "" }, hidden);
            let vis = !self.cap_hidden(&self.candidates[i].1.cap);
            let cap = self.candidates[i].1.cap.clone();
            ui.horizontal(|ui| {
                if eye_button(ui, vis) { eye_toggle = Some(cap.clone()); }
                if ui.selectable_label(self.sel_cand == Some(i), text).clicked() { self.sel_cand = Some(i); self.sel_port = self.port_for_cand(i); }
            });
        }
        if let Some(cap) = eye_toggle { self.toggle_cap_hidden(&cap); }
        if !self.hidden_caps.is_empty() && ui.small_button("Show all hidden openings").clicked() { self.hidden_caps.clear(); }
        if let Some(i) = self.sel_cand {
            ui.horizontal_wrapped(|ui| {
                ui.menu_button("Assign to source", |ui| {
                    for s in self.project.sources.iter() { if ui.button(&s.name).clicked() { act = Some((i, 0, s.name.clone())); ui.close_menu(); } }
                    if self.project.sources.is_empty() { ui.label("Create a source first"); }
                });
                if ui.button("Pressure outlet").clicked() { act = Some((i, 1, String::new())); }
                if self.port_for_cand(i).is_some() && ui.button("Unassign").clicked() { act = Some((i, 2, String::new())); }
            });
        }
        if let Some((i, what, src)) = act {
            match what {
                0 => self.assign_to_source(Target::Cand(i), &src),
                1 => self.make_outlet(Target::Cand(i)),
                _ => { if let Some(k) = self.port_for_cand(i) { self.project.ports.remove(k); self.sel_port = None; } }
            }
        }
    }

    fn outlets_ui(&mut self, ui: &mut egui::Ui) {
        let mut remove = None;
        let mut any = false;
        let mut eye_key: Option<[i64; 3]> = None;
        for i in 0..self.project.ports.len() {
            let sel = self.sel_port == Some(i);
            let p = &mut self.project.ports[i];
            if let PortKind::PressureOutlet { pressure } = &mut p.kind {
                any = true;
                let vis = !self.hidden_caps.contains(&Self::cap_key(&p.cap));
                let key = Self::cap_key(&p.cap);
                ui.horizontal(|ui| {
                    if eye_button(ui, vis) { eye_key = Some(key); }
                    ui.label(egui::RichText::new("■").color(theme::OUTLET));
                    if ui.selectable_label(sel, &p.name).clicked() { self.sel_port = Some(i); }
                    pressure_drag(ui, pressure, 5.0);
                    if ui.small_button("x").clicked() { remove = Some(i); }
                });
            }
        }
        if !any { ui.label(egui::RichText::new("No outlets yet. Right-click an opening > Pressure outlet.").color(theme::TEXT_DIM)); }
        if let Some(k) = eye_key { if !self.hidden_caps.remove(&k) { self.hidden_caps.insert(k); } }
        if let Some(i) = remove { self.project.ports.remove(i); self.sel_port = None; }
    }

    fn current_issues(&self) -> Vec<Issue> {
        let mut v: Vec<Issue> = self.leak_notes.iter().map(|m| classify_issue(m)).collect();
        if let Some(s) = &self.last { v.extend(issues_of(&s.lock().unwrap(), &self.project)); }
        let mut seen = std::collections::HashSet::new();
        v.retain(|i| seen.insert(i.title.clone()));
        v.sort_by(|a, b| b.sev.cmp(&a.sev));
        v
    }

    /// Voxelize at the current settings and look for air escaping through anything that is not a port.
    fn check_leaks(&mut self) {
        if self.bodies.is_empty() || self.project.ports.is_empty() { self.status = "Define the duct and its ports first.".into(); return; }
        match build_project_domain(&self.project, &self.bodies) {
            Ok(d) => {
                self.leak_notes.clear();
                self.sealed_check = peregrine_geom::sealed_regions(&d);
                if let Some(n) = peregrine_geom::repair_note(&d) { self.leak_notes.push(n); }
                if let Some(n) = peregrine_geom::review_note(&d) { self.leak_notes.push(n); }
                if let Some(n) = peregrine_geom::thin_wall_note(&d) { self.leak_notes.push(n); }
                match peregrine_geom::leak_warning(&self.project, &d) {
                    Some(m) => { self.status = "Leak found: see the marker and the Exceptions tab.".into(); self.leak_notes.push(m); }
                    None => self.status = if d.sealed_idx.is_empty() { "No leaks: the duct is sealed except at its ports.".into() } else { format!("Sealed {} small gap region(s) automatically; no leaks left.", self.sealed_check.len()) },
                }
            }
            Err(e) => self.status = format!("Leak check failed: {e:#}"),
        }
    }

    fn add_plug(&mut self, at: [f64; 3]) {
        let name = Self::unique_name(|n| self.project.plugs.iter().any(|p| p.name == n), "plug");
        self.project.plugs.push(peregrine_project::Plug { name, center: at, radius_mm: self.plug_radius });
        self.check_leaks();
    }

    fn issue_counts(&self) -> (usize, usize) {
        let v = self.current_issues();
        (v.iter().filter(|i| i.sev == Sev::Error).count(), v.iter().filter(|i| i.sev == Sev::Warning).count())
    }

    fn exceptions_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Check for leaks").on_hover_text("Voxelize the duct and look for air escaping through anything that is not a port, without running the simulation").clicked() { self.check_leaks(); }
        });
        let issues = self.current_issues();
        if issues.is_empty() {
            ui.label(egui::RichText::new(if self.last.is_some() { "No warnings or errors." } else { "Nothing to report yet. Warnings and errors from a run appear here." }).color(theme::TEXT_DIM));
            return;
        }
        ui.label(egui::RichText::new("Click an item to read the details.").color(theme::TEXT_DIM));
        ui.add_space(2.0);
        let mut locate: Option<[f64; 3]> = None;
        let mut seal: Option<[f64; 3]> = None;
        let mut toggle_sealed = false;
        let mut plug_r = self.plug_radius;
        let focus = self.focus_issue.take();
        for (k, it) in issues.iter().enumerate() {
            let id = ui.make_persistent_id(("issue", &it.title, k));
            let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, false);
            if focus.as_deref() == Some(it.title.as_str()) { state.set_open(true); }
            let at = issue_point(it, &issues, &self.project);
            state
                .show_header(ui, |ui| {
                    chip(ui, match it.sev { Sev::Error => "ERROR", Sev::Warning => "WARNING", Sev::Info => "INFO" }, sev_color(it.sev));
                    ui.label(egui::RichText::new(&it.title).strong());
                    if at.is_some() { ui.label(egui::RichText::new("(located)").color(theme::TEXT_DIM)); }
                })
                .body(|ui| {
                    ui.label(egui::RichText::new(&it.detail).color(theme::TEXT));
                    if let Some(p) = at {
                        ui.horizontal(|ui| {
                            if ui.button("Show in 3D view").on_hover_text("Centre the view on the marker for this item").clicked() { locate = Some(p); }
                            if it.title.starts_with("Small gaps sealed") {
                                if ui.button(if self.show_sealed { "Hide regions" } else { "Show regions" }).clicked() { toggle_sealed = true; }
                            }
                            if it.title.starts_with("Air leaks") {
                                if ui.button("Seal this leak").on_hover_text("Adds a small solid plug at the marker and checks again").clicked() { seal = Some(p); }
                                ui.add(egui::DragValue::new(&mut plug_r).speed(0.1).range(0.5..=15.0).prefix("r ").suffix(" mm"));
                            }
                        });
                    }
                });
        }
        self.plug_radius = plug_r;
        if toggle_sealed { self.show_sealed = !self.show_sealed; }
        if let Some(p) = seal { self.add_plug(p); }
        if let Some(p) = locate { self.cam.target = p; self.status = format!("Centred on ({:.1}, {:.1}, {:.1}) mm.", p[0], p[1], p[2]); }
    }

    fn results_panel(&mut self, ui: &mut egui::Ui) {
        let Some(snap) = self.last.clone() else {
            ui.label(egui::RichText::new("No results yet. Press Run.").color(theme::TEXT_DIM));
            return;
        };
        let s = snap.lock().unwrap();
        theme::section(ui, "Run", |ui| {
            ui.add(egui::ProgressBar::new(if s.total > 0 { s.step as f32 / s.total as f32 } else { 0.0 }).text(format!("{} / {}", s.step, s.total)).fill(theme::ACCENT));
            ui.label(format!("Engine: {}", s.engine));
            let iss = issues_of(&s, &self.project);
            if !iss.is_empty() {
                let (e, w) = (iss.iter().filter(|i| i.sev == Sev::Error).count(), iss.iter().filter(|i| i.sev == Sev::Warning).count());
                ui.horizontal(|ui| {
                    if e > 0 { chip(ui, &format!("{e} error{}", if e == 1 { "" } else { "s" }), sev_color(Sev::Error)); }
                    if w > 0 { chip(ui, &format!("{w} warning{}", if w == 1 { "" } else { "s" }), sev_color(Sev::Warning)); }
                    if ui.link("Details").clicked() { self.tab = Tab::Exceptions; }
                });
            }
        });
        theme::section(ui, "Flow report", |ui| {
            egui::Grid::new("rep").striped(true).show(ui, |ui| {
                for h in ["Port", "Flow CFM", "Pressure psi", "Share", "Exit speed / angle"] { ui.label(egui::RichText::new(h).color(theme::TEXT_DIM)); }
                ui.end_row();
                for g in &s.sources {
                    ui.label(egui::RichText::new(&g.name).strong().color(theme::INLET));
                    ui.label(format!("{:.3}", cfm(g.total_flow_l_min)));
                    ui.label(format!("{:.4}", psi(g.pressure_pa)));
                    ui.label(match g.split { SplitMode::Plenum => format!("plenum {:.4} psi", psi(g.commanded)), _ => format!("cmd {:.2} CFM", cfm(g.commanded)) });
                    ui.label("");
                    ui.end_row();
                    for r in s.report.iter().filter(|r| r.source.as_deref() == Some(g.name.as_str())) {
                        ui.label(format!("   {}", r.name));
                        ui.label(format!("{:.3}", cfm(r.flow_l_min)));
                        ui.label(format!("{:.4}", psi(r.pressure_pa)));
                        ui.label(format!("{:.0}%", 100.0 * r.flow_l_min / g.total_flow_l_min.abs().max(1e-9)));
                        ui.label("");
                        ui.end_row();
                    }
                }
                for r in s.report.iter().filter(|r| r.source.is_none()) {
                    ui.label(egui::RichText::new(&r.name).color(if r.inlet { theme::INLET } else { theme::OUTLET }));
                    ui.label(format!("{:.3}", cfm(r.flow_l_min)));
                    ui.label(format!("{:.4}", psi(r.pressure_pa)));
                    ui.label("");
                    ui.label(if r.inlet { String::new() } else { format!("{:.1} m/s  {:.0}° from normal", r.mean_speed, r.angle_deg) });
                    ui.end_row();
                }
            });
        });
        theme::section(ui, "Display", |ui| {
            ui.horizontal(|ui| {
                ui.label("Mesh");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Shaded, "Shaded");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Ghost, "Ghost");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Hidden, "Hidden");
            });
            ui.checkbox(&mut self.show_sealed, "Show auto-sealed regions").on_hover_text("Outline the places where tiny gaps were closed automatically (grouped into regions).");
            ui.checkbox(&mut self.show_markers, "Show warning markers").on_hover_text("A clickable marker in the 3D view wherever a warning or error points at a place.");
            ui.checkbox(&mut self.show_ports, "Show ports, openings and blockers").on_hover_text("Turn off to see only the geometry and the flow.");
            ui.checkbox(&mut self.show_lines, "Flow trajectories (streamlines)");
            ui.checkbox(&mut self.probe_on, "Probe pressure and speed").on_hover_text("Hover anywhere over the duct to read the pressure and air speed at that point. Click to pin a reading.");
            if self.probe_on {
                if let Some(snap) = self.last.clone() {
                    let s = snap.lock().unwrap();
                    let mut del = None;
                    for (k, pt) in self.probe_pins.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let txt = probe_cell(&s, *pt).map_or("outside".to_string(), |(pr, sp)| format!("{:.4} psi   {:.1} m/s", psi(pr as f64), sp));
                            ui.label(format!("{}: ({:.1}, {:.1}, {:.1}) mm  {}", k + 1, pt[0], pt[1], pt[2], txt));
                            if ui.small_button("x").clicked() { del = Some(k); }
                        });
                    }
                    if let Some(k) = del { self.probe_pins.remove(k); }
                }
                if !self.probe_pins.is_empty() && ui.small_button("Clear pins").clicked() { self.probe_pins.clear(); }
            }
            if self.show_lines {
                ui.horizontal(|ui| {
                    ui.label("Colour by");
                    ui.selectable_value(&mut self.line_color, LineColor::Speed, "Speed");
                    ui.selectable_value(&mut self.line_color, LineColor::Source, "Source");
                    ui.selectable_value(&mut self.line_color, LineColor::Pressure, "Pressure");
                });
                ui.add(egui::Slider::new(&mut self.line_width, 0.5..=5.0).text("width"));
                ui.add(egui::Slider::new(&mut self.line_cutoff, 0.0..=self.lines_vmax.max(0.5)).text("cutoff m/s")).on_hover_text("Hide every part of a streamline slower than this. 0 shows everything.");
                ui.label(egui::RichText::new("Lines per source").color(theme::TEXT_DIM));
                for src in &mut self.project.sources {
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("■").color(source_color(0)));
                        ui.label(&src.name);
                        ui.add(egui::DragValue::new(&mut src.streamlines).speed(1.0).range(0..=2000));
                    });
                }
                if self.line_color == LineColor::Speed {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(240.0), 10.0), Sense::hover());
                    for k in 0..60 {
                        let r = Rect::from_min_max(Pos2::new(rect.min.x + rect.width() * k as f32 / 60.0, rect.min.y), Pos2::new(rect.min.x + rect.width() * (k + 1) as f32 / 60.0, rect.max.y));
                        ui.painter().rect_filled(r, 0.0, theme::ramp(k as f32 / 59.0));
                    }
                    ui.horizontal(|ui| { ui.label("0"); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(format!("{:.1} m/s", self.lines_vmax))); });
                }
                if self.line_color == LineColor::Pressure {
                    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(240.0), 10.0), Sense::hover());
                    for k in 0..60 {
                        let r = Rect::from_min_max(Pos2::new(rect.min.x + rect.width() * k as f32 / 60.0, rect.min.y), Pos2::new(rect.min.x + rect.width() * (k + 1) as f32 / 60.0, rect.max.y));
                        ui.painter().rect_filled(r, 0.0, theme::ramp(k as f32 / 59.0));
                    }
                    ui.horizontal(|ui| { ui.label(format!("{:.4}", psi(self.lines_prange.0 as f64))); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(format!("{:.4} psi", psi(self.lines_prange.1 as f64)))); });
                }
                ui.label(egui::RichText::new(format!("{} lines drawn{}", self.lines.len(), if self.lines_job.is_some() { " (updating...)" } else { "" })).color(theme::TEXT_DIM));
            }
            ui.checkbox(&mut self.show_slice, "Slice plane");
        });
        theme::section(ui, "Zoom run", |ui| {
            ui.label(egui::RichText::new("Re-simulate a box of the duct at much finer cells. The channels are cut at the top of the box and fed with the flow this run measured there. Use it for the bottom plate and its slots.").color(theme::TEXT_DIM));
            ui.horizontal(|ui| {
                ui.label("Top of box z");
                ui.add(egui::DragValue::new(&mut self.zoom_plane).speed(0.5).suffix(" mm"));
                ui.label("Bottom z");
                ui.add(egui::DragValue::new(&mut self.zoom_zmin).speed(0.5).suffix(" mm"));
            });
            ui.horizontal(|ui| {
                ui.label("x");
                ui.add(egui::DragValue::new(&mut self.zoom_box[0]).speed(0.5));
                ui.add(egui::DragValue::new(&mut self.zoom_box[2]).speed(0.5));
                ui.label("y");
                ui.add(egui::DragValue::new(&mut self.zoom_box[1]).speed(0.5));
                ui.add(egui::DragValue::new(&mut self.zoom_box[3]).speed(0.5));
            });
            ui.horizontal(|ui| {
                ui.label("Cell size");
                ui.add(egui::DragValue::new(&mut self.zoom_dx).speed(0.01).range(0.05..=1.0).suffix(" mm"));
            });
            let n: f64 = [self.zoom_box[2] - self.zoom_box[0] + 6.0, self.zoom_box[3] - self.zoom_box[1] + 6.0, self.zoom_plane - self.zoom_zmin + 6.0].iter().map(|l| l.max(1.0) / self.zoom_dx).product();
            ui.label(egui::RichText::new(format!("about {:.1} M cells, {:.1} GB of graphics memory", n / 1e6, n * 172.0 / 1e9)).color(theme::TEXT_DIM));
            let ready = !s.vel.is_empty() && !s.cells.is_empty();
            if ui.add_enabled(ready, egui::Button::new("Create zoom simulation (new tab)")).on_hover_text("Needs a finished or running coarse simulation in this tab").clicked() {
                let coarse = peregrine_geom::zoom::CoarseField { dims: s.dims, origin: s.origin, dx: s.dx, cells: s.cells.to_vec(), vel: s.vel.to_vec() };
                let spec = peregrine_geom::zoom::ZoomSpec { plane_z: self.zoom_plane, z_min: self.zoom_zmin, x_range: [self.zoom_box[0], self.zoom_box[2]], y_range: [self.zoom_box[1], self.zoom_box[3]], fine_dx: self.zoom_dx };
                match peregrine_geom::zoom::zoom_project(&self.project, &coarse, &spec) {
                    Ok((p, notes)) => { self.status = format!("Zoom project built: {}", notes.join("; ")); self.cmds.push(ShellCmd::Zoom(Box::new(p))); }
                    Err(e) => self.status = format!("Zoom failed: {e:#}"),
                }
            }
        });
        if self.show_slice { theme::section(ui, "Slice", |ui| {
            ui.horizontal(|ui| {
                ui.label("Field");
                ui.selectable_value(&mut self.field, Field::Speed, "Speed");
                ui.selectable_value(&mut self.field, Field::Pressure, "Pressure");
            });
            ui.horizontal(|ui| {
                ui.label("Normal");
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() { ui.selectable_value(&mut self.slice_axis, i, *n); }
            });
            ui.add(egui::Slider::new(&mut self.slice_pos, 0.0..=1.0).text("position"));
            if !s.speed.is_empty() {
                let (lo, hi) = field_range(&s, self.field);
                let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(240.0), 12.0), Sense::hover());
                for k in 0..60 {
                    let r = Rect::from_min_max(Pos2::new(rect.min.x + rect.width() * k as f32 / 60.0, rect.min.y), Pos2::new(rect.min.x + rect.width() * (k + 1) as f32 / 60.0, rect.max.y));
                    ui.painter().rect_filled(r, 0.0, theme::ramp(k as f32 / 59.0));
                }
                ui.horizontal(|ui| {
                    if self.field == Field::Speed { ui.label(format!("{lo:.1}")); } else { ui.label(format!("{:.4}", psi(lo as f64))); }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(if self.field == Field::Speed { format!("{hi:.1} m/s") } else { format!("{:.4} psi", psi(hi as f64)) }));
                });
            }
        }); }
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme::CANVAS);
        // input
        let d = resp.drag_delta();
        let pan = resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (self.tool == Tool::Pan && resp.dragged_by(egui::PointerButton::Primary));
        if pan { self.cam.pan(rect, d.x, d.y); } else if resp.dragged_by(egui::PointerButton::Primary) { self.cam.orbit(d.x, d.y); }
        if resp.hovered() { self.cam.zoom(ui.input(|i| i.smooth_scroll_delta.y)); }

        // geometry: drawn by the graphics card (uploaded once, projected and depth-tested there)
        {
            if self.gl_bodies.len() != self.bodies.len() {
                self.gl_bodies = vec![None; self.bodies.len()];
            }
            let mut items: Vec<viewport_gl::DrawItem> = vec![];
            for i in 0..self.bodies.len() {
                let b = &self.bodies[i];
                if self.mesh_mode == MeshMode::Hidden || !self.visible[i] || b.role == BodyRole::Ignore { continue; }
                let selected = self.sel_bodies.contains(&i);
                let suggested = self.project.is_suggested(&b.name);
                let ghost = self.mesh_mode == MeshMode::Ghost;
                let (col, alpha) = if selected {
                    (Color32::from_rgb(0x4D, 0xA3, 0xFF), if ghost { 90 } else { 210 })
                } else if suggested {
                    (Color32::from_rgb(0xE8, 0xB8, 0x4A), if ghost { 70 } else { 200 })
                } else {
                    match (b.role, ghost) {
                        (BodyRole::Duct, true) => (Color32::from_rgb(0x8F, 0xB4, 0xD9), 28),
                        (_, true) => (Color32::from_rgb(0xB0, 0xB0, 0xB0), 40),
                        (BodyRole::Duct, false) => (Color32::from_rgb(0x8F, 0xB4, 0xD9), 255),
                        _ => (Color32::from_rgb(0xB0, 0xB0, 0xB0), 255),
                    }
                };
                if self.gl_bodies[i].is_none() { self.gl_bodies[i] = Some(std::sync::Arc::new(viewport_gl::GlBody::new(&b.mesh))); }
                let body = self.gl_bodies[i].clone().unwrap();
                items.push(viewport_gl::DrawItem { body, color: [col.r() as f32 / 255.0, col.g() as f32 / 255.0, col.b() as f32 / 255.0, alpha as f32 / 255.0] });
            }
            painter.add(viewport_gl::callback(rect, self.cam.gl_uniform(rect), items));
        }
        // slice
        if let (true, Some(snap)) = (self.show_slice, &self.last) {
            let s = snap.lock().unwrap();
            if !s.speed.is_empty() { draw_slice(&painter, rect, &self.cam, &s, self.field, self.slice_axis, self.slice_pos); }
        }
        if self.show_lines && !self.lines.is_empty() {
            draw_streamlines(&painter, rect, &self.cam, &self.lines, self.line_color, self.lines_vmax, self.lines_prange, self.line_width, self.line_cutoff);
        }
        // port caps: unassigned openings in grey, assigned ports coloured
        let show_ports = self.show_ports;
        for (i, (_, c)) in self.candidates.iter().enumerate() {
            if !show_ports { break; }
            if self.cap_hidden(&c.cap) || !self.cap_visible_from_eye(&c.cap) { continue; }
            if self.port_for_cand(i).is_some() { continue; }
            let sel = self.sel_cand == Some(i);
            viewport::draw_polygon(&painter, rect, &self.cam, &c.cap.polygon, Stroke::new(if sel { 2.5 } else { 1.0 }, if sel { theme::ACCENT } else { Color32::from_gray(200) }));
        }
        for (i, p) in self.project.ports.iter().enumerate() {
            if !show_ports { break; }
            if self.cap_hidden(&p.cap) || !self.cap_visible_from_eye(&p.cap) { continue; }
            let col = if p.kind.is_inlet() { theme::INLET } else { theme::OUTLET };
            viewport::draw_polygon(&painter, rect, &self.cam, &p.cap.polygon, Stroke::new(if self.sel_port == Some(i) { 3.5 } else { 2.5 }, col));
            if let Some((pos, _)) = self.cam.project(rect, p.cap.centroid()) {
                let label = match p.kind.source() { Some(sname) => format!("{} <- {}", p.name, sname), None => p.name.clone() };
                painter.text(pos + Vec2::new(8.0, -8.0), Align2::LEFT_BOTTOM, label, FontId::proportional(11.0), col);
            }
        }
        // blockers: shut openings (red, crossed)
        if show_ports {
            if self.tool == Tool::Block {
                if self.block_stale && !self.bodies.is_empty() { self.refresh_block_cands(); }
                for (i, (_, c)) in self.block_cands.iter().enumerate() {
                    if self.blocker_at(&c.cap).is_some() { continue; }
                    let sel = self.sel_block == Some(i);
                    viewport::draw_polygon(&painter, rect, &self.cam, &c.cap.polygon, Stroke::new(if sel { 3.0 } else { 1.5 }, Color32::from_rgb(0x2D, 0xC8, 0xF0)));
                }
            }
            let red = Color32::from_rgb(0xE0, 0x4A, 0x4A);
            for (i, b) in self.project.blockers.iter().enumerate() {
                viewport::draw_polygon(&painter, rect, &self.cam, &b.cap.polygon, Stroke::new(if self.sel_block == Some(i) { 3.5 } else { 2.0 }, red));
                if let Some((pos, _)) = self.cam.project(rect, b.cap.centroid()) {
                    painter.line_segment([pos + Vec2::new(-3.5, -3.5), pos + Vec2::new(3.5, 3.5)], Stroke::new(1.5, red));
                    painter.line_segment([pos + Vec2::new(-3.5, 3.5), pos + Vec2::new(3.5, -3.5)], Stroke::new(1.5, red));
                }
            }
        }
        if show_ports {
            for p in &self.project.plugs {
                if let Some((q, _)) = self.cam.project(rect, p.center) {
                    let red = Color32::from_rgb(0xE0, 0x4A, 0x4A);
                    painter.circle_stroke(q, 6.0, Stroke::new(2.0, red));
                    painter.line_segment([q + Vec2::new(-3.0, -3.0), q + Vec2::new(3.0, 3.0)], Stroke::new(1.5, red));
                    painter.line_segment([q + Vec2::new(-3.0, 3.0), q + Vec2::new(3.0, -3.0)], Stroke::new(1.5, red));
                }
            }
        }
        // loops found by the Rim tool
        for (li, l) in self.rim_loops.iter().enumerate() {
            let sel = self.rim_sel == Some(li);
            viewport::draw_polygon(&painter, rect, &self.cam, &l.l.points, Stroke::new(if sel { 3.5 } else { 1.5 }, if sel { Color32::from_rgb(0xFF, 0xD0, 0x40) } else { Color32::from_rgb(0xB0, 0x90, 0x30) }));
        }
        // regions closed automatically (opt-in)
        if self.show_sealed {
            let regions: Vec<[f64; 5]> = match &self.last { Some(s) if !s.lock().unwrap().sealed.is_empty() => s.lock().unwrap().sealed.clone(), _ => self.sealed_check.clone() };
            let col = Color32::from_rgb(0xE0, 0x5A, 0xE0);
            for r in &regions {
                let c = [r[0], r[1], r[2]];
                if let Some((q, _)) = self.cam.project(rect, c) {
                    let edge = self.cam.project(rect, [c[0] + r[4].max(1.0), c[1], c[2]]).map_or(8.0, |(e, _)| q.distance(e)).clamp(7.0, 60.0);
                    painter.circle_filled(q, edge, Color32::from_rgba_unmultiplied(0xE0, 0x5A, 0xE0, 40));
                    painter.circle_stroke(q, edge, Stroke::new(1.5, col));
                    painter.text(q + Vec2::new(edge + 3.0, -edge), Align2::LEFT_TOP, format!("sealed {} cells", r[3] as usize), FontId::proportional(10.0), col);
                }
            }
        }
        // markers for warnings and errors that point at a place
        let markers: Vec<(Pos2, Issue)> = if self.show_markers {
            let all = self.current_issues();
            all.iter().filter(|it| !it.title.starts_with("Small gaps sealed")).filter_map(|it| issue_point(it, &all, &self.project).and_then(|p| self.cam.project(rect, p)).map(|(q, _)| (q, it.clone()))).collect()
        } else { vec![] };
        let hover_pos = resp.hover_pos();
        for (q, it) in &markers {
            let col = sev_color(it.sev);
            let hot = hover_pos.map_or(false, |h| h.distance(*q) < 12.0);
            painter.circle_filled(*q, if hot { 11.0 } else { 9.0 }, col);
            painter.circle_stroke(*q, if hot { 11.0 } else { 9.0 }, Stroke::new(1.5, Color32::BLACK));
            painter.text(*q, Align2::CENTER_CENTER, "!", FontId::proportional(13.0), Color32::BLACK);
            if hot {
                let galley = painter.layout_no_wrap(it.title.clone(), FontId::proportional(11.0), Color32::WHITE);
                let r = Rect::from_min_size(*q + Vec2::new(14.0, -8.0), galley.size() + Vec2::new(10.0, 6.0));
                painter.rect_filled(r, 2.0, Color32::from_black_alpha(220));
                painter.galley(r.min + Vec2::new(5.0, 3.0), galley, Color32::WHITE);
            }
        }
        // probe: hover anywhere in the duct to read the pressure and speed there, click to pin the reading
        if self.probe_on && self.last.is_some() {
            let snap = self.last.clone().unwrap();
            let s = snap.lock().unwrap();
            if !s.speed.is_empty() {
                if self.tool == Tool::Orbit && resp.clicked() {
                    if let Some(pp) = resp.interact_pointer_pos() {
                        if let Some((pt, _, _)) = probe_ray(&s, &self.cam, rect, pp) { self.probe_pins.push(pt); }
                    }
                }
                let mut tags: Vec<([f64; 3], bool)> = self.probe_pins.iter().map(|p| (*p, true)).collect();
                if let Some(hp) = resp.hover_pos() {
                    if let Some((pt, _, _)) = probe_ray(&s, &self.cam, rect, hp) { tags.push((pt, false)); }
                }
                for (pt, pinned) in tags {
                    let Some((q, _)) = self.cam.project(rect, pt) else { continue };
                    let Some((pr, sp)) = probe_cell(&s, pt) else { continue };
                    painter.circle_filled(q, 4.5, if pinned { Color32::from_rgb(0xFF, 0xD0, 0x40) } else { Color32::WHITE });
                    painter.circle_stroke(q, 4.5, Stroke::new(1.0, Color32::BLACK));
                    let galley = painter.layout_no_wrap(format!("{:.4} psi   {:.1} m/s", psi(pr as f64), sp), FontId::proportional(12.0), Color32::WHITE);
                    let r = Rect::from_min_size(q + Vec2::new(10.0, -22.0), galley.size() + Vec2::new(10.0, 6.0));
                    painter.rect_filled(r, 2.0, Color32::from_black_alpha(215));
                    painter.galley(r.min + Vec2::new(5.0, 3.0), galley, Color32::WHITE);
                }
            }
        }
        let clicked_marker = if resp.clicked() { resp.interact_pointer_pos().and_then(|pp| markers.iter().find(|(q, _)| q.distance(pp) < 12.0).map(|(_, it)| it.title.clone())) } else { None };
        if let Some(t) = &clicked_marker { self.tab = Tab::Exceptions; self.focus_issue = Some(t.clone()); }
        // hand-drawn loop preview
        if let Some(bi) = self.draw_body {
            let mesh = &self.bodies[bi].mesh;
            let pts: Vec<Pos2> = self.draw_path.iter().filter_map(|&v| self.cam.project(rect, mesh.vertices[v as usize]).map(|x| x.0)).collect();
            if pts.len() > 1 { painter.add(Shape::line(pts, Stroke::new(3.0, Color32::from_rgb(0xFF, 0xD0, 0x40)))); }
            for (k, &v) in self.draw_pts.iter().enumerate() {
                if let Some((q, _)) = self.cam.project(rect, mesh.vertices[v as usize]) {
                    painter.circle_filled(q, if k == 0 { 6.0 } else { 4.0 }, if k == 0 { Color32::from_rgb(0x3F, 0xB9, 0x50) } else { Color32::WHITE });
                }
            }
        }
        // hover highlight, left-click select, right-click menu
        let hover = if resp.hovered() && self.show_ports && self.tool != Tool::Block { resp.hover_pos().and_then(|pp| self.hit_test(rect, pp)) } else { None };
        if hover.is_some() && ui.input(|i| i.key_pressed(egui::Key::V) && !i.modifiers.command) {
            if let Some(cap) = hover.and_then(|t| self.target_cap(t).cloned()) {
                self.toggle_cap_hidden(&cap);
                self.status = "Opening hidden. Use the eye in the Ports tab (or 'Show all hidden openings') to bring it back.".into();
            }
        }
        if let (Some(t), Some(pos)) = (hover, resp.hover_pos()) {
            if let Some(cap) = self.target_cap(t) {
                viewport::draw_polygon(&painter, rect, &self.cam, &cap.polygon, Stroke::new(3.0, theme::ACCENT));
            }
            let text = match t {
                Target::Port(i) => format!("{}  - right-click to edit", self.project.ports[i].name),
                Target::Cand(i) => format!("Opening #{i}  {:.1} mm²  - right-click to assign", self.candidates[i].1.area),
            };
            let galley = painter.layout_no_wrap(text, FontId::proportional(11.0), Color32::WHITE);
            let r = Rect::from_min_size(pos + Vec2::new(14.0, 14.0), galley.size() + Vec2::new(10.0, 6.0));
            painter.rect_filled(r, 2.0, Color32::from_black_alpha(200));
            painter.galley(r.min + Vec2::new(5.0, 3.0), galley, Color32::WHITE);
        }
        // part under the cursor (when no opening is hovered): name label
        if hover.is_none() && resp.hovered() && self.mesh_mode != MeshMode::Hidden {
            if let Some(pos) = resp.hover_pos() {
                if let Some(bi) = self.pick_body(rect, pos) {
                    let b = &self.bodies[bi];
                    let tag = match self.project.duct_of(&b.name) { Some(d) => format!("  (duct: {})", d.name), None if self.project.is_suggested(&b.name) => "  (★ looks like a duct)".to_string(), None => String::new() };
                    let galley = painter.layout_no_wrap(format!("{}{}", b.name, tag), FontId::proportional(11.0), Color32::WHITE);
                    let r = Rect::from_min_size(pos + Vec2::new(14.0, 14.0), galley.size() + Vec2::new(10.0, 6.0));
                    painter.rect_filled(r, 2.0, Color32::from_black_alpha(200));
                    painter.galley(r.min + Vec2::new(5.0, 3.0), galley, Color32::WHITE);
                }
            }
        }
        if clicked_marker.is_some() {
        } else if resp.clicked() && self.tool == Tool::Block {
            if let Some(pp) = resp.interact_pointer_pos() {
                // an existing blocker is removed, a small hole is closed
                let hit = |cap: &PortCap, cam: &Camera| -> Option<f32> {
                    let pts: Vec<Option<(Pos2, f64)>> = cap.polygon.iter().map(|p| cam.project(rect, *p)).collect();
                    if pts.len() < 3 || pts.iter().any(|p| p.is_none()) { return None; }
                    let poly: Vec<[f64; 2]> = pts.iter().map(|p| { let q = p.unwrap().0; [q.x as f64, q.y as f64] }).collect();
                    let c = cam.project(rect, cap.centroid()).map(|x| x.0)?;
                    if point_in_polygon(&poly, [pp.x as f64, pp.y as f64]) { Some(0.0) } else if c.distance(pp) < 16.0 { Some(c.distance(pp)) } else { None }
                };
                let rm = (0..self.project.blockers.len()).filter_map(|i| hit(&self.project.blockers[i].cap, &self.cam).map(|d| (d, i))).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let add = (0..self.block_cands.len()).filter(|&i| self.blocker_at(&self.block_cands[i].1.cap).is_none()).filter_map(|i| hit(&self.block_cands[i].1.cap, &self.cam).map(|d| (d, i))).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                match (rm, add) {
                    (Some((dr, i)), a) if a.map_or(true, |x| dr <= x.0) => { let n = self.project.blockers[i].name.clone(); self.remove_blocker(i); self.status = format!("Removed blocker {n}."); }
                    (_, Some((_, i))) => {
                        let (body, c) = self.block_cands[i].clone();
                        self.status = format!("Blocked a {:.1} mm hole. Click a red outline to remove it again.", c.equiv_diameter);
                        self.add_blocker(body, c.cap);
                    }
                    _ => self.status = "No small hole there. Draw its edge with the Draw Loop tool (L) and press 'Add as blocker'.".into(),
                }
            }
        } else if resp.clicked() && self.tool == Tool::Draw {
            if let Some(pp) = resp.interact_pointer_pos() {
                match self.pick_body_tri(rect, pp) {
                    Some((bi, ti)) => {
                        // nearest corner of the clicked triangle
                        let tri = self.bodies[bi].mesh.triangles[ti];
                        let proj = self.cam.projector(rect);
                        let best = tri.iter().filter_map(|&v| proj.project(self.bodies[bi].mesh.vertices[v as usize]).map(|(q, _)| (q.distance(pp), v))).min_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        if let Some((_, v)) = best {
                            // clicking the first point again closes the loop
                            let near_first = self.draw_body == Some(bi) && self.draw_pts.len() >= 3 && self.draw_pts.first().and_then(|&f| proj.project(self.bodies[bi].mesh.vertices[f as usize])).map_or(false, |(q, _)| q.distance(pp) < 14.0);
                            if near_first { self.draw_close(); } else { self.draw_add_vertex(bi, v); }
                        }
                    }
                    None => self.status = "Click on the part's surface, on the edge of the opening.".into(),
                }
            }
        } else if resp.clicked() && self.tool == Tool::Rim {
            if let Some(pp) = resp.interact_pointer_pos() {
                match self.pick_body_tri(rect, pp) {
                    Some((bi, ti)) => self.rim_pick(bi, ti),
                    None => self.status = "Click on a part's surface near the curved opening.".into(),
                }
            }
        } else if resp.clicked() && !self.probe_on {
            if let Some(pp) = resp.interact_pointer_pos() {
                if let Some(t) = self.hit_test(rect, pp) {
                    self.select_target(t);
                    self.tab = Tab::Ports;
                } else if let Some(bi) = self.pick_body(rect, pp) {
                    let add = ui.input(|x| x.modifiers.ctrl || x.modifiers.command);
                    self.select_body(bi, add);
                } else {
                    self.sel_bodies.clear();
                }
            }
        }
        if resp.secondary_clicked() {
            let pp = resp.interact_pointer_pos();
            self.ctx_target = pp.and_then(|pp| self.hit_test(rect, pp));
            self.ctx_body = if self.ctx_target.is_none() { pp.and_then(|pp| self.pick_body(rect, pp)) } else { None };
            if let Some(t) = self.ctx_target { self.select_target(t); }
            if let Some(bi) = self.ctx_body { if !self.sel_bodies.contains(&bi) { self.select_body(bi, false); } }
        }
        resp.context_menu(|ui| self.context_menu_ui(ui));
        // axis gizmo
        let o = Pos2::new(rect.min.x + 40.0, rect.max.y - 40.0);
        for (v, c, n) in [([1.0, 0.0, 0.0], Color32::from_rgb(0xE0, 0x5A, 0x4F), "X"), ([0.0, 1.0, 0.0], Color32::from_rgb(0x4F, 0xC0, 0x5A), "Y"), ([0.0, 0.0, 1.0], Color32::from_rgb(0x4F, 0x8F, 0xE0), "Z")] {
            let t = self.cam.target;
            if let (Some((a, _)), Some((b, _))) = (self.cam.project(rect, t), self.cam.project(rect, [t[0] + v[0] * 10.0, t[1] + v[1] * 10.0, t[2] + v[2] * 10.0])) {
                let dir = (b - a).normalized() * 24.0;
                painter.line_segment([o, o + dir], Stroke::new(2.0, c));
                painter.text(o + dir * 1.25, Align2::CENTER_CENTER, n, FontId::proportional(10.0), c);
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.label(&self.status);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if !self.bodies.is_empty() {
                    let b = project_bounds(&self.project, &self.bodies);
                    let dx = self.project.solver.cell_size_mm;
                    let n: f64 = (0..3).map(|i| ((b.max[i] - b.min[i]) / dx).ceil().max(1.0)).product();
                    let (gpu, _) = peregrine_voxel::Grid { origin: [0.0; 3], dx, dims: [n as usize, 1, 1] }.memory_estimate();
                    ui.label(egui::RichText::new(format!("{:.2} M cells  |  ~{} MB GPU", n / 1e6, gpu >> 20)).color(theme::TEXT_DIM));
                }
                if let Some(Some(g)) = GPU.get() {
                    let v = g.vram_bytes.map_or(String::new(), |b| format!(", {:.1} GB", b as f64 / 1073741824.0));
                    ui.label(egui::RichText::new(format!("{}{}", g.name, v)).color(theme::TEXT_DIM));
                }
                if let Some(l) = &self.last {
                    let s = l.lock().unwrap();
                    if s.finished && s.error.is_none() { ui.label("Done"); }
                }
            });
        });
    }
}

/// An eye icon button: open eye = shown, struck-through = hidden. Returns true when clicked.
fn eye_button(ui: &mut egui::Ui, visible: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(20.0, 16.0), Sense::click());
    let col = if visible { Color32::from_rgb(0xD8, 0xD8, 0xD8) } else { Color32::from_rgb(0x70, 0x70, 0x70) };
    let c = rect.center();
    let p = ui.painter();
    if resp.hovered() { p.rect_filled(rect, 2.0, theme::WIDGET_HOVER); }
    let pts: Vec<Pos2> = (0..=16).map(|k| { let t = k as f32 / 16.0; let x = -7.5 + 15.0 * t; let y = -4.5 * (1.0 - (2.0 * t - 1.0).powi(2)); c + Vec2::new(x, y) }).collect();
    let low: Vec<Pos2> = pts.iter().map(|q| Pos2::new(q.x, c.y + (c.y - q.y))).collect();
    p.add(Shape::line(pts, Stroke::new(1.4, col)));
    p.add(Shape::line(low, Stroke::new(1.4, col)));
    p.circle_filled(c, 2.4, col);
    if !visible { p.line_segment([c + Vec2::new(-8.0, 6.0), c + Vec2::new(8.0, -6.0)], Stroke::new(1.8, Color32::from_rgb(0xE0, 0x5A, 0x4F))); }
    resp.on_hover_text(if visible { "Hide this opening (or hover it and press V)" } else { "Show this opening" }).clicked()
}

fn source_color(i: usize) -> Color32 {
    const PAL: [(u8, u8, u8); 6] = [(0x2D, 0xC8, 0xF0), (0xF0, 0x88, 0x3E), (0x7C, 0xE0, 0x5A), (0xE0, 0x5A, 0xA0), (0xB0, 0x8C, 0xF0), (0xF2, 0xD0, 0x3A)];
    let c = PAL[i % PAL.len()];
    Color32::from_rgb(c.0, c.1, c.2)
}

fn draw_streamlines(painter: &egui::Painter, rect: Rect, cam: &Camera, lines: &[Streamline], mode: LineColor, vmax: f32, prange: (f32, f32), width: f32, cutoff: f32) {
    let mut shapes: Vec<Shape> = Vec::with_capacity(lines.len() * 8);
    for l in lines {
        // parts of the line that are fast enough to show (the whole line when there is no cutoff)
        let mut runs: Vec<(usize, usize)> = vec![];
        if cutoff <= 0.0 {
            runs.push((0, l.pts.len()));
        } else {
            let mut start: Option<usize> = None;
            for (i, &v) in l.speed.iter().enumerate() {
                if v >= cutoff { if start.is_none() { start = Some(i); } } else if let Some(a) = start.take() { runs.push((a, i)); }
            }
            if let Some(a) = start { runs.push((a, l.speed.len())); }
        }
        for (a, b) in runs {
            let pts: Vec<Option<Pos2>> = l.pts[a..b].iter().map(|p| cam.project(rect, [p[0] as f64, p[1] as f64, p[2] as f64]).map(|x| x.0)).collect();
            let spd = &l.speed[a..b];
            let n = pts.len();
            match mode {
                LineColor::Source => {
                    let mut run: Vec<Pos2> = vec![];
                    for p in pts.iter().chain(std::iter::once(&None)) {
                        match p {
                            Some(q) => run.push(*q),
                            None => { if run.len() > 1 { shapes.push(Shape::line(std::mem::take(&mut run), Stroke::new(width, source_color(l.source)))); } else { run.clear(); } }
                        }
                    }
                }
                LineColor::Pressure => {
                    let prs = &l.pressure[a..b];
                    let mut i = 0;
                    while i + 1 < n {
                        let j = (i + 4).min(n - 1);
                        let seg: Vec<Pos2> = pts[i..=j].iter().flatten().copied().collect();
                        if seg.len() == j - i + 1 {
                            let v = prs[i..=j].iter().sum::<f32>() / (j - i + 1) as f32;
                            shapes.push(Shape::line(seg, Stroke::new(width, theme::ramp((v - prange.0) / (prange.1 - prange.0).max(1e-6)))));
                        }
                        i = j;
                    }
                }
                LineColor::Speed => {
                    let mut i = 0;
                    while i + 1 < n {
                        let j = (i + 4).min(n - 1);
                        let seg: Vec<Pos2> = pts[i..=j].iter().flatten().copied().collect();
                        if seg.len() == j - i + 1 {
                            let v = spd[i..=j].iter().sum::<f32>() / (j - i + 1) as f32;
                            shapes.push(Shape::line(seg, Stroke::new(width, theme::ramp(v / vmax))));
                        }
                        i = j;
                    }
                }
            }
        }
    }
    painter.extend(shapes);
}

fn fan_library_menu(ui: &mut egui::Ui) -> Option<FanPreset> {
    let mut chosen = None;
    for p in presets::fan_presets() {
        let label = format!("{}{}", p.name, if p.verified { "" } else { "  (estimate)" });
        if ui.button(label).on_hover_text(p.note).clicked() { chosen = Some(p); }
    }
    chosen
}

/// Gauge pressure (Pa) and speed (m/s) in the fluid cell containing `p` (mm), if it is in fluid.
fn probe_cell(s: &Snapshot, p: [f64; 3]) -> Option<(f32, f32)> {
    let mut c = [0usize; 3];
    for k in 0..3 {
        let g = ((p[k] - s.origin[k]) / s.dx).floor();
        if g < 0.0 || g >= s.dims[k] as f64 { return None; }
        c[k] = g as usize;
    }
    let i = c[0] + s.dims[0] * (c[1] + s.dims[1] * c[2]);
    if s.cells[i] == 1 { return None; }
    Some((s.pressure[i], s.speed[i]))
}

/// True when a fluid cell is enclosed by walls (inside a duct) rather than in the open air around it:
/// walls are met within reach in at least four of the six axis directions.
fn inside_duct(s: &Snapshot, p: [f64; 3]) -> bool {
    let mut c = [0usize; 3];
    for k in 0..3 {
        let g = ((p[k] - s.origin[k]) / s.dx).floor();
        if g < 0.0 || g >= s.dims[k] as f64 { return false; }
        c[k] = g as usize;
    }
    let reach = ((40.0 / s.dx) as usize).clamp(20, 400); // look up to about 40 mm each way
    let mut walls = 0;
    for axis in 0..3 {
        for dir in [-1i64, 1] {
            let mut q = [c[0] as i64, c[1] as i64, c[2] as i64];
            for _ in 0..reach {
                q[axis] += dir;
                if q[axis] < 0 || q[axis] >= s.dims[axis] as i64 { break; }
                if s.cells[q[0] as usize + s.dims[0] * (q[1] as usize + s.dims[1] * q[2] as usize)] == 1 { walls += 1; break; }
            }
        }
    }
    walls >= 4
}

/// The first fluid cell along the view ray through a screen point: the point (mm), its pressure (Pa) and speed (m/s).
fn probe_ray(s: &Snapshot, cam: &Camera, rect: Rect, pos: Pos2) -> Option<([f64; 3], f32, f32)> {
    let (o, d) = cam.ray(rect, pos);
    let (mut t0, mut t1) = (0.0f64, f64::MAX);
    for k in 0..3 {
        let lo = s.origin[k];
        let hi = s.origin[k] + s.dims[k] as f64 * s.dx;
        if d[k].abs() < 1e-12 {
            if o[k] < lo || o[k] > hi { return None; }
        } else {
            let (a, b) = ((lo - o[k]) / d[k], (hi - o[k]) / d[k]);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
    }
    if t0 > t1 { return None; }
    let step = s.dx * 0.5;
    let mut t = t0 + 1e-6;
    while t < t1 {
        let pt = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
        if let Some((pr, sp)) = probe_cell(s, pt) {
            if inside_duct(s, pt) { return Some((pt, pr, sp)); }
        }
        t += step;
    }
    None
}

fn field_range(s: &Snapshot, f: Field) -> (f32, f32) {
    let v = if f == Field::Speed { &s.speed } else { &s.pressure };
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (x, &c) in v.iter().zip(s.cells.iter()) {
        if c != 1 { lo = lo.min(*x); hi = hi.max(*x); }
    }
    if lo >= hi { (0.0, 1.0) } else { (if f == Field::Speed { 0.0 } else { lo }, hi) }
}

fn draw_slice(painter: &egui::Painter, rect: Rect, cam: &Camera, s: &Snapshot, field: Field, axis: usize, pos: f32) {
    let dims = s.dims;
    let k = ((pos * dims[axis] as f32) as usize).min(dims[axis] - 1);
    let (a1, a2) = match axis { 0 => (1, 2), 1 => (0, 2), _ => (0, 1) };
    let stride = ((dims[a1].max(dims[a2]) + 179) / 180).max(1);
    let data = if field == Field::Speed { &s.speed } else { &s.pressure };
    let (lo, hi) = field_range(s, field);
    let mut mesh = egui::Mesh::default();
    let p = |i1: usize, i2: usize| {
        let mut c = [0usize; 3];
        c[axis] = k;
        c[a1] = i1;
        c[a2] = i2;
        [s.origin[0] + c[0] as f64 * s.dx + if axis == 0 { 0.5 * s.dx } else { 0.0 }, s.origin[1] + c[1] as f64 * s.dx + if axis == 1 { 0.5 * s.dx } else { 0.0 }, s.origin[2] + c[2] as f64 * s.dx + if axis == 2 { 0.5 * s.dx } else { 0.0 }]
    };
    let mut i2 = 0;
    while i2 < dims[a2] {
        let mut i1 = 0;
        while i1 < dims[a1] {
            let mut c = [0usize; 3];
            c[axis] = k; c[a1] = i1; c[a2] = i2;
            let idx = c[0] + dims[0] * (c[1] + dims[1] * c[2]);
            let cell = s.cells[idx];
            if cell != 1 {
                let t = (data[idx] - lo) / (hi - lo).max(1e-9);
                let col = theme::ramp(t);
                let corners = [p(i1, i2), p(i1 + stride, i2), p(i1 + stride, i2 + stride), p(i1, i2 + stride)];
                let pts: Vec<_> = corners.iter().filter_map(|q| cam.project(rect, *q)).collect();
                if pts.len() == 4 {
                    let base = mesh.vertices.len() as u32;
                    for (pp, _) in &pts { mesh.vertices.push(egui::epaint::Vertex { pos: *pp, uv: egui::epaint::WHITE_UV, color: col }); }
                    mesh.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                }
            }
            i1 += stride;
        }
        i2 += stride;
    }
    painter.add(Shape::mesh(mesh));
}

fn draw_tool_icon(p: &egui::Painter, r: Rect, t: Tool) {
    let c = r.center();
    let s = Stroke::new(1.6, Color32::WHITE);
    match t {
        Tool::Orbit => {
            p.circle_stroke(c, 8.0, s);
            p.add(Shape::line(vec![c + Vec2::new(8.0, 0.0), c + Vec2::new(11.0, -4.0), c + Vec2::new(5.0, -4.0)], s));
        }
        Tool::Pan => {
            for d in [Vec2::X, Vec2::Y] {
                p.line_segment([c - d * 9.0, c + d * 9.0], s);
                let n = Vec2::new(d.y, d.x);
                for sgn in [-1.0, 1.0] {
                    let tip = c + d * 9.0 * sgn;
                    p.line_segment([tip, tip - d * 3.0 * sgn + n * 3.0], s);
                    p.line_segment([tip, tip - d * 3.0 * sgn - n * 3.0], s);
                }
            }
        }
        Tool::PickPort => {
            p.circle_stroke(c, 8.0, s);
            p.circle_filled(c, 2.5, Color32::WHITE);
        }
        Tool::Rim => {
            // a wavy (non-planar) loop
            let pts: Vec<Pos2> = (0..=24).map(|k| { let a = k as f32 / 24.0 * std::f32::consts::TAU; c + Vec2::new(8.5 * a.cos(), 5.5 * a.sin() + 2.5 * (2.0 * a).cos()) }).collect();
            p.add(Shape::line(pts, s));
        }
        Tool::Draw => {
            // a pen drawing a loop
            let pts: Vec<Pos2> = (0..=18).map(|k| { let a = k as f32 / 18.0 * std::f32::consts::TAU * 0.92; c + Vec2::new(8.0 * a.cos(), 6.5 * a.sin()) }).collect();
            p.add(Shape::line(pts, s));
            p.circle_filled(c + Vec2::new(8.0, 0.0), 2.2, Color32::WHITE);
        }
        Tool::Block => {
            p.circle_stroke(c, 8.0, s);
            p.line_segment([c + Vec2::new(-5.0, -5.0), c + Vec2::new(5.0, 5.0)], s);
            p.line_segment([c + Vec2::new(-5.0, 5.0), c + Vec2::new(5.0, -5.0)], s);
        }
        Tool::Slice => {
            p.rect_stroke(Rect::from_center_size(c, Vec2::splat(16.0)), 0.0, s, egui::StrokeKind::Inside);
            p.line_segment([c - Vec2::new(8.0, -3.0), c + Vec2::new(8.0, 3.0)], s);
        }
    }
}

impl App {
    /// Per-frame upkeep that must also run for tabs that are not on screen (streamline jobs, run bookkeeping).
    fn background(&mut self, ctx: &egui::Context) {
        self.update_lines();
        if self.running() || self.lines_job.is_some() { ctx.request_repaint_after(std::time::Duration::from_millis(100)); }
        if let Some(r) = &self.run {
            let s = r.snap.lock().unwrap();
            if s.finished {
                let msg = match &s.error { Some(e) => format!("Run failed: {e}"), None => format!("Run complete ({} steps)", s.step) };
                drop(s);
                self.status = msg;
                self.run = None;
            }
        }
    }

    fn title(&self) -> String { self.project.name.clone() }

    /// A new tab with the same geometry, ports and settings but no results.
    fn duplicate(&self) -> App {
        let mut a = App::new();
        a.project = self.project.clone();
        a.project.name = format!("{} copy", self.project.name);
        a.bodies = self.bodies.clone();
        a.visible = self.visible.clone();
        a.gl_bodies = self.gl_bodies.clone();
        a.candidates = self.candidates.clone();
        a.cam = self.cam.clone();
        a.backend = self.backend;
        a.mesh_mode = self.mesh_mode;
        a.hide_screws = self.hide_screws;
        a.screw_max_d = self.screw_max_d;
        a.status = "Duplicated tab: change the settings and press Run to compare.".into();
        a
    }

    fn tab_strip(&mut self, ui: &mut egui::Ui) {
        let infos = self.tab_infos.clone();
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 3.0;
            for (i, t) in infos.iter().enumerate() {
                let active = i == self.tab_active;
                let fill = if active { theme::APP_BG } else { theme::BAR };
                egui::Frame::new().fill(fill).stroke(Stroke::new(1.0, if active { theme::ACCENT } else { theme::BORDER })).corner_radius(egui::CornerRadius::same(2)).inner_margin(egui::Margin::symmetric(6, 1)).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if t.running {
                            let (r, resp) = ui.allocate_exact_size(Vec2::splat(9.0), Sense::hover());
                            ui.painter().circle_filled(r.center(), 3.5, theme::ACCENT);
                            resp.on_hover_text("Simulation running");
                        }
                        let title = egui::RichText::new(&t.title).color(if active { Color32::WHITE } else { theme::TEXT_DIM });
                        if ui.add(egui::Button::new(title).fill(Color32::TRANSPARENT).stroke(Stroke::NONE)).clicked() && !active { self.cmds.push(ShellCmd::Switch(i)); }
                        if ui.add(egui::Button::new("×").small().fill(Color32::TRANSPARENT).stroke(Stroke::NONE)).on_hover_text("Close tab (stops its run)").clicked() { self.cmds.push(ShellCmd::Close(i)); }
                    });
                });
            }
            if ui.add(egui::Button::new("+").fill(theme::BAR)).on_hover_text("New simulation tab (Ctrl+T)").clicked() { self.cmds.push(ShellCmd::New); }
        });
    }

    fn ui(&mut self, ctx: &egui::Context) {
        if std::mem::take(&mut self.draw_close_requested) { self.draw_close(); }
        if std::mem::take(&mut self.draw_undo_requested) { self.draw_undo(); }
        if std::mem::take(&mut self.draw_clear_requested) { self.draw_clear(); }
        let kb = ctx.wants_keyboard_input();
        let (mut do_undo, mut do_redo) = (false, false);
        ctx.input(|i| {
            if !kb && i.modifiers.command && i.key_pressed(egui::Key::Z) { if i.modifiers.shift { do_redo = true; } else { do_undo = true; } }
            if !kb && i.modifiers.command && i.key_pressed(egui::Key::Y) { do_redo = true; }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::T) { self.cmds.push(ShellCmd::New); }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::W) { self.cmds.push(ShellCmd::Close(self.tab_active)); }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Tab) && !self.tab_infos.is_empty() { self.cmds.push(ShellCmd::Switch((self.tab_active + 1) % self.tab_infos.len())); }
            if i.modifiers.ctrl { return; }
            if i.key_pressed(egui::Key::O) { self.tool = Tool::Orbit; }
            if i.key_pressed(egui::Key::H) { self.tool = Tool::Pan; }
            if i.key_pressed(egui::Key::P) { self.tool = Tool::PickPort; }
            if i.key_pressed(egui::Key::S) { self.tool = Tool::Slice; }
            if i.key_pressed(egui::Key::R) { self.tool = Tool::Rim; }
            if i.key_pressed(egui::Key::L) { self.tool = Tool::Draw; }
            if i.key_pressed(egui::Key::B) { self.tool = Tool::Block; self.block_stale = true; }
            if self.tool == Tool::Draw {
                if i.key_pressed(egui::Key::Enter) { self.draw_close_requested = true; }
                if i.key_pressed(egui::Key::Backspace) { self.draw_undo_requested = true; }
                if i.key_pressed(egui::Key::Escape) { self.draw_clear_requested = true; }
            }
            if i.key_pressed(egui::Key::M) {
                self.mesh_mode = match self.mesh_mode { MeshMode::Shaded => MeshMode::Ghost, MeshMode::Ghost => MeshMode::Hidden, MeshMode::Hidden => MeshMode::Shaded };
            }
        });
        if do_undo { self.undo_project(); }
        if do_redo { self.redo_project(); }
        let bar = egui::Frame::new().fill(theme::BAR).inner_margin(egui::Margin::symmetric(8, 3)).stroke(Stroke::new(1.0, theme::BORDER));
        egui::TopBottomPanel::top("menu").frame(bar).show(ctx, |ui| self.menu_bar(ui));
        egui::TopBottomPanel::top("options").frame(bar).exact_height(30.0).show(ctx, |ui| self.options_bar(ui));
        egui::TopBottomPanel::top("tabs")
            .frame(egui::Frame::new().fill(theme::DEEP).inner_margin(egui::Margin::symmetric(6, 2)).stroke(Stroke::new(1.0, theme::BORDER)))
            .exact_height(30.0)
            .show(ctx, |ui| self.tab_strip(ui));
        egui::TopBottomPanel::bottom("status").frame(bar).exact_height(22.0).show(ctx, |ui| self.status_bar(ui));
        egui::SidePanel::left("tools").exact_width(40.0).resizable(false)
            .frame(egui::Frame::new().fill(theme::APP_BG).stroke(Stroke::new(1.0, theme::BORDER))).show(ctx, |ui| self.tool_palette(ui));
        egui::SidePanel::right("dock").default_width(320.0).min_width(260.0)
            .frame(egui::Frame::new().fill(theme::APP_BG).inner_margin(egui::Margin::same(6)).stroke(Stroke::new(1.0, theme::BORDER))).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 3.0;
                    let (n_err, n_warn) = self.issue_counts();
                    for (t, n) in [(Tab::Bodies, "Bodies"), (Tab::Ports, "Ports"), (Tab::Results, "Results"), (Tab::Exceptions, "Exceptions")] {
                        let sel = self.tab == t;
                        let text = egui::RichText::new(n).color(if sel { Color32::WHITE } else { theme::TEXT_DIM });
                        if ui.add(egui::Button::new(text).fill(if sel { theme::BAR } else { theme::APP_BG }).stroke(Stroke::NONE)).clicked() { self.tab = t; }
                        if t == Tab::Exceptions {
                            if n_err > 0 { chip(ui, &n_err.to_string(), sev_color(Sev::Error)); }
                            if n_warn > 0 { chip(ui, &n_warn.to_string(), sev_color(Sev::Warning)); }
                        }
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                    Tab::Bodies => self.bodies_panel(ui),
                    Tab::Ports => self.ports_panel(ui),
                    Tab::Results => self.results_panel(ui),
                    Tab::Exceptions => self.exceptions_panel(ui),
                });
            });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::CANVAS)).show(ctx, |ui| self.viewport_ui(ui));
        self.tutorial_ui(ctx);
        self.track_changes(ctx);
    }
}

/// Tabbed workspace: every tab is an independent simulation (own geometry, ports, settings, run and results).
struct Shell {
    docs: Vec<App>,
    active: usize,
    counter: usize,
}

impl Shell {
    fn new_doc(&mut self) -> App {
        self.counter += 1;
        let mut a = App::new();
        a.project.name = format!("untitled {}", self.counter);
        a
    }

    fn close(&mut self, i: usize) {
        if i >= self.docs.len() { return; }
        if let Some(r) = &self.docs[i].run { r.stop.store(true, Ordering::Relaxed); }
        self.docs.remove(i);
        if self.docs.is_empty() {
            let d = self.new_doc();
            self.docs.push(d);
            self.active = 0;
        } else if i < self.active || self.active >= self.docs.len() {
            self.active = self.active.saturating_sub(1).min(self.docs.len() - 1);
        }
    }
}

impl eframe::App for Shell {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        for d in &mut self.docs { d.background(ctx); }
        self.active = self.active.min(self.docs.len() - 1);
        let infos: Vec<TabInfo> = self.docs.iter().map(|d| TabInfo { title: d.title(), running: d.running() }).collect();
        let active = self.active;
        {
            let d = &mut self.docs[active];
            d.tab_infos = infos;
            d.tab_active = active;
            d.ui(ctx);
        }
        let cmds = std::mem::take(&mut self.docs[active].cmds);
        for c in cmds {
            match c {
                ShellCmd::New => { let d = self.new_doc(); self.docs.push(d); self.active = self.docs.len() - 1; }
                ShellCmd::Duplicate => { let d = self.docs[self.active].duplicate(); self.docs.push(d); self.active = self.docs.len() - 1; }
                ShellCmd::Close(i) => self.close(i),
                ShellCmd::Switch(i) => { if i < self.docs.len() { self.active = i; } }
                ShellCmd::Zoom(p) => {
                    let mut d = self.docs[self.active].duplicate();
                    d.project = *p;
                    d.project_path = None;
                    d.last = None;
                    d.reclassify();
                    d.tab = Tab::Ports;
                    d.status = "Zoom simulation ready: check the fine cell size, then press Run.".into();
                    self.docs.push(d);
                    self.active = self.docs.len() - 1;
                }
                ShellCmd::Tutorial => { let mut d = self.new_doc(); d.start_tutorial(); self.docs.push(d); self.active = self.docs.len() - 1; }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_are_found_in_messages() {
        let m = "Air can leave the duct through something that is not a port, near (3.1, 13.8, 79.8) mm: a gap";
        assert_eq!(parse_coords(m), Some([3.1, 13.8, 79.8]));
        assert_eq!(parse_coords("no place here (just words)"), None);
        assert_eq!(parse_coords("near (-6.1, 23.0, -62.3) mm"), Some([-6.1, 23.0, -62.3]));
    }

    #[test]
    fn leak_issue_is_located_and_major_leak_reuses_it() {
        let leak = classify_issue("Air can leave the duct through something that is not a port, near (3.1, 13.8, 79.8) mm: x");
        let major = Issue { sev: Sev::Error, title: "Major leak: most of the air is not leaving through the outlets".into(), detail: "93 L/min".into() };
        let p = Project::new("t");
        let all = vec![leak.clone(), major.clone()];
        assert_eq!(issue_point(&leak, &all, &p), Some([3.1, 13.8, 79.8]));
        assert_eq!(issue_point(&major, &all, &p), Some([3.1, 13.8, 79.8]));
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    #[test]
    fn display_units() {
        assert!((cfm(28.3168) - 1.0).abs() < 1e-9);
        assert!((psi(6894.757) - 1.0).abs() < 1e-9);
        assert!((psi(12500.0) - 1.813).abs() < 1e-3); // WS9290 maximum static pressure
    }
}
