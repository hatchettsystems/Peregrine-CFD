// Release builds on Windows are GUI-subsystem apps (no console window).
#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

mod theme;
mod viewport;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use peregrine_core::port_cap::point_in_polygon;
use peregrine_core::vec3::{dot, len, sub};
use peregrine_core::{cap_from_loop_with_grid, external_visibility, inward_clearance, shapes, smooth_region_loops, PortCandidate, PortCap, RayGrid, TriMesh};
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
enum Tool { Orbit, Pan, PickPort, Slice, Rim }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab { Bodies, Ports, Results }
/// What the active tab asks the tab shell to do after the frame.
enum ShellCmd { New, Duplicate, Close(usize), Switch(usize) }

/// An edge loop found by the Rim tool with its quality scores.
#[derive(Clone)]
struct RimInfo { l: peregrine_core::RimLoop, vis: f64, clear: f64 }

#[derive(Clone)]
struct TabInfo { title: String, running: bool }

#[derive(Clone, Copy, PartialEq, Eq)]
enum MeshMode { Shaded, Ghost, Hidden }
#[derive(Clone, Copy, PartialEq, Eq)]
enum LineColor { Speed, Source }
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
    lines_request_sig: u64,
    lines_job: Option<std::sync::mpsc::Receiver<(u64, Vec<Streamline>)>>,
    // tab shell plumbing
    tab_infos: Vec<TabInfo>,
    tab_active: usize,
    cmds: Vec<ShellCmd>,
}

fn main() -> eframe::Result {
    let mut viewport = egui::ViewportBuilder::default().with_inner_size([1360.0, 820.0]).with_title("Peregrine CFD");
    if let Ok(icon) = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon256.png")) {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }
    let opts = eframe::NativeOptions { viewport, ..Default::default() };
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
            lines_request_sig: 0,
            lines_job: None,
            tab_infos: vec![],
            tab_active: 0,
            cmds: vec![],
        }
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        self.visible = vec![true; bodies.len()];
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
        if !self.show_hidden { self.candidates.retain(|c| c.1.visibility >= 0.5 && c.1.clearance >= 0.5); }
        self.candidates.extend(self.manual_openings.iter().cloned());
        self.sel_cand = None;
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
                let cand = PortCandidate { area, parent_area: 0.0, circularity: (4.0 * std::f64::consts::PI * area / (perim * perim)).min(1.0), equiv_diameter: 2.0 * (area / std::f64::consts::PI).sqrt(), visibility: info.vis, clearance: info.clear, cap };
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

    fn open_geometry(&mut self, path: PathBuf) {
        match load_bodies(&path, &self.project) {
            Ok(b) => {
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
        if problems.is_empty() { Ok(()) } else { Err(problems.join("; ")) }
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
                        s.warnings = base_warnings.clone();
                        s.warnings.extend(notes.clone());
                        if sim.ref_speed > ref_speed * 1.01 {
                            s.warnings.push(format!("Reference speed raised to {:.1} m/s: the estimated peak flow speed is higher than the project setting.", sim.ref_speed));
                        }
                        s.warnings.extend(sim.units.warnings());
                    }
                    let mut done = 0;
                    while done < total && !stop2.load(Ordering::Relaxed) {
                        let n = 250.min(total - done);
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
                        let mac: Vec<[f32; 4]> = sim.mac().to_vec();
                        let solid = |i: usize| cells[i] == 1;
                        let speed: Vec<f32> = mac.iter().enumerate().map(|(i, m)| if solid(i) { 0.0 } else { u.vel_to_phys((m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() as f64) as f32 }).collect();
                        let pressure: Vec<f32> = mac.iter().enumerate().map(|(i, m)| if solid(i) { 0.0 } else { u.pressure(m[0] as f64) as f32 }).collect();
                        let vel: Vec<[f32; 3]> = mac.iter().enumerate().map(|(i, m)| if solid(i) { [0.0; 3] } else { [u.vel_to_phys(m[1] as f64) as f32, u.vel_to_phys(m[2] as f64) as f32, u.vel_to_phys(m[3] as f64) as f32] }).collect();
                        let report = sim.report();
                        let sources = sim.source_reports();
                        let mut s = snap2.lock().unwrap();
                        s.sources = sources;
                        s.step = done;
                        s.speed = speed;
                        s.pressure = pressure;
                        s.vel = Arc::new(vel);
                        s.report = report;
                        s.version += 1;
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
            let (dims, origin, dx, cells, vel) = { let s = snap.lock().unwrap(); (s.dims, s.origin, s.dx, s.cells.clone(), s.vel.clone()) };
            let field = VelocityField { dims, origin, dx, cells: &cells, vel: &vel };
            let lines = sl::streamlines(&field, &specs, 600.0);
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
            });
        });
    }

    fn options_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            ui.label(egui::RichText::new(match self.tool { Tool::Orbit => "Orbit", Tool::Pan => "Pan", Tool::PickPort => "Port Pick", Tool::Slice => "Slice", Tool::Rim => "Curved Opening" }).strong());
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
            for (t, tip) in [(Tool::Orbit, "Orbit (O)"), (Tool::Pan, "Pan (H)"), (Tool::PickPort, "Port pick (P)"), (Tool::Rim, "Curved opening: click the rim or the wall at a non-flat opening (R)"), (Tool::Slice, "Slice (S)")] {
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

    fn pick_body_tri(&self, rect: Rect, pos: Pos2) -> Option<(usize, usize)> {
        let proj = self.cam.projector(rect);
        let mut best: Option<(f64, usize, usize)> = None;
        for (bi, b) in self.bodies.iter().enumerate() {
            if !self.visible[bi] || b.role == BodyRole::Ignore { continue; }
            let bb = b.mesh.bounds();
            // cheap reject: screen box of the bounding box
            let corners = [[bb.min[0], bb.min[1], bb.min[2]], [bb.max[0], bb.min[1], bb.min[2]], [bb.min[0], bb.max[1], bb.min[2]], [bb.max[0], bb.max[1], bb.min[2]], [bb.min[0], bb.min[1], bb.max[2]], [bb.max[0], bb.min[1], bb.max[2]], [bb.min[0], bb.max[1], bb.max[2]], [bb.max[0], bb.max[1], bb.max[2]]];
            let pc: Vec<Pos2> = corners.iter().filter_map(|c| proj.project(*c).map(|x| x.0)).collect();
            if pc.len() == 8 {
                let (mut lo, mut hi) = (Pos2::new(f32::MAX, f32::MAX), Pos2::new(f32::MIN, f32::MIN));
                for q in &pc { lo.x = lo.x.min(q.x); lo.y = lo.y.min(q.y); hi.x = hi.x.max(q.x); hi.y = hi.y.max(q.y); }
                if pos.x < lo.x - 2.0 || pos.x > hi.x + 2.0 || pos.y < lo.y - 2.0 || pos.y > hi.y + 2.0 { continue; }
            }
            for ti in 0..b.mesh.triangles.len() {
                let [a, bb2, c] = b.mesh.tri(ti);
                let (Some((pa, da)), Some((pb, _)), Some((pc2, _))) = (proj.project(a), proj.project(bb2), proj.project(c)) else { continue };
                // point in 2D triangle
                let sign = |p1: Pos2, p2: Pos2, p3: Pos2| (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
                let (d1, d2, d3) = (sign(pos, pa, pb), sign(pos, pb, pc2), sign(pos, pc2, pa));
                let neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
                let posv = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
                if neg && posv { continue; }
                if best.map_or(true, |x| da < x.0) { best = Some((da, bi, ti)); }
            }
        }
        best.map(|x| (x.1, x.2))
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

    fn hit_test(&self, rect: Rect, pos: Pos2) -> Option<Target> {
        let mut items: Vec<(Target, &PortCap)> = vec![];
        for (i, (_, c)) in self.candidates.iter().enumerate() { items.push((Target::Cand(i), &c.cap)); }
        for (i, p) in self.project.ports.iter().enumerate() { items.push((Target::Port(i), &p.cap)); }
        let mut inside: Option<(f64, Target)> = None;
        let mut near: Option<(f32, Target)> = None;
        for (t, cap) in items {
            let pts: Vec<Option<(Pos2, f64)>> = cap.polygon.iter().map(|p| self.cam.project(rect, *p)).collect();
            if pts.iter().any(|p| p.is_none()) || pts.len() < 3 { continue; }
            let poly: Vec<[f64; 2]> = pts.iter().map(|p| { let q = p.unwrap().0; [q.x as f64, q.y as f64] }).collect();
            let depth = self.cam.project(rect, cap.centroid()).map_or(f64::MAX, |p| p.1);
            if point_in_polygon(&poly, [pos.x as f64, pos.y as f64]) && inside.map_or(true, |b| depth < b.0) { inside = Some((depth, t)); }
            if let Some((c, _)) = self.cam.project(rect, cap.centroid()) {
                let d = c.distance(pos);
                if d < 18.0 && near.map_or(true, |b| d < b.0) { near = Some((d, t)); }
            }
        }
        inside.map(|b| b.1).or(near.map(|b| b.1)).map(|t| self.resolve(t))
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
        if let Target::Port(i) = t {
            ui.separator();
            if ui.button("Show in Ports panel").clicked() { self.sel_port = Some(i); self.tab = Tab::Ports; ui.close_menu(); }
            if ui.button("Remove port").clicked() { self.project.ports.remove(i); self.sel_port = None; ui.close_menu(); }
        }
    }

    fn ports_panel(&mut self, ui: &mut egui::Ui) {
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
                    SourceKind::Flow { l_per_min } => { ui.horizontal(|ui| { ui.label("Total flow"); ui.add(egui::DragValue::new(l_per_min).speed(0.1).range(0.0..=2000.0).suffix(" L/min")); }); }
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
                                edited |= ui.add(egui::DragValue::new(q).speed(0.5).suffix(" L/min")).changed();
                                edited |= ui.add(egui::DragValue::new(dp).speed(1.0).suffix(" Pa")).changed();
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
        if !self.rim_loops.is_empty() {
            ui.separator();
            ui.label(egui::RichText::new("Edge loops at your click").strong());
            let mut add: Option<usize> = None;
            for li in 0..self.rim_loops.len() {
                let info = &self.rim_loops[li];
                let l = &info.l;
                let kind = if l.planarity_dev > 0.05 { format!("curved, {:.1} mm off-plane", l.planarity_dev) } else { "flat".to_string() };
                let text = format!("loop {}  {:.0} mm around  ({kind})  visible {:.0}%, air behind {:.0}%", li + 1, l.perimeter, info.vis * 100.0, info.clear * 100.0);
                ui.horizontal(|ui| {
                    if ui.selectable_label(self.rim_sel == Some(li), text).clicked() { self.rim_sel = Some(li); }
                    if self.rim_sel == Some(li) && ui.button("Add as opening").clicked() { add = Some(li); }
                });
            }
            if let Some(li) = add { self.add_rim_opening(li); }
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
        for i in 0..self.candidates.len() {
            let status = match self.port_for_cand(i).map(|k| &self.project.ports[k].kind) {
                Some(PortKind::PressureOutlet { .. }) => "outlet".to_string(),
                Some(k) => format!("inlet <- {}", k.source().unwrap_or("?")),
                None => "unassigned".to_string(),
            };
            let (body, c) = &self.candidates[i];
            let hidden = if c.visibility < 0.5 { "  hidden" } else if c.clearance < 0.5 { "  pocket" } else { "" };
            let text = format!("#{i}  {body}  {:.1} mm²{}{}  [{status}]", c.area, if c.cap.is_surface() { "  curved" } else { "" }, hidden);
            if ui.selectable_label(self.sel_cand == Some(i), text).clicked() { self.sel_cand = Some(i); self.sel_port = self.port_for_cand(i); }
        }
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
        for i in 0..self.project.ports.len() {
            let sel = self.sel_port == Some(i);
            let p = &mut self.project.ports[i];
            if let PortKind::PressureOutlet { pressure } = &mut p.kind {
                any = true;
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("■").color(theme::OUTLET));
                    if ui.selectable_label(sel, &p.name).clicked() { self.sel_port = Some(i); }
                    ui.add(egui::DragValue::new(pressure).speed(1.0).suffix(" Pa"));
                    if ui.small_button("x").clicked() { remove = Some(i); }
                });
            }
        }
        if !any { ui.label(egui::RichText::new("No outlets yet. Right-click an opening > Pressure outlet.").color(theme::TEXT_DIM)); }
        if let Some(i) = remove { self.project.ports.remove(i); self.sel_port = None; }
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
            if let Some(e) = &s.error { ui.colored_label(Color32::from_rgb(0xE0, 0x5A, 0x4F), e); }
            for w in &s.warnings { ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), w); }
            if s.inflation > 3.0 { ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), format!("Viscosity is {:.0}x physical at this cell size: patterns are qualitative, pressures overestimated.", s.inflation)); }
        });
        theme::section(ui, "Flow report", |ui| {
            egui::Grid::new("rep").striped(true).show(ui, |ui| {
                for h in ["Port", "Flow L/min", "Pressure Pa", "Share", "Exit speed / angle"] { ui.label(egui::RichText::new(h).color(theme::TEXT_DIM)); }
                ui.end_row();
                for g in &s.sources {
                    ui.label(egui::RichText::new(&g.name).strong().color(theme::INLET));
                    ui.label(format!("{:.2}", g.total_flow_l_min));
                    ui.label(format!("{:.1}", g.pressure_pa));
                    ui.label(match g.split { SplitMode::Plenum => format!("plenum {:.0} Pa", g.commanded), _ => format!("cmd {:.1}", g.commanded) });
                    ui.label("");
                    ui.end_row();
                    for r in s.report.iter().filter(|r| r.source.as_deref() == Some(g.name.as_str())) {
                        ui.label(format!("   {}", r.name));
                        ui.label(format!("{:.2}", r.flow_l_min));
                        ui.label(format!("{:.1}", r.pressure_pa));
                        ui.label(format!("{:.0}%", 100.0 * r.flow_l_min / g.total_flow_l_min.abs().max(1e-9)));
                        ui.label("");
                        ui.end_row();
                    }
                }
                for r in s.report.iter().filter(|r| r.source.is_none()) {
                    ui.label(egui::RichText::new(&r.name).color(if r.inlet { theme::INLET } else { theme::OUTLET }));
                    ui.label(format!("{:.2}", r.flow_l_min));
                    ui.label(format!("{:.1}", r.pressure_pa));
                    ui.label("");
                    ui.label(if r.inlet { String::new() } else { format!("{:.1} m/s  {:.0}° from normal", r.mean_speed, r.angle_deg) });
                    ui.end_row();
                }
            });
        });
        {
            let q_in: f64 = s.report.iter().filter(|r| r.inlet).map(|r| r.flow_l_min).sum();
            let net: f64 = s.report.iter().map(|r| r.flow_l_min).sum();
            if q_in > 1e-6 && net.abs() > 0.08 * q_in && s.step > 2000 {
                ui.colored_label(Color32::from_rgb(0xF2, 0xD0, 0x3A), format!("{:.1} L/min of the {:.1} L/min going in is not accounted for by your outlets. Air is probably leaving through duct openings you did not define as ports (or the run has not settled yet).", net.abs(), q_in));
            }
        }
        theme::section(ui, "Display", |ui| {
            ui.horizontal(|ui| {
                ui.label("Mesh");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Shaded, "Shaded");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Ghost, "Ghost");
                ui.selectable_value(&mut self.mesh_mode, MeshMode::Hidden, "Hidden");
            });
            ui.checkbox(&mut self.show_lines, "Flow trajectories (streamlines)");
            if self.show_lines {
                ui.horizontal(|ui| {
                    ui.label("Colour by");
                    ui.selectable_value(&mut self.line_color, LineColor::Speed, "Speed");
                    ui.selectable_value(&mut self.line_color, LineColor::Source, "Source");
                });
                ui.add(egui::Slider::new(&mut self.line_width, 0.5..=5.0).text("width"));
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
                ui.label(egui::RichText::new(format!("{} lines drawn{}", self.lines.len(), if self.lines_job.is_some() { " (updating...)" } else { "" })).color(theme::TEXT_DIM));
            }
            ui.checkbox(&mut self.show_slice, "Slice plane");
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
                ui.horizontal(|ui| { ui.label(format!("{lo:.1}")); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(format!("{hi:.1} {}", if self.field == Field::Speed { "m/s" } else { "Pa" }))); });
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

        // geometry
        for (i, b) in self.bodies.iter().enumerate() {
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
                    (BodyRole::Duct, false) => (Color32::from_rgb(0x8F, 0xB4, 0xD9), 150),
                    _ => (Color32::from_rgb(0xB0, 0xB0, 0xB0), 255),
                }
            };
            viewport::draw_mesh(&painter, rect, &self.cam, &b.mesh, col, alpha, false);
        }
        // slice
        if let (true, Some(snap)) = (self.show_slice, &self.last) {
            let s = snap.lock().unwrap();
            if !s.speed.is_empty() { draw_slice(&painter, rect, &self.cam, &s, self.field, self.slice_axis, self.slice_pos); }
        }
        if self.show_lines && !self.lines.is_empty() {
            draw_streamlines(&painter, rect, &self.cam, &self.lines, self.line_color, self.lines_vmax, self.line_width);
        }
        // port caps: unassigned openings in grey, assigned ports coloured
        for (i, (_, c)) in self.candidates.iter().enumerate() {
            if self.port_for_cand(i).is_some() { continue; }
            let sel = self.sel_cand == Some(i);
            viewport::draw_polygon(&painter, rect, &self.cam, &c.cap.polygon, Stroke::new(if sel { 2.5 } else { 1.0 }, if sel { theme::ACCENT } else { Color32::from_gray(200) }));
        }
        for (i, p) in self.project.ports.iter().enumerate() {
            let col = if p.kind.is_inlet() { theme::INLET } else { theme::OUTLET };
            viewport::draw_polygon(&painter, rect, &self.cam, &p.cap.polygon, Stroke::new(if self.sel_port == Some(i) { 3.5 } else { 2.5 }, col));
            if let Some((pos, _)) = self.cam.project(rect, p.cap.centroid()) {
                let label = match p.kind.source() { Some(sname) => format!("{} <- {}", p.name, sname), None => p.name.clone() };
                painter.text(pos + Vec2::new(8.0, -8.0), Align2::LEFT_BOTTOM, label, FontId::proportional(11.0), col);
            }
        }
        // loops found by the Rim tool
        for (li, l) in self.rim_loops.iter().enumerate() {
            let sel = self.rim_sel == Some(li);
            viewport::draw_polygon(&painter, rect, &self.cam, &l.l.points, Stroke::new(if sel { 3.5 } else { 1.5 }, if sel { Color32::from_rgb(0xFF, 0xD0, 0x40) } else { Color32::from_rgb(0xB0, 0x90, 0x30) }));
        }
        // hover highlight, left-click select, right-click menu
        let hover = if resp.hovered() { resp.hover_pos().and_then(|pp| self.hit_test(rect, pp)) } else { None };
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
        if resp.clicked() && self.tool == Tool::Rim {
            if let Some(pp) = resp.interact_pointer_pos() {
                match self.pick_body_tri(rect, pp) {
                    Some((bi, ti)) => self.rim_pick(bi, ti),
                    None => self.status = "Click on a part's surface near the curved opening.".into(),
                }
            }
        } else if resp.clicked() {
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
                if let Some(l) = &self.last {
                    let s = l.lock().unwrap();
                    if s.finished && s.error.is_none() { ui.label("Done"); }
                }
            });
        });
    }
}

fn source_color(i: usize) -> Color32 {
    const PAL: [(u8, u8, u8); 6] = [(0x2D, 0xC8, 0xF0), (0xF0, 0x88, 0x3E), (0x7C, 0xE0, 0x5A), (0xE0, 0x5A, 0xA0), (0xB0, 0x8C, 0xF0), (0xF2, 0xD0, 0x3A)];
    let c = PAL[i % PAL.len()];
    Color32::from_rgb(c.0, c.1, c.2)
}

fn draw_streamlines(painter: &egui::Painter, rect: Rect, cam: &Camera, lines: &[Streamline], mode: LineColor, vmax: f32, width: f32) {
    let mut shapes: Vec<Shape> = Vec::with_capacity(lines.len() * 8);
    for l in lines {
        let pts: Vec<Option<Pos2>> = l.pts.iter().map(|p| cam.project(rect, [p[0] as f64, p[1] as f64, p[2] as f64]).map(|x| x.0)).collect();
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
            LineColor::Speed => {
                let mut i = 0;
                while i + 1 < n {
                    let j = (i + 4).min(n - 1);
                    let seg: Vec<Pos2> = pts[i..=j].iter().flatten().copied().collect();
                    if seg.len() == j - i + 1 {
                        let v = l.speed[i..=j].iter().sum::<f32>() / (j - i + 1) as f32;
                        shapes.push(Shape::line(seg, Stroke::new(width, theme::ramp(v / vmax))));
                    }
                    i = j;
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
        ctx.input(|i| {
            if i.modifiers.ctrl && i.key_pressed(egui::Key::T) { self.cmds.push(ShellCmd::New); }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::W) { self.cmds.push(ShellCmd::Close(self.tab_active)); }
            if i.modifiers.ctrl && i.key_pressed(egui::Key::Tab) && !self.tab_infos.is_empty() { self.cmds.push(ShellCmd::Switch((self.tab_active + 1) % self.tab_infos.len())); }
            if i.modifiers.ctrl { return; }
            if i.key_pressed(egui::Key::O) { self.tool = Tool::Orbit; }
            if i.key_pressed(egui::Key::H) { self.tool = Tool::Pan; }
            if i.key_pressed(egui::Key::P) { self.tool = Tool::PickPort; }
            if i.key_pressed(egui::Key::S) { self.tool = Tool::Slice; }
            if i.key_pressed(egui::Key::R) { self.tool = Tool::Rim; }
            if i.key_pressed(egui::Key::M) {
                self.mesh_mode = match self.mesh_mode { MeshMode::Shaded => MeshMode::Ghost, MeshMode::Ghost => MeshMode::Hidden, MeshMode::Hidden => MeshMode::Shaded };
            }
        });
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
                    for (t, n) in [(Tab::Bodies, "Bodies"), (Tab::Ports, "Ports"), (Tab::Results, "Results")] {
                        let sel = self.tab == t;
                        let text = egui::RichText::new(n).color(if sel { Color32::WHITE } else { theme::TEXT_DIM });
                        if ui.add(egui::Button::new(text).fill(if sel { theme::BAR } else { theme::APP_BG }).stroke(Stroke::NONE)).clicked() { self.tab = t; }
                    }
                });
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| match self.tab {
                    Tab::Bodies => self.bodies_panel(ui),
                    Tab::Ports => self.ports_panel(ui),
                    Tab::Results => self.results_panel(ui),
                });
            });
        egui::CentralPanel::default().frame(egui::Frame::new().fill(theme::CANVAS)).show(ctx, |ui| self.viewport_ui(ui));
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
            }
        }
    }
}
