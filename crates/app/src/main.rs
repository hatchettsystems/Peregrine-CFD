mod theme;
mod viewport;

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};
use peregrine_core::{shapes, PortCandidate};
use peregrine_geom::{build_project_domain, domain_bounds, duct_port_candidates, load_bodies, Body};
use peregrine_project::{BodyRole, Port, PortKind, Project};
use peregrine_solver::{Backend, PortReport, Simulation};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use viewport::Camera;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool { Orbit, Pan, PickPort, Slice }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab { Bodies, Ports, Results }
#[derive(Clone, Copy, PartialEq, Eq)]
enum Field { Speed, Pressure }

#[derive(Default)]
struct Snapshot {
    step: usize,
    total: usize,
    engine: String,
    dims: [usize; 3],
    origin: [f64; 3],
    dx: f64,
    cells: Vec<u8>,
    speed: Vec<f32>,
    pressure: Vec<f32>,
    report: Vec<PortReport>,
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
    new_name: String,
}

fn main() -> eframe::Result {
    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1360.0, 820.0]).with_title("Peregrine CFD"),
        ..Default::default()
    };
    eframe::run_native("Peregrine CFD", opts, Box::new(|cc| {
        theme::apply(&cc.egui_ctx);
        let mut app = App::new();
        if std::env::args().any(|a| a == "--demo") {
            app.load_demo();
            if std::env::args().any(|a| a == "--run") {
                app.project.solver.max_steps = 1750;
                app.backend = Backend::Cpu;
                app.start_run();
            }
        }
        Ok(Box::new(app))
    }))
}

impl App {
    fn new() -> App {
        App {
            project: Project::new("untitled"),
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
            new_name: String::new(),
        }
    }

    fn set_bodies(&mut self, bodies: Vec<Body>) {
        for b in &mut self.bodies {
            let _ = b;
        }
        self.visible = vec![true; bodies.len()];
        self.bodies = bodies;
        self.reclassify();
        let b = peregrine_geom::domain_bounds(&self.bodies, 0.0);
        if b.is_valid() {
            self.cam.frame(b.min, b.max);
        }
    }

    fn reclassify(&mut self) {
        for b in &mut self.bodies {
            b.role = self.project.geometry.classify(&b.name);
        }
        self.refresh_candidates();
    }

    fn refresh_candidates(&mut self) {
        self.candidates = duct_port_candidates(&self.bodies, 0.5);
        self.sel_cand = None;
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
        self.add_port_from(0, "inlet", PortKind::VelocityInlet { speed: 8.0 });
        self.add_port_from(1, "outlet", PortKind::PressureOutlet { pressure: 0.0 });
        self.status = "Demo loaded: straight duct with a nozzle block. Press Simulation > Run.".into();
    }

    fn open_geometry(&mut self, path: PathBuf) {
        match load_bodies(&path, &self.project) {
            Ok(b) => {
                let n = b.len();
                self.project.geometry_file = Some(path.clone());
                self.set_bodies(b);
                self.status = format!("Loaded {n} bodies from {}", path.display());
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
            match Project::load(&p) {
                Ok(pr) => {
                    self.project = pr;
                    self.project_path = Some(p.clone());
                    if let Some(g) = self.project.geometry_file.clone() {
                        let full = if g.is_absolute() { g } else { p.parent().unwrap_or(std::path::Path::new(".")).join(g) };
                        match load_bodies(&full, &self.project) {
                            Ok(b) => self.set_bodies(b),
                            Err(e) => self.status = format!("Geometry: {e:#}"),
                        }
                    }
                    self.status = format!("Opened {}", p.display());
                }
                Err(e) => self.status = format!("Open failed: {e:#}"),
            }
        }
    }

    fn can_run(&self) -> Result<(), String> {
        if self.bodies.is_empty() { return Err("no geometry loaded".into()); }
        if !self.project.ports.iter().any(|p| p.kind.is_inlet()) { return Err("define at least one inlet".into()); }
        if !self.project.ports.iter().any(|p| !p.kind.is_inlet()) { return Err("define at least one outlet".into()); }
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
                let mut sim = Simulation::new(&project, &domain, backend)?;
                {
                    let mut s = snap2.lock().unwrap();
                    s.dims = domain.grid.dims;
                    s.origin = domain.grid.origin;
                    s.dx = domain.grid.dx;
                    s.cells = domain.cells.clone();
                    s.engine = sim.engine_name();
                    s.inflation = sim.units.viscosity_inflation();
                    s.warnings = sim.units.warnings();
                    if domain.odd_columns > 0 {
                        s.warnings.push(format!("{} ray columns crossed a surface an odd number of times; a body may not be watertight.", domain.odd_columns));
                    }
                }
                let total = project.solver.max_steps;
                let mut done = 0;
                while done < total && !stop2.load(Ordering::Relaxed) {
                    let n = 250.min(total - done);
                    sim.advance(n);
                    done += n;
                    if !sim.is_finite() {
                        anyhow::bail!("solution diverged at step {done}; use a coarser cell or lower reference speed");
                    }
                    let u = sim.units;
                    let mac: Vec<[f32; 4]> = sim.mac().to_vec();
                    let cells = sim.cells().to_vec();
                    let speed = mac.iter().zip(&cells).map(|(m, &c)| if c == 1 { 0.0 } else { u.vel_to_phys((m[1] * m[1] + m[2] * m[2] + m[3] * m[3]).sqrt() as f64) as f32 }).collect();
                    let pressure = mac.iter().zip(&cells).map(|(m, &c)| if c == 1 { 0.0 } else { u.pressure(m[0] as f64) as f32 }).collect();
                    let report = sim.report();
                    let mut s = snap2.lock().unwrap();
                    s.step = done;
                    s.speed = speed;
                    s.pressure = pressure;
                    s.report = report;
                    s.version += 1;
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
        self.status = "Running...".into();
    }

    fn running(&self) -> bool {
        self.run.as_ref().map_or(false, |r| !r.snap.lock().unwrap().finished)
    }

    // ---------- UI pieces ----------

    fn menu_bar(&mut self, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
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
                if ui.button("Frame all").clicked() {
                    let b = domain_bounds(&self.bodies, 0.0);
                    if b.is_valid() { self.cam.frame(b.min, b.max); }
                    ui.close_menu();
                }
            });
            ui.menu_button("Geometry", |ui| {
                if ui.button("Re-run role rules").clicked() { self.project.geometry.overrides.clear(); self.reclassify(); ui.close_menu(); }
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
            ui.label(egui::RichText::new(match self.tool { Tool::Orbit => "Orbit", Tool::Pan => "Pan", Tool::PickPort => "Port Pick", Tool::Slice => "Slice" }).strong());
            ui.separator();
            ui.label("Cell size");
            ui.add(egui::DragValue::new(&mut self.project.solver.cell_size_mm).speed(0.01).range(0.05..=2.0).suffix(" mm"));
            ui.label("Ref. speed");
            ui.add(egui::DragValue::new(&mut self.project.solver.reference_speed).speed(0.1).range(1.0..=100.0).suffix(" m/s"));
            ui.label("Steps");
            ui.add(egui::DragValue::new(&mut self.project.solver.max_steps).speed(50.0).range(250..=500_000));
            ui.label("Padding");
            ui.add(egui::DragValue::new(&mut self.project.domain.padding_mm).speed(0.2).range(0.0..=100.0).suffix(" mm"));
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
            for (t, tip) in [(Tool::Orbit, "Orbit (O)"), (Tool::Pan, "Pan (H)"), (Tool::PickPort, "Port pick (P)"), (Tool::Slice, "Slice (S)")] {
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
        theme::section(ui, "Bodies", |ui| {
            if self.bodies.is_empty() {
                ui.label(egui::RichText::new("No geometry. File > Open Geometry...").color(theme::TEXT_DIM));
            }
            let mut changed = false;
            for i in 0..self.bodies.len() {
                ui.horizontal(|ui| {
                    let eye = if self.visible[i] { "o" } else { "-" };
                    if ui.small_button(eye).on_hover_text("Visibility").clicked() { self.visible[i] = !self.visible[i]; }
                    let name = self.bodies[i].name.clone();
                    ui.label(egui::RichText::new(&name).strong());
                    let mut role = self.bodies[i].role;
                    egui::ComboBox::from_id_salt(("role", i)).width(70.0).selected_text(format!("{role:?}")).show_ui(ui, |ui| {
                        for r in [BodyRole::Duct, BodyRole::Solid, BodyRole::Ignore] { ui.selectable_value(&mut role, r, format!("{r:?}")); }
                    });
                    if role != self.bodies[i].role {
                        self.project.geometry.overrides.insert(name, role);
                        changed = true;
                    }
                });
            }
            if changed { self.reclassify(); }
        });
        theme::section(ui, "Role rules (regex)", |ui| {
            let mut changed = false;
            for r in &mut self.project.geometry.rules {
                ui.horizontal(|ui| {
                    changed |= ui.add(egui::TextEdit::singleline(&mut r.pattern).desired_width(150.0).font(FontId::monospace(11.0))).changed();
                    egui::ComboBox::from_id_salt(&r.pattern).width(60.0).selected_text(format!("{:?}", r.role)).show_ui(ui, |ui| {
                        for k in [BodyRole::Duct, BodyRole::Solid, BodyRole::Ignore] { changed |= ui.selectable_value(&mut r.role, k, format!("{k:?}")).changed(); }
                    });
                });
            }
            if changed { self.reclassify(); }
        });
    }

    fn ports_panel(&mut self, ui: &mut egui::Ui) {
        theme::section(ui, "Detected openings", |ui| {
            if self.candidates.is_empty() { ui.label(egui::RichText::new("None. Duct bodies need planar faces with openings.").color(theme::TEXT_DIM)); }
            for i in 0..self.candidates.len() {
                let (body, c) = &self.candidates[i];
                let text = format!("#{i}  {body}   {:.1} mm2", c.area);
                if ui.selectable_label(self.sel_cand == Some(i), text).clicked() { self.sel_cand = Some(i); self.tool = Tool::PickPort; }
            }
            if let Some(i) = self.sel_cand {
                ui.separator();
                ui.horizontal(|ui| { ui.label("Name"); ui.text_edit_singleline(&mut self.new_name); });
                let name = if self.new_name.trim().is_empty() { format!("port_{i}") } else { self.new_name.trim().to_string() };
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Velocity inlet").clicked() { self.add_port_from(i, &name, PortKind::VelocityInlet { speed: 5.0 }); self.sel_cand = None; }
                    if ui.button("Flow inlet").clicked() { self.add_port_from(i, &name, PortKind::FlowInlet { l_per_min: 5.0 }); self.sel_cand = None; }
                    if ui.button("Fan inlet").clicked() { self.add_port_from(i, &name, PortKind::FanInlet { curve: vec![(0.0, 300.0), (20.0, 200.0), (40.0, 0.0)] }); self.sel_cand = None; }
                    if ui.button("Pressure outlet").clicked() { self.add_port_from(i, &name, PortKind::PressureOutlet { pressure: 0.0 }); self.sel_cand = None; }
                });
            }
        });
        theme::section(ui, "Boundary conditions", |ui| {
            let mut remove = None;
            for (i, p) in self.project.ports.iter_mut().enumerate() {
                let col = if p.kind.is_inlet() { theme::INLET } else { theme::OUTLET };
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("■").color(col));
                    if ui.selectable_label(self.sel_port == Some(i), &p.name).clicked() { self.sel_port = Some(i); }
                    ui.label(egui::RichText::new(&p.body).color(theme::TEXT_DIM));
                    if ui.small_button("x").on_hover_text("Remove").clicked() { remove = Some(i); }
                });
                ui.indent(i, |ui| match &mut p.kind {
                    PortKind::VelocityInlet { speed } => { ui.horizontal(|ui| { ui.label("Velocity"); ui.add(egui::DragValue::new(speed).speed(0.1).suffix(" m/s")); }); }
                    PortKind::FlowInlet { l_per_min } => { ui.horizontal(|ui| { ui.label("Flow"); ui.add(egui::DragValue::new(l_per_min).speed(0.1).suffix(" L/min")); }); }
                    PortKind::PressureOutlet { pressure } => { ui.horizontal(|ui| { ui.label("Pressure"); ui.add(egui::DragValue::new(pressure).speed(1.0).suffix(" Pa")); }); }
                    PortKind::FanInlet { curve } => {
                        ui.label("Fan curve: flow (L/min) -> pressure (Pa)");
                        let mut del = None;
                        for (j, (q, dp)) in curve.iter_mut().enumerate() {
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(q).speed(0.5).suffix(" L/min"));
                                ui.add(egui::DragValue::new(dp).speed(1.0).suffix(" Pa"));
                                if ui.small_button("-").clicked() { del = Some(j); }
                            });
                        }
                        if let Some(j) = del { if curve.len() > 2 { curve.remove(j); } }
                        if ui.small_button("+ point").clicked() { let l = *curve.last().unwrap(); curve.push((l.0 + 10.0, (l.1 - 20.0).max(0.0))); }
                    }
                });
            }
            if let Some(i) = remove { self.project.ports.remove(i); self.sel_port = None; }
            if self.project.ports.is_empty() { ui.label(egui::RichText::new("No ports yet. Pick an opening above.").color(theme::TEXT_DIM)); }
        });
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
        theme::section(ui, "Ports", |ui| {
            egui::Grid::new("rep").striped(true).show(ui, |ui| {
                for h in ["Port", "Flow L/min", "Pressure Pa"] { ui.label(egui::RichText::new(h).color(theme::TEXT_DIM)); }
                ui.end_row();
                for r in &s.report {
                    ui.label(egui::RichText::new(&r.name).color(if r.inlet { theme::INLET } else { theme::OUTLET }));
                    ui.label(format!("{:.2}", r.flow_l_min));
                    ui.label(format!("{:.1}", r.pressure_pa));
                    ui.end_row();
                }
            });
        });
        theme::section(ui, "Slice", |ui| {
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
        });
    }

    fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme::CANVAS);
        // input
        let d = resp.drag_delta();
        let pan = resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle) || (self.tool == Tool::Pan && resp.dragged_by(egui::PointerButton::Primary));
        if pan { self.cam.pan(rect, d.x, d.y); } else if resp.dragged_by(egui::PointerButton::Primary) && self.tool != Tool::PickPort { self.cam.orbit(d.x, d.y); }
        if resp.hovered() { self.cam.zoom(ui.input(|i| i.smooth_scroll_delta.y)); }

        // geometry
        for (i, b) in self.bodies.iter().enumerate() {
            if !self.visible[i] || b.role == BodyRole::Ignore { continue; }
            let (col, alpha) = match b.role { BodyRole::Duct => (Color32::from_rgb(0x8F, 0xB4, 0xD9), 150), _ => (Color32::from_rgb(0xB0, 0xB0, 0xB0), 255) };
            viewport::draw_mesh(&painter, rect, &self.cam, &b.mesh, col, alpha, false);
        }
        // slice
        if let Some(snap) = &self.last {
            let s = snap.lock().unwrap();
            if !s.speed.is_empty() { draw_slice(&painter, rect, &self.cam, &s, self.field, self.slice_axis, self.slice_pos); }
        }
        // port caps
        for (i, (_, c)) in self.candidates.iter().enumerate() {
            let sel = self.sel_cand == Some(i);
            viewport::draw_polygon(&painter, rect, &self.cam, &c.cap.polygon, Stroke::new(if sel { 2.5 } else { 1.0 }, if sel { theme::ACCENT } else { Color32::from_gray(200) }));
        }
        for (i, p) in self.project.ports.iter().enumerate() {
            let col = if p.kind.is_inlet() { theme::INLET } else { theme::OUTLET };
            viewport::draw_polygon(&painter, rect, &self.cam, &p.cap.polygon, Stroke::new(if self.sel_port == Some(i) { 3.5 } else { 2.5 }, col));
            if let Some((pos, _)) = self.cam.project(rect, p.cap.centroid()) {
                painter.text(pos + Vec2::new(8.0, -8.0), Align2::LEFT_BOTTOM, &p.name, FontId::proportional(11.0), col);
            }
        }
        // pick
        if self.tool == Tool::PickPort && resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                let mut best = (30.0f32, None);
                for (i, (_, c)) in self.candidates.iter().enumerate() {
                    if let Some((p, _)) = self.cam.project(rect, c.cap.centroid()) {
                        let dd = p.distance(pos);
                        if dd < best.0 { best = (dd, Some(i)); }
                    }
                }
                if best.1.is_some() { self.sel_cand = best.1; self.tab = Tab::Ports; }
            }
        }
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
                    let b = domain_bounds(&self.bodies, self.project.domain.padding_mm);
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

fn field_range(s: &Snapshot, f: Field) -> (f32, f32) {
    let v = if f == Field::Speed { &s.speed } else { &s.pressure };
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for (x, &c) in v.iter().zip(&s.cells) {
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
        Tool::Slice => {
            p.rect_stroke(Rect::from_center_size(c, Vec2::splat(16.0)), 0.0, s, egui::StrokeKind::Inside);
            p.line_segment([c - Vec2::new(8.0, -3.0), c + Vec2::new(8.0, 3.0)], s);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.running() { ctx.request_repaint_after(std::time::Duration::from_millis(100)); }
        ctx.input(|i| {
            if i.key_pressed(egui::Key::O) { self.tool = Tool::Orbit; }
            if i.key_pressed(egui::Key::H) { self.tool = Tool::Pan; }
            if i.key_pressed(egui::Key::P) { self.tool = Tool::PickPort; }
            if i.key_pressed(egui::Key::S) && !i.modifiers.ctrl { self.tool = Tool::Slice; }
        });
        let bar = egui::Frame::new().fill(theme::BAR).inner_margin(egui::Margin::symmetric(8, 3)).stroke(Stroke::new(1.0, theme::BORDER));
        egui::TopBottomPanel::top("menu").frame(bar).show(ctx, |ui| self.menu_bar(ui));
        egui::TopBottomPanel::top("options").frame(bar).exact_height(30.0).show(ctx, |ui| self.options_bar(ui));
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
        // finish bookkeeping
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
}
