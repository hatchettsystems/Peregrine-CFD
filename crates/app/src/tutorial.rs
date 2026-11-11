//! First-run tutorial: a built-in sample duct and a step-by-step guide through every major feature.
use super::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Act { DefineDuct, AddFan, AssignInlet, MakeOutlet, BlockHole, Run }

pub struct Step {
    pub title: &'static str,
    pub body: &'static str,
    pub tab: Option<Tab>,
    pub tool: Option<Tool>,
    pub act: Option<(&'static str, Act)>,
}

pub const STEPS: &[Step] = &[
    Step { title: "Welcome to Peregrine CFD", body: "This short tour uses a sample duct that is already loaded. It walks through every major feature: defining a duct, sources, ports, curved openings, blockers, running the simulation, and reading the results.\n\nYou can leave at any time with 'Skip tutorial', and rerun it from Help > Run tutorial.", tab: Some(Tab::Bodies), tool: Some(Tool::Orbit), act: None },
    Step { title: "1. Move around the 3D view", body: "Left-drag orbits, right- or middle-drag pans, the wheel zooms. Press M to cycle the mesh between Shaded, Ghost and Hidden. Hover a part to see its name; click it to find it in the Bodies list.", tab: Some(Tab::Bodies), tool: Some(Tool::Orbit), act: None },
    Step { title: "2. Define the duct", body: "The STEP file may hold many parts. You decide which ones form a duct: select parts in the Bodies list (or Ctrl-click them in the view) and press 'Define as duct'. A duct can be several parts; gaps up to 0.5 mm between them are sealed automatically.\n\nHere the duct is three parts: duct_main, duct_tip and screw_boss. The nozzle is not part of it.", tab: Some(Tab::Bodies), tool: Some(Tool::Orbit), act: Some(("Define it for me", Act::DefineDuct)) },
    Step { title: "3. Create a source", body: "Open the Ports tab. A source is what pushes the air: a fan with a real curve (the library has Sunon, Delta, blowers, axial fans), a flow rate, or a fixed speed. Use '+ Fan', '+ Flow' or '+ Velocity'.\n\nIf a fan feeds several openings, the flow is split by cross-section area, by weights, or by a shared plenum pressure.", tab: Some(Tab::Ports), tool: Some(Tool::Orbit), act: Some(("Add a fan for me", Act::AddFan)) },
    Step { title: "4. Assign the inlet", body: "The detected openings are listed under Openings and outlined in the view. Right-click an opening in the 3D view > Assign to source (or select it in the list). Do this for the square opening at the left end.", tab: Some(Tab::Ports), tool: Some(Tool::Orbit), act: Some(("Assign it for me", Act::AssignInlet)) },
    Step { title: "5. Mark the outlet (a curved opening)", body: "Not every opening is flat. This duct ends in a saddle-shaped scoop, and Peregrine finds it automatically. Right-click it > Pressure outlet.\n\nIf an opening is ever missed, use the Curved Opening tool (R) and click its rim, or the Draw Loop tool (L) to trace the edge by hand: click points along the rim and press Enter to close the loop.", tab: Some(Tab::Ports), tool: Some(Tool::Rim), act: Some(("Make it an outlet", Act::MakeOutlet)) },
    Step { title: "6. Block a hole", body: "Small holes such as screw holes must be shut or air would escape through them. Choose the Blocker tool (B) and click a small hole, or right-click any opening > Block. Missing hole? Draw its edge with the Draw Loop tool and press 'Add as blocker'.\n\nThe sample has a small screw hole (the little part under the duct).", tab: Some(Tab::Ports), tool: Some(Tool::Block), act: Some(("Block it for me", Act::BlockHole)) },
    Step { title: "7. Undo and redo", body: "Made a mistake? Ctrl+Z undoes and Ctrl+Shift+Z (or Ctrl+Y) redoes, for ports, sources, blockers and ducts. The Undo / Redo buttons at the top of the Ports tab do the same.", tab: Some(Tab::Ports), tool: Some(Tool::Orbit), act: None },
    Step { title: "8. Run the simulation", body: "Check Cell size, Steps and Backend in the toolbar (smaller cells are more accurate but slower; the GPU is used when available), then press Run. The progress bar is in the Results tab and you can keep working, or open another simulation tab with Ctrl+T.", tab: Some(Tab::Results), tool: Some(Tool::Orbit), act: Some(("Run it for me", Act::Run)) },
    Step { title: "9. Read the results", body: "The Flow report lists each port's flow, pressure and exit angle. Flow trajectories show where the air goes: set the number of lines per source, colour by speed or source, and use 'cutoff' to hide the slow tail of every line. Turn off 'Show ports, openings and blockers' to see only the flow, and press M to hide the mesh.", tab: Some(Tab::Results), tool: Some(Tool::Orbit), act: None },
    Step { title: "10. Exceptions", body: "Warnings and errors from a run live in the Exceptions tab. Each item has a coloured chip (red error, yellow warning, blue info) and expands to the full explanation. The tab shows how many there are.", tab: Some(Tab::Exceptions), tool: Some(Tool::Orbit), act: None },
    Step { title: "You are ready", body: "Save your work with File > Save Project, import your own STEP or STL with File > Open Geometry, and use tabs to compare variations (File > Duplicate This Tab).\n\nRerun this tour any time from Help > Run tutorial.", tab: Some(Tab::Bodies), tool: Some(Tool::Orbit), act: None },
];

fn flag_path() -> Option<PathBuf> {
    let base = std::env::var_os("APPDATA").or_else(|| std::env::var_os("XDG_CONFIG_HOME")).map(PathBuf::from).or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("peregrine-cfd").join("tutorial_done"))
}

pub fn tutorial_done() -> bool { flag_path().map_or(false, |p| p.exists()) }

pub fn mark_done() {
    if let Some(p) = flag_path() {
        if let Some(d) = p.parent() { let _ = std::fs::create_dir_all(d); }
        let _ = std::fs::write(p, "1");
    }
}

impl App {
    /// Load the sample duct and begin the tutorial.
    pub fn start_tutorial(&mut self) {
        self.reset_history();
        self.project = Project::new("Tutorial sample duct");
        self.project.solver.cell_size_mm = 0.5;
        self.project.solver.reference_speed = 0.0;
        self.project.solver.max_steps = 3000;
        self.project.domain.padding_mm = 4.0;
        self.last = None;
        self.run = None;
        self.lines.clear();
        self.manual_openings.clear();
        self.rim_loops.clear();
        self.block_cands.clear();
        let main = shapes::hollow_tube(14.0, 9.0, 40.0);
        let mut tip = shapes::scooped_round_tube(8.0, 4.5, 24.0, 3.0, 32);
        tip.translate([40.0, 0.0, 0.0]);
        let mut boss = shapes::hollow_tube(5.0, 2.0, 5.0);
        boss.translate([18.0, 0.0, -13.0]);
        let nozzle = shapes::solid_box([74.0, -7.0, -7.0], [84.0, 7.0, 7.0]);
        let bodies = vec![
            Body { name: "duct_main".into(), role: BodyRole::Duct, mesh: main },
            Body { name: "duct_tip".into(), role: BodyRole::Duct, mesh: tip },
            Body { name: "screw_boss".into(), role: BodyRole::Duct, mesh: boss },
            Body { name: "nozzle".into(), role: BodyRole::Solid, mesh: nozzle },
        ];
        self.set_bodies(bodies);
        self.mesh_mode = MeshMode::Ghost;
        self.tut = Some(0);
        self.enter_step(0);
    }

    fn enter_step(&mut self, i: usize) {
        let st = &STEPS[i];
        if let Some(t) = st.tab { self.tab = t; }
        if let Some(t) = st.tool { self.tool = t; if t == Tool::Block { self.block_stale = true; } }
        self.status = format!("Tutorial {}/{}: {}", i + 1, STEPS.len(), st.title);
    }

    fn end_tutorial(&mut self) {
        mark_done();
        self.tut = None;
        self.status = "Tutorial closed. Help > Run tutorial starts it again.".into();
    }

    pub fn tutorial_act(&mut self, act: Act) {
        match act {
            Act::DefineDuct => {
                let idx: Vec<usize> = ["duct_main", "duct_tip", "screw_boss"].iter().filter_map(|n| self.bodies.iter().position(|b| b.name == *n)).collect();
                self.define_duct_from(&idx, "sample_duct");
            }
            Act::AddFan => {
                if self.project.sources.is_empty() {
                    if let Some(p) = presets::fan_presets().into_iter().next() { self.add_fan_preset(&p); }
                }
            }
            Act::AssignInlet => {
                if self.project.sources.is_empty() { self.tutorial_act(Act::AddFan); }
                let src = self.project.sources.first().map(|s| s.name.clone());
                let cand = (0..self.candidates.len()).find(|&i| { let c = &self.candidates[i].1.cap; !c.is_surface() && c.normal[0] < -0.9 });
                if let (Some(s), Some(i)) = (src, cand) { self.assign_to_source(Target::Cand(i), &s); } else { self.status = "Define the duct first (step 2).".into(); }
            }
            Act::MakeOutlet => {
                let cand = (0..self.candidates.len()).filter(|&i| self.port_for_cand(i).is_none()).max_by(|&a, &b| self.candidates[a].1.cap.centroid()[0].partial_cmp(&self.candidates[b].1.cap.centroid()[0]).unwrap());
                if let Some(i) = cand { self.make_outlet(Target::Cand(i)); } else { self.status = "Define the duct first (step 2).".into(); }
            }
            Act::BlockHole => {
                self.refresh_block_cands();
                let pick = self.block_cands.iter().filter(|c| c.1.cap.centroid()[2] < -8.0).min_by(|a, b| a.1.equiv_diameter.partial_cmp(&b.1.equiv_diameter).unwrap()).cloned();
                if let Some((b, c)) = pick { self.add_blocker(b, c.cap); } else { self.status = "Define the duct first (step 2).".into(); }
            }
            Act::Run => self.start_run(),
        }
    }

    pub fn tutorial_ui(&mut self, ctx: &egui::Context) {
        let Some(i) = self.tut else { return };
        let st = &STEPS[i];
        let mut go: Option<isize> = None;
        let mut skip = false;
        let mut act: Option<Act> = None;
        egui::Window::new("Tutorial")
            .anchor(Align2::LEFT_BOTTOM, [56.0, -34.0])
            .collapsible(false)
            .resizable(false)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.set_width(380.0);
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(format!("Step {} of {}", i + 1, STEPS.len())).color(theme::TEXT_DIM));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| { if ui.small_button("Skip tutorial").clicked() { skip = true; } });
                });
                ui.add(egui::ProgressBar::new((i + 1) as f32 / STEPS.len() as f32).desired_height(4.0).fill(theme::ACCENT));
                ui.add_space(4.0);
                ui.label(egui::RichText::new(st.title).strong().size(15.0));
                ui.add_space(2.0);
                ui.label(st.body);
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    if let Some((label, a)) = st.act { if ui.button(label).clicked() { act = Some(a); } }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let last = i + 1 == STEPS.len();
                        if ui.add(egui::Button::new(egui::RichText::new(if last { "Finish" } else { "Next" }).color(Color32::WHITE)).fill(theme::ACCENT)).clicked() { go = Some(1); }
                        if i > 0 && ui.button("Back").clicked() { go = Some(-1); }
                    });
                });
            });
        if let Some(a) = act { self.tutorial_act(a); }
        if skip { self.end_tutorial(); return; }
        if let Some(d) = go {
            let n = i as isize + d;
            if n >= STEPS.len() as isize { self.end_tutorial(); } else if n >= 0 { self.tut = Some(n as usize); self.enter_step(n as usize); }
        }
    }
}
