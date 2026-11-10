//! Adobe / Photoshop-style dark theme. All colours live here.
use eframe::egui;
use eframe::egui::{Color32, CornerRadius, FontId, Stroke, TextStyle};

pub const APP_BG: Color32 = Color32::from_rgb(0x32, 0x32, 0x32); // panels
pub const CANVAS: Color32 = Color32::from_rgb(0x28, 0x28, 0x28); // viewport
pub const DEEP: Color32 = Color32::from_rgb(0x1E, 0x1E, 0x1E); // inputs, wells
pub const BAR: Color32 = Color32::from_rgb(0x3B, 0x3B, 0x3B); // menu / options / status bars
pub const BORDER: Color32 = Color32::from_rgb(0x1A, 0x1A, 0x1A);
pub const TEXT: Color32 = Color32::from_rgb(0xC8, 0xC8, 0xC8);
pub const TEXT_DIM: Color32 = Color32::from_rgb(0x8C, 0x8C, 0x8C);
pub const ACCENT: Color32 = Color32::from_rgb(0x2D, 0x8C, 0xEB);
pub const WIDGET: Color32 = Color32::from_rgb(0x46, 0x46, 0x46);
pub const WIDGET_HOVER: Color32 = Color32::from_rgb(0x58, 0x58, 0x58);
pub const INLET: Color32 = Color32::from_rgb(0x3F, 0xB9, 0x50);
pub const OUTLET: Color32 = Color32::from_rgb(0xF0, 0x88, 0x3E);

pub fn apply(ctx: &egui::Context) {
    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT);
    v.panel_fill = APP_BG;
    v.window_fill = APP_BG;
    v.extreme_bg_color = DEEP;
    v.faint_bg_color = Color32::from_rgb(0x2C, 0x2C, 0x2C);
    v.window_stroke = Stroke::new(1.0, BORDER);
    v.selection.bg_fill = ACCENT;
    v.selection.stroke = Stroke::new(1.0, Color32::WHITE);
    v.hyperlink_color = ACCENT;
    let r = CornerRadius::same(2);
    for w in [&mut v.widgets.noninteractive, &mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
        w.corner_radius = r;
    }
    v.widgets.noninteractive.bg_fill = APP_BG;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.inactive.bg_fill = WIDGET;
    v.widgets.inactive.weak_bg_fill = WIDGET;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.bg_fill = WIDGET_HOVER;
    v.widgets.hovered.weak_bg_fill = WIDGET_HOVER;
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x70, 0x70, 0x70));
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.active.bg_fill = ACCENT;
    v.widgets.active.weak_bg_fill = ACCENT;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.window_corner_radius = CornerRadius::same(3);
    v.menu_corner_radius = CornerRadius::same(2);
    v.popup_shadow.color = Color32::from_black_alpha(120);
    v.window_shadow.color = Color32::from_black_alpha(120);
    ctx.set_visuals(v);

    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = egui::vec2(6.0, 4.0);
    style.spacing.button_padding = egui::vec2(8.0, 2.0);
    style.spacing.interact_size.y = 20.0;
    style.spacing.window_margin = egui::Margin::same(6);
    style.text_styles = [
        (TextStyle::Heading, FontId::proportional(13.0)),
        (TextStyle::Body, FontId::proportional(12.0)),
        (TextStyle::Button, FontId::proportional(12.0)),
        (TextStyle::Small, FontId::proportional(10.5)),
        (TextStyle::Monospace, FontId::monospace(11.5)),
    ]
    .into();
    ctx.set_style(style);
}

/// Panel section header (uppercase caption with a hairline), like a Photoshop panel group.
pub fn section(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(egui::RichText::new(title.to_uppercase()).size(10.5).color(TEXT_DIM).strong())
        .default_open(true)
        .show(ui, add);
    ui.add_space(2.0);
}

/// Photoshop-ish colour ramp for scalar fields (dark blue -> cyan -> green -> yellow -> red).
pub fn ramp(t: f32) -> Color32 {
    let stops = [(0.0, [0x1B, 0x2A, 0x6B]), (0.25, [0x1F, 0x8F, 0xD0]), (0.5, [0x3F, 0xB9, 0x50]), (0.75, [0xF2, 0xD0, 0x3A]), (1.0, [0xE0, 0x3A, 0x2F])];
    let t = t.clamp(0.0, 1.0);
    for w in stops.windows(2) {
        if t <= w[1].0 {
            let k = (t - w[0].0) / (w[1].0 - w[0].0);
            let c = |i: usize| (w[0].1[i] as f32 + k * (w[1].1[i] as f32 - w[0].1[i] as f32)) as u8;
            return Color32::from_rgb(c(0), c(1), c(2));
        }
    }
    Color32::WHITE
}
