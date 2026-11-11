//! Built-in fan library for common 3D-printer cooling fans.
//!
//! Each entry is described by its maximum static pressure (flow = 0) and maximum free-air flow
//! (pressure = 0), taken from distributor/datasheet listings where available. The full curve is
//! generated as `P(Q) = Pmax * (1 - (Q/Qmax)^n)`, which is a smooth approximation, not a measured curve.
//! Entries with `verified == false` use ESTIMATED numbers (no datasheet value was found) and the
//! app warns when they are used. Replace them with the figures from your own fan's datasheet.

const CFM: f64 = 28.3168; // L/min per CFM
const INH2O: f64 = 249.089; // Pa per inch of water
const MMH2O: f64 = 9.80665; // Pa per mm of water

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FanType {
    Blower,
    Axial,
}

#[derive(Clone, Debug)]
pub struct FanPreset {
    pub name: &'static str,
    pub kind: FanType,
    /// Maximum free-air flow (L/min).
    pub qmax_lpm: f64,
    /// Maximum static pressure (Pa).
    pub pmax_pa: f64,
    /// True when both numbers come from a datasheet/distributor listing.
    pub verified: bool,
    pub note: &'static str,
}

impl FanPreset {
    /// Pressure rise (Pa) versus flow (L/min), 9 points from shut-off to free delivery.
    pub fn curve(&self) -> Vec<(f64, f64)> {
        let n = match self.kind {
            FanType::Blower => 1.6,
            FanType::Axial => 1.3,
        };
        (0..=8)
            .map(|i| {
                let x = i as f64 / 8.0;
                (self.qmax_lpm * x, self.pmax_pa * (1.0 - x.powf(n)))
            })
            .collect()
    }
}

pub fn fan_presets() -> Vec<FanPreset> {
    use FanType::*;
    vec![
        FanPreset { name: "Sunon 5015 blower 12V (MF50151V1)", kind: Blower, qmax_lpm: 4.5 * CFM, pmax_pa: 0.670 * INH2O, verified: true, note: "DigiKey listing: 4.5 CFM, 0.670 inH2O" },
        FanPreset { name: "Sunon 5015 blower 12V low (MF50151V2)", kind: Blower, qmax_lpm: 3.5 * CFM, pmax_pa: 0.400 * INH2O, verified: true, note: "DigiKey listing: 3.5 CFM, 0.400 inH2O" },
        FanPreset { name: "Sunon 5015 blower 12V high (MF50151VX)", kind: Blower, qmax_lpm: 5.4 * CFM, pmax_pa: 0.970 * INH2O, verified: true, note: "DigiKey listing: 5.4 CFM, 0.970 inH2O (part is obsolete)" },
        FanPreset { name: "Delta BFB0524HH 5015 blower 24V", kind: Blower, qmax_lpm: 4.6 * CFM, pmax_pa: 0.90 * INH2O, verified: false, note: "Flow 4.6 CFM from DigiKey; pressure ESTIMATED" },
        FanPreset { name: "Delta 4028 blower", kind: Blower, qmax_lpm: 3.0 * CFM, pmax_pa: 0.80 * INH2O, verified: false, note: "ESTIMATE: no datasheet found; enter your fan's values" },
        FanPreset { name: "WS7040 blower 24V", kind: Blower, qmax_lpm: 250.0, pmax_pa: 6500.0, verified: false, note: "Pressure 6.5 kPa free-blowing (seller listing); flow ESTIMATED" },
        FanPreset { name: "WS9290 blower 24V (WS9290-24-220-X200)", kind: Blower, qmax_lpm: 47.0 * 1000.0 / 60.0, pmax_pa: 12500.0, verified: true, note: "Spec sheet: 47 m3/h (27.6 CFM), 12.5 kPa max static pressure, 22,500 rpm free air" },
        FanPreset { name: "WS9290B blower 48V (WS9290B-48-240-X200)", kind: Blower, qmax_lpm: 55.0 * 1000.0 / 60.0, pmax_pa: 14000.0, verified: true, note: "Spec sheet: 55 m3/h (32.4 CFM), 14 kPa max static pressure, 26,000 rpm free air" },
        FanPreset { name: "4010 blower 24V (typical)", kind: Blower, qmax_lpm: 2.0 * CFM, pmax_pa: 0.60 * INH2O, verified: false, note: "ESTIMATE: typical class value; enter your fan's values" },
        FanPreset { name: "Sunon 50x15 axial 12V (MF50151V1-1000U)", kind: Axial, qmax_lpm: 17.0 * CFM, pmax_pa: 0.230 * INH2O, verified: true, note: "DigiKey listing: 17.0 CFM, 0.230 inH2O" },
        FanPreset { name: "40x10 axial 24V (Orion OD4010 / NMB 04010SS)", kind: Axial, qmax_lpm: 6.7 * CFM, pmax_pa: 0.2 * INH2O, verified: true, note: "Distributor listings: 6.7 CFM, 0.2 inH2O (one datasheet says 0.1)" },
        FanPreset { name: "40x10 axial 24V high speed (DREMC DR-4010A-24T)", kind: Axial, qmax_lpm: 9.8 * CFM, pmax_pa: 8.6 * MMH2O, verified: true, note: "Datasheet: 9.8 CFM, 8.6 mmH2O" },
        FanPreset { name: "25x10 axial 5V (Orion OD2510, 12k rpm)", kind: Axial, qmax_lpm: 2.7 * CFM, pmax_pa: 0.23 * INH2O, verified: true, note: "Datasheet: 2.7 CFM, 0.23 inH2O" },
        FanPreset { name: "25x10 axial 12V (Orion OD2510-12HB02A)", kind: Axial, qmax_lpm: 2.7 * CFM, pmax_pa: 0.27 * INH2O, verified: true, note: "Datasheet: 2.7 CFM, 0.27 inH2O" },
    ]
}

pub fn find_preset(name: &str) -> Option<FanPreset> {
    fan_presets().into_iter().find(|p| p.name == name)
}

/// Warnings for sources that use estimated library curves.
pub fn preset_warnings(project: &crate::Project) -> Vec<String> {
    project
        .sources
        .iter()
        .filter_map(|s| s.preset.as_deref().and_then(find_preset).filter(|p| !p.verified).map(|p| format!("Source '{}': fan '{}' uses an ESTIMATED curve ({}). Replace it with datasheet values.", s.name, p.name, p.note)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn curves_are_monotone_and_hit_endpoints() {
        for p in fan_presets() {
            let c = p.curve();
            assert_eq!(c.len(), 9);
            assert!((c[0].1 - p.pmax_pa).abs() < 1e-9 && c[0].0 == 0.0, "{}", p.name);
            assert!(c[8].1.abs() < 1e-9 && (c[8].0 - p.qmax_lpm).abs() < 1e-9, "{}", p.name);
            assert!(c.windows(2).all(|w| w[1].1 <= w[0].1 && w[1].0 > w[0].0), "{}", p.name);
        }
    }
    #[test]
    fn sunon_numbers() {
        let p = find_preset("Sunon 5015 blower 12V (MF50151V1)").unwrap();
        assert!((p.pmax_pa - 166.9).abs() < 0.2 && (p.qmax_lpm - 127.4).abs() < 0.2);
    }
}
