/// Lowest relaxation time used; under-resolved cases get extra (implicit LES) viscosity.
pub const TAU_FLOOR: f64 = 0.52;

/// Conversion between SI and lattice units.
#[derive(Clone, Copy, Debug)]
pub struct Units {
    pub dx: f64,
    pub dt: f64,
    pub rho: f64,
    pub nu: f64,
}

impl Units {
    /// `u_ref` (m/s) is mapped to `u_lat` in lattice units (about 0.05 keeps Mach low).
    pub fn new(dx_m: f64, u_ref: f64, u_lat: f64, rho: f64, nu: f64) -> Units {
        Units { dx: dx_m, dt: u_lat * dx_m / u_ref, rho, nu }
    }
    pub fn nu_lat(&self) -> f64 { self.nu * self.dt / (self.dx * self.dx) }
    pub fn tau(&self) -> f64 { 3.0 * self.nu_lat() + 0.5 }
    /// Relaxation time actually used: physical tau raised to `TAU_FLOOR` for stability.
    pub fn tau_used(&self) -> f64 { self.tau().max(TAU_FLOOR) }
    /// How much larger the simulated viscosity is than the physical one (1 = exact).
    pub fn viscosity_inflation(&self) -> f64 { (self.tau_used() - 0.5) / (self.tau() - 0.5) }
    pub fn vel_to_lat(&self, u: f64) -> f64 { u * self.dt / self.dx }
    pub fn vel_to_phys(&self, u: f64) -> f64 { u * self.dx / self.dt }
    fn p_scale(&self) -> f64 { self.rho * (self.dx / self.dt).powi(2) }
    /// Gauge pressure (Pa) from lattice density.
    pub fn pressure(&self, rho_lat: f64) -> f64 { (rho_lat - 1.0) / 3.0 * self.p_scale() }
    pub fn rho_lat(&self, pressure: f64) -> f64 { 1.0 + 3.0 * pressure / self.p_scale() }
    /// Human-readable warnings about the chosen discretisation.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = vec![];
        if self.tau() < TAU_FLOOR {
            w.push(format!("cells are too coarse for the physical viscosity (tau = {:.4}); using tau = {TAU_FLOOR} which raises effective viscosity {:.0}x. Results are qualitative; refine the grid for accuracy.", self.tau(), self.viscosity_inflation()));
        }
        if self.tau() > 2.0 {
            w.push(format!("tau = {:.3} is large; accuracy suffers. Use a finer cell.", self.tau()));
        }
        w
    }
}
