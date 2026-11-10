//! Lattice-Boltzmann solver (D3Q19, BGK + Smagorinsky). CPU reference implementation.
pub mod lbm;
pub mod sim;
pub mod units;

pub use lbm::{Lbm, LbmConfig, PortBc, PortInfo};
pub use sim::{PortReport, Simulation};
pub use units::Units;
