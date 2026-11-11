//! Lattice-Boltzmann solver (D3Q19, BGK + Smagorinsky). CPU reference implementation.
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod lbm;
pub mod sim;
pub mod units;

pub use lbm::{Engine, Lbm, LbmConfig, PortBc, PortInfo};
pub use sim::{split_flows, Backend, SourceReport, PortReport, Simulation};
pub use units::Units;
