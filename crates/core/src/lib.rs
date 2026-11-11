//! Shared geometry types: triangle meshes, port caps, planar-face detection, STL I/O and test shapes.

pub mod mesh;
pub mod planar;
pub mod port_cap;
pub mod rim;
pub mod shapes;
pub mod stl;
pub mod vec3;

pub use mesh::{Aabb, TriMesh};
pub use planar::{find_port_candidates, planar_regions, PlanarRegion, PortCandidate};
pub use port_cap::PortCap;
pub use rim::{cap_from_loop, cap_from_loop_with_grid, external_visibility, inward_clearance, find_curved_openings, find_curved_openings_with, find_exposure_openings, smooth_region_loops, triangulate_loop, CurvedOpening, CurvedParams, EdgeGraph, RayGrid, RimLoop};
