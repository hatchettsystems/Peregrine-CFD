# Peregrine CFD - plan

Airflow simulation for 3D-printer part-cooling ducts. Import a STEP file (a single duct or the whole
toolhead), find the duct bodies by name, let the user define inlets and outlets, run a GPU
lattice-Boltzmann solver and inspect velocity and pressure. Heat transfer is out of scope.

## Decisions

| Topic | Decision |
|---|---|
| Duct model | Hollow solids: the duct is a solid with an internal channel. Ports are the channel openings. |
| Physics | Incompressible, isothermal air. Velocity and pressure only. |
| Target | Windows, RTX 3070 (8 GB). wgpu compute (DX12/Vulkan), shaders in WGSL. |
| Solver | D3Q19 LBM, BGK + Smagorinsky LES, half-way bounce-back walls, equilibrium velocity/pressure ports. |
| CAD | OpenCASCADE for STEP. A small C++/cxx shim (`crates/occt`, planned) reads names + per-solid meshes. STL is accepted as a fallback. |
| UI | egui/eframe on wgpu (milestone 6). A CLI drives the same project file. |
| Language | Rust. C++ only for the OCCT shim, WGSL for GPU kernels. |

## UI look and layout (Adobe / Photoshop style)

- Dark neutral palette (about #323232 panels, #1E1E1E canvas, #535353 borders), 1 px hairline separators,
  compact 11-12 px type, one blue accent (#2D8CEB) for selection and focus. No gradients.
- Layout: menu bar (File Edit View Geometry Simulation Window Help) on top, a context-sensitive
  Options bar under it, a narrow vertical Tools palette on the left (Select, Orbit/Pan, Pick Face,
  Define Port, Slice Plane, Probe), the 3D viewport in the centre, and docked panels on the right
  (Bodies = "Layers", Ports/BCs = "Properties", Results = "Adjustments") with tabs and collapsible groups.
- Bottom status bar for cell count, VRAM estimate and solver progress. Body visibility eye icons and
  role badges in the Bodies panel behave like Photoshop layers.
- All colours live in one `theme.rs`, so the look is a single file to adjust.

## Pipeline

STEP -> bodies (name, mesh) -> role by name rules -> planar-face hole detection (port candidates)
-> user assigns port types -> voxelize + port caps -> LBM -> report/VTK.

## Crates

- `core`    - TriMesh, PortCap, planar face/hole detection, STL reader, test shapes.
- `project` - serde/TOML project file: role rules, ports, fluid, domain, solver settings.
- `geom`    - body loading and role classification (STL now, STEP via OCCT shim).
- `voxel`   - parity voxelizer, port caps, leak diagnostics, flood-fill of the fluid region.
- `solver`  - lattice units, CPU reference LBM (validated), fan-curve controller; GPU backend later.
- `post`    - port report (flow, pressure), VTK export.
- `cli`     - `peregrine demo | ports | run`.

## Milestones

1. [x] Workspace, project file, mesh core, port detection, voxelizer, CPU LBM, VTK, CLI, tests.
2. [ ] STEP import through OCCT shim: assembly tree, names, units, per-solid meshes.
3. [ ] wgpu LBM backend (D3Q19, single-buffer AA pattern, ~8 GB budget -> ~25M cells), parity-tested against CPU.
4. [ ] egui app (Photoshop-style theme, see above): 3D viewport, body tree + role overrides, port picking, BC panel, live run view.
5. [ ] Results: slice planes, streamlines, per-port report, screenshots.
6. [ ] Local grid refinement near nozzle, Bouzidi walls, validation against OpenFOAM.

## Status (end of first session)

Working and tested on the CPU: mesh core, planar-hole port detection, project file, parity voxelizer with
port caps, D3Q19 solver (validated against laminar square-duct pressure drop, mass flux conserved),
VTK export, and a CLI (`demo`, `ports`, `set-port`, `run`). Input today is STL; STEP waits on the OCCT shim.

Known accuracy limit: at 0.5 mm cells and 8-12 m/s the physical viscosity gives tau ~ 0.5004, which BGK cannot
run stably. The solver raises tau to 0.55 (extra viscosity) and says so; flow patterns are usable, absolute
pressure drops are not. Fix before trusting numbers: TRT/regularized or cumulant collision, lattice speed 0.1,
finer cells on the GPU, interpolated walls, non-equilibrium-extrapolation inlets (the inlet currently delivers
about 7% less than the set speed, which the port report shows honestly).

## Risks

- OCCT build time and Windows toolchain (MSVC + CMake). Mitigation: STL fallback, prebuilt artifact in CI.
- Resolution: duct features of 1-2 mm need about 0.15-0.25 mm cells. The tool prints a cell/VRAM estimate first.
- Non-watertight CAD leaks the parity voxelizer. The voxelizer reports odd-parity columns.
- Staircase walls limit wall-shear accuracy. Results are engineering estimates; validate before trusting numbers.
