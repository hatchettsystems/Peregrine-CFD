# Peregrine CFD

Airflow simulation for 3D-printer part-cooling ducts, written in Rust.

- Import a STEP assembly (a duct or a whole toolhead) or STL files. Bodies are found by name
  (`duct`, `duct_1`, `duct_L`, `duct_R`, ...; editable regex rules).
- Ducts are hollow solids. Their openings are detected automatically; you assign each as a velocity inlet,
  flow inlet, fan-curve inlet or pressure outlet. Any number of inlets and outlets.
- GPU lattice-Boltzmann solver (wgpu: DX12 / Vulkan / Metal) with a CPU fallback. Velocity and pressure only.
- Photoshop-style dark UI with slice views, port report and VTK export.

![UI](docs/screenshot.png)

See `docs/BUILD.md` to build and `docs/PLAN.md` for the design and status.

Accuracy: at coarse cell sizes the solver adds artificial viscosity to stay stable and says so. Flow patterns
are usable; absolute pressure drops are only trustworthy when the report shows no viscosity warning.
