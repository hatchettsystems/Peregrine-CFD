# Building Peregrine CFD

## Windows (RTX 3070 target)

**Easiest:** download `peregrine-windows-x64.zip` from the latest `windows` run under the repository's
Actions tab (artifact `peregrine-windows-x64`), unzip, and run `peregrine-app.exe`.

**Build it yourself in one step:** from PowerShell with Rust (MSVC toolchain) and Git installed,
```
.\scripts\build-windows.ps1
```
installs OpenCASCADE via vcpkg, builds, and writes `dist\peregrine-windows-x64.zip`. The manual steps:

1. Install Rust (rustup, stable) and the "Desktop development with C++" build tools (MSVC).
2. Install OpenCASCADE with vcpkg (needed for STEP import):
   ```
   git clone https://github.com/microsoft/vcpkg C:\vcpkg
   C:\vcpkg\bootstrap-vcpkg.bat
   C:\vcpkg\vcpkg install opencascade:x64-windows
   ```
3. Build and run:
   ```
   set OCCT_ROOT=C:\vcpkg\installed\x64-windows
   set PATH=%PATH%;C:\vcpkg\installed\x64-windows\bin
   cargo run --release -p peregrine-app
   ```
   Without `OCCT_ROOT` the build still succeeds; STEP import reports a clear error and STL still works.
4. The solver uses wgpu (DX12 or Vulkan). The adapter is picked automatically; choose GPU/CPU in the options bar.

## Linux

```
sudo apt install libocct-data-exchange-dev libocct-modeling-algorithms-dev libocct-ocaf-dev \
     libxkbcommon-x11-0 libgl1-mesa-dri
cargo run --release -p peregrine-app -- --demo
```

## Command line

```
cargo run --release -p peregrine-cli -- demo out/demo
cargo run --release -p peregrine-cli -- ports out/demo/project.toml
cargo run --release -p peregrine-cli -- set-port out/demo/project.toml 0 inlet velocity=8
cargo run --release -p peregrine-cli -- set-port out/demo/project.toml 1 outlet outlet=0
cargo run --release -p peregrine-cli -- run out/demo/project.toml --steps 3000
```

## GPU memory

The solver stores 19 f32 populations per cell in one buffer plus 21 bytes of fields: about 100 bytes/cell.
A single storage buffer is limited by the adapter (often 2-4 GB), which caps a run at roughly 20-40 M cells.
The status bar shows the estimate for the current cell size.
