# One-shot Windows build. Run from a "x64 Native Tools Command Prompt" / PowerShell with Rust and Git installed.
# Installs OpenCASCADE through vcpkg if OCCT_ROOT is not set, builds, and writes dist\peregrine-windows-x64.zip
$ErrorActionPreference = "Stop"
if (-not $env:OCCT_ROOT) {
    $vcpkg = if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "C:\vcpkg" }
    if (-not (Test-Path "$vcpkg\vcpkg.exe")) {
        git clone https://github.com/microsoft/vcpkg $vcpkg
        & "$vcpkg\bootstrap-vcpkg.bat" -disableMetrics
    }
    & "$vcpkg\vcpkg.exe" install opencascade:x64-windows
    $env:OCCT_ROOT = "$vcpkg\installed\x64-windows"
}
$env:PATH = "$env:OCCT_ROOT\bin;$env:PATH"
cargo build --release -p peregrine-app -p peregrine-cli
& "$PSScriptRoot\package-windows.ps1" -OcctRoot $env:OCCT_ROOT
