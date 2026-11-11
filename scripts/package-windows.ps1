# Collects the release exe files plus the OpenCASCADE DLLs into dist\peregrine-windows-x64.zip
param([string]$OcctRoot = $env:OCCT_ROOT)
$ErrorActionPreference = "Stop"
$dist = "dist\peregrine-windows-x64"
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item target\release\peregrine-app.exe, target\release\peregrine.exe $dist
if ($OcctRoot) { Copy-Item "$OcctRoot\bin\*.dll" $dist }
Copy-Item README.md, docs\BUILD.md $dist
Compress-Archive -Path "$dist\*" -DestinationPath "dist\peregrine-windows-x64.zip" -Force
Write-Host "wrote dist\peregrine-windows-x64.zip"
