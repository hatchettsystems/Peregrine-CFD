# Collects the release exe files plus the OpenCASCADE DLLs they need into dist\peregrine-windows-x64.zip
param([string]$OcctRoot = $env:OCCT_ROOT)
$ErrorActionPreference = "Stop"
$dist = "dist\peregrine-windows-x64"
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $dist | Out-Null
Copy-Item target\release\peregrine-app.exe, target\release\peregrine.exe $dist
if ($OcctRoot) {
    $bin = Join-Path $OcctRoot "bin"
    # find dumpbin (Visual Studio) to follow the imports; without it, fall back to copying every DLL
    $dumpbin = Get-ChildItem "C:\Program Files\Microsoft Visual Studio" -Recurse -Filter dumpbin.exe -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "Hostx64\\x64" } | Select-Object -First 1
    $copied = @{}
    function Add-Deps([string]$file) {
        $out = & $dumpbin.FullName /dependents $file 2>$null
        foreach ($line in $out) {
            $name = $line.Trim()
            if ($name -match "^[A-Za-z0-9_.+-]+\.dll$") {
                $src = Join-Path $bin $name
                if ((Test-Path $src) -and -not $copied.ContainsKey($name.ToLower())) {
                    $copied[$name.ToLower()] = $true
                    Copy-Item $src $dist
                    Add-Deps $src
                }
            }
        }
    }
    if ($dumpbin) {
        foreach ($exe in @("peregrine-app.exe", "peregrine.exe")) { Add-Deps (Join-Path $dist $exe) }
        Write-Host "copied $($copied.Count) DLLs the programs import"
        if ($copied.Count -lt 5) { Write-Host "too few DLLs found, copying all"; Copy-Item "$bin\*.dll" $dist }
    } else {
        Write-Host "dumpbin not found: copying every DLL"
        Copy-Item "$bin\*.dll" $dist
    }
}
Copy-Item README.md, docs\BUILD.md $dist
Compress-Archive -Path "$dist\*" -DestinationPath "dist\peregrine-windows-x64.zip" -Force
Write-Host "wrote dist\peregrine-windows-x64.zip"
Get-Item dist\peregrine-windows-x64.zip | ForEach-Object { Write-Host ("size {0:N1} MB" -f ($_.Length / 1MB)) }
