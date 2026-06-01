# Run a single AirSim YAML config through PathPlayer, with env launch/teardown.
# Used for diagnostics: gives a single isolated log file you can inspect for
# stderr from the pipeline.
#
# Usage: powershell -File run_one_config.ps1 <config-basename-without-yaml>
param([Parameter(Mandatory=$true)][string] $ConfigBase)

$ErrorActionPreference = 'Stop'

$ProjectRoot   = 'E:\Workspace\DNU\Diploma2026\VisualOdometry'
$ReleaseDir    = Join-Path $ProjectRoot 'build\Release'
$ConfigPath    = Join-Path $ProjectRoot ("configs\airsim\{0}.yaml" -f $ConfigBase)
$FlightPath    = Join-Path $ReleaseDir 'flight_path.csv'
$EnvShortcut   = Join-Path $ReleaseDir 'Enviroment.exe.lnk'
$PathPlayerExe = Join-Path $ReleaseDir 'PathPlayer.exe'
$ResultsDir    = Join-Path $ProjectRoot 'results\airsim_eval'
$LogsDir       = Join-Path $ResultsDir  'logs'

foreach ($p in @($ConfigPath, $FlightPath, $EnvShortcut, $PathPlayerExe)) {
    if (-not (Test-Path $p)) { throw "Missing prerequisite: $p" }
}
New-Item -ItemType Directory -Force $LogsDir | Out-Null

$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($EnvShortcut)
$EnvExe     = $lnk.TargetPath
$EnvArgs    = $lnk.Arguments
$EnvWorkDir = $lnk.WorkingDirectory
$EnvProcName = [System.IO.Path]::GetFileNameWithoutExtension($EnvExe)

function Wait-EnvReady {
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $iar = $c.BeginConnect('127.0.0.1', 41451, $null, $null)
            $ok = $iar.AsyncWaitHandle.WaitOne(1000)
            if ($ok -and $c.Connected) { $c.EndConnect($iar); $c.Close(); return $true }
            $c.Close()
        } catch { }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Stop-Env {
    Get-Process -Name $EnvProcName -ErrorAction SilentlyContinue | ForEach-Object { try { $_.Kill() } catch {} }
    Get-Process -Name 'CrashReportClient' -ErrorAction SilentlyContinue | ForEach-Object { try { $_.Kill() } catch {} }
    Start-Sleep -Seconds 5
}

function Start-Env {
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName         = $EnvExe
    $startInfo.Arguments        = $EnvArgs
    $startInfo.WorkingDirectory = $EnvWorkDir
    $startInfo.UseShellExecute  = $true
    [System.Diagnostics.Process]::Start($startInfo) | Out-Null
}

$stdLog = Join-Path $ResultsDir ("run_{0}.log" -f $ConfigBase)
$outCsv = Join-Path $LogsDir   ("{0}.csv"     -f $ConfigBase)

Write-Host "[+] $ConfigBase" -ForegroundColor Yellow
Stop-Env
Write-Host "    launching env..."
Start-Env
if (-not (Wait-EnvReady)) { Stop-Env; throw "RPC port 41451 never opened" }
Write-Host "    RPC ready, starting PathPlayer..."

$stagePath = Join-Path $ProjectRoot 'playback_log.csv'
if (Test-Path $stagePath) { Remove-Item $stagePath -Force }

$proc = Start-Process -FilePath $PathPlayerExe `
    -ArgumentList @("`"$FlightPath`"", "`"$ConfigPath`"") `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $stdLog `
    -RedirectStandardError  ($stdLog + '.err') `
    -PassThru -NoNewWindow

$exitedInTime = $proc.WaitForExit(5 * 60 * 1000)
if (-not $exitedInTime) {
    Write-Host "    TIMEOUT" -ForegroundColor Red
    try { $proc.Kill() } catch {}
    $proc.WaitForExit()
}

if (Test-Path ($stdLog + '.err')) {
    Get-Content ($stdLog + '.err') | Add-Content $stdLog
    Remove-Item ($stdLog + '.err') -Force
}

if (Test-Path $stagePath) {
    Move-Item $stagePath $outCsv -Force
    $rows = (Get-Content $outCsv | Measure-Object -Line).Lines - 1
    Write-Host ("    {0} frames -> {1}" -f $rows, (Split-Path -Leaf $outCsv)) -ForegroundColor Green
}

Stop-Env
Write-Host "[done]"
