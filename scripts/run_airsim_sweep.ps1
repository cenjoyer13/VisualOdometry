# Run every AirSim YAML config under configs/airsim/ through PathPlayer
# against a recorded flight-path CSV, launching and shutting down the
# Unreal environment between runs so each config sees the same initial scene.
#
# Outputs (under results/<ResultsSubdir>/):
#   logs/<config_basename>.csv     <- renamed playback_log.csv per run
#   run_<config_basename>.log      <- PathPlayer stdout+stderr
#
# Then run scripts/aggregate_results.py against the same subdir to produce
# the plots + summary table.
param(
    # Full path to the recorded flight CSV that PathPlayer should replay.
    [string] $FlightPath = 'E:\Workspace\DNU\Diploma2026\VisualOdometry\build\Release\flight_path.csv',
    # Folder name under results/ that holds the logs/, plots/, and summary.
    [string] $ResultsSubdir = 'airsim_eval'
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
$ProjectRoot   = 'E:\Workspace\DNU\Diploma2026\VisualOdometry'
$ReleaseDir    = Join-Path $ProjectRoot 'build\Release'
$ConfigsDir    = Join-Path $ProjectRoot 'configs\airsim'
$EnvShortcut   = Join-Path $ReleaseDir 'Enviroment.exe.lnk'
$PathPlayerExe = Join-Path $ReleaseDir 'PathPlayer.exe'
$ResultsDir    = Join-Path $ProjectRoot ("results\{0}" -f $ResultsSubdir)
$LogsDir       = Join-Path $ResultsDir  'logs'

# Pre-flight checks
foreach ($p in @($FlightPath, $EnvShortcut, $PathPlayerExe, $ConfigsDir)) {
    if (-not (Test-Path $p)) { throw "Missing prerequisite: $p" }
}
New-Item -ItemType Directory -Force $LogsDir | Out-Null

# Resolve the shortcut -> real exe path; we'll launch the .exe directly so
# we can track and kill the process reliably (a .lnk launched via the shell
# detaches from PowerShell's process handle).
$wsh = New-Object -ComObject WScript.Shell
$lnk = $wsh.CreateShortcut($EnvShortcut)
$EnvExe     = $lnk.TargetPath
$EnvArgs    = $lnk.Arguments
$EnvWorkDir = $lnk.WorkingDirectory
if (-not (Test-Path $EnvExe)) { throw "Environment exe not found: $EnvExe" }
$EnvProcName = [System.IO.Path]::GetFileNameWithoutExtension($EnvExe)

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------
$AirsimRpcPort   = 41451
$EnvReadyTimeout = [TimeSpan]::FromSeconds(90)   # max wait for RPC port
$EnvReadyPoll    = [TimeSpan]::FromSeconds(2)
$PathPlayerCap   = [TimeSpan]::FromMinutes(5)    # hard kill after this
$EnvShutdownWait = [TimeSpan]::FromSeconds(5)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Wait-EnvReady {
    param([int] $Port, [TimeSpan] $Timeout, [TimeSpan] $Poll)
    $deadline = (Get-Date).Add($Timeout)
    while ((Get-Date) -lt $deadline) {
        try {
            $c = New-Object System.Net.Sockets.TcpClient
            $iar = $c.BeginConnect('127.0.0.1', $Port, $null, $null)
            $ok = $iar.AsyncWaitHandle.WaitOne(1000)
            if ($ok -and $c.Connected) { $c.EndConnect($iar); $c.Close(); return $true }
            $c.Close()
        } catch { }
        Start-Sleep -Seconds $Poll.TotalSeconds
    }
    return $false
}

function Stop-Env {
    # Kill every process named after the env exe, plus the CrashReportClient
    # Unreal sometimes spawns, then wait for ports to free.
    Get-Process -Name $EnvProcName -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_.Kill() } catch { }
    }
    Get-Process -Name 'CrashReportClient' -ErrorAction SilentlyContinue | ForEach-Object {
        try { $_.Kill() } catch { }
    }
    Start-Sleep -Seconds $EnvShutdownWait.TotalSeconds
}

function Start-Env {
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName         = $EnvExe
    $startInfo.Arguments        = $EnvArgs
    $startInfo.WorkingDirectory = $EnvWorkDir
    $startInfo.UseShellExecute  = $true   # let Unreal own its window
    [System.Diagnostics.Process]::Start($startInfo) | Out-Null
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
$configs = Get-ChildItem $ConfigsDir -Filter '*.yaml' | Sort-Object Name
Write-Host ("Found {0} configs in {1}" -f $configs.Count, $ConfigsDir) -ForegroundColor Cyan

# Make sure no env is already running.
Stop-Env

$i = 0
foreach ($cfg in $configs) {
    $i++
    $base   = [System.IO.Path]::GetFileNameWithoutExtension($cfg.Name)
    $stdLog = Join-Path $ResultsDir ("run_{0}.log" -f $base)
    $outCsv = Join-Path $LogsDir   ("{0}.csv"     -f $base)

    Write-Host ""
    Write-Host ("[{0,2}/{1}] {2}" -f $i, $configs.Count, $base) -ForegroundColor Yellow

    if (Test-Path $outCsv) {
        Write-Host "      skipping (log already exists)" -ForegroundColor DarkGray
        continue
    }

    # 1. Boot the environment.
    Write-Host "      launching environment..."
    Start-Env
    if (-not (Wait-EnvReady -Port $AirsimRpcPort -Timeout $EnvReadyTimeout -Poll $EnvReadyPoll)) {
        Write-Host "      ERROR: RPC port $AirsimRpcPort never opened. Skipping." -ForegroundColor Red
        Stop-Env
        continue
    }
    Write-Host "      RPC ready, starting PathPlayer..."

    # 2. Run PathPlayer with project root as CWD so models/ resolves.
    #    PathPlayer expects: <trajectory.csv> <odometry_config.yaml>
    $stagePath = Join-Path $ProjectRoot 'playback_log.csv'
    if (Test-Path $stagePath) { Remove-Item $stagePath -Force }

    $proc = Start-Process -FilePath $PathPlayerExe `
        -ArgumentList @("`"$FlightPath`"", "`"$($cfg.FullName)`"") `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdLog `
        -RedirectStandardError  ($stdLog + '.err') `
        -PassThru -NoNewWindow

    $exitedInTime = $proc.WaitForExit([int]$PathPlayerCap.TotalMilliseconds)
    if (-not $exitedInTime) {
        Write-Host "      timeout: killing PathPlayer after $($PathPlayerCap.TotalMinutes) min" -ForegroundColor Red
        try { $proc.Kill() } catch { }
        $proc.WaitForExit()
    }

    # Merge stderr into stdout log for a single run_*.log file.
    if (Test-Path ($stdLog + '.err')) {
        Get-Content ($stdLog + '.err') | Add-Content $stdLog
        Remove-Item ($stdLog + '.err') -Force
    }

    # 3. Capture log
    if (Test-Path $stagePath) {
        Move-Item $stagePath $outCsv -Force
        $rows = (Get-Content $outCsv | Measure-Object -Line).Lines - 1
        Write-Host ("      done. {0} frames -> {1}" -f $rows, (Split-Path -Leaf $outCsv)) -ForegroundColor Green
    } else {
        Write-Host "      ERROR: PathPlayer produced no playback_log.csv" -ForegroundColor Red
    }

    # 4. Shut down env for clean scene reset.
    Write-Host "      shutting environment down..."
    Stop-Env
}

Write-Host ""
Write-Host "Sweep complete. Logs in $LogsDir" -ForegroundColor Cyan
