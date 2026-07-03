# Installs the AI DJ workout-mix server as a native Windows service via NSSM.
# Run this in an elevated PowerShell (Run as administrator).

$ErrorActionPreference = "Stop"

$pythonExe  = "E:\Code\AI_BPM\.venv\Scripts\python.exe"
$workDir    = "E:\Code\AI_DJ"
$logOut     = "E:\Code\AI_DJ\service.out.log"
$logErr     = "E:\Code\AI_DJ\service.err.log"
$serviceName = "AIDJService"

# Remove the old scheduled-task attempt, if present.
if (Get-ScheduledTask -TaskName "AI DJ Service" -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName "AI DJ Service" -Confirm:$false
    Write-Host "Removed old 'AI DJ Service' scheduled task."
}

# Install NSSM if it's not already available.
$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) {
    winget install --id NSSM.NSSM --accept-source-agreements --accept-package-agreements
    # winget installs to a WinGet Links dir that's on PATH for new sessions;
    # refresh PATH in this session so `nssm` resolves immediately.
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
    $nssm = Get-Command nssm -ErrorAction SilentlyContinue
}
if (-not $nssm) { throw "nssm not found on PATH after install - open a new elevated terminal and re-run this script." }

# Remove any previous install of this service so re-running is idempotent.
if (Get-Service -Name $serviceName -ErrorAction SilentlyContinue) {
    nssm stop $serviceName confirm | Out-Null
    nssm remove $serviceName confirm | Out-Null
}

nssm install $serviceName $pythonExe "-u -m ai_dj.server --port 8765"
nssm set $serviceName AppDirectory $workDir
nssm set $serviceName AppStdout $logOut
nssm set $serviceName AppStderr $logErr
nssm set $serviceName AppRotateFiles 1
nssm set $serviceName AppRotateBytes 1048576
nssm set $serviceName Start SERVICE_AUTO_START
nssm set $serviceName AppExit Default Restart
nssm set $serviceName AppRestartDelay 5000
nssm set $serviceName Description "AI DJ workout-mix Flask service (port 8765) for the Running app."

Start-Service $serviceName
Start-Sleep -Seconds 2
Get-Service $serviceName | Format-List Name, Status, StartType
Write-Host "`nLogs: $logOut / $logErr"
