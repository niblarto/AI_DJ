# Installs the AI DJ servers as native Windows services via NSSM:
#   AIDJService    - workout-mix Flask service (port 8765) for the Running app
#   AIDJWebService - web GUI landing page (port 8766, http://localhost:8766)
# Run this in an elevated PowerShell (Run as administrator). Re-running is idempotent.

$ErrorActionPreference = "Stop"

$pythonExe = "E:\Code\AI_BPM\.venv\Scripts\python.exe"
$workDir   = "E:\Code\AI_DJ"

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

function Install-AiDjService {
    param(
        [string]$Name,
        [string]$Arguments,
        [string]$LogPrefix,
        [string]$Description
    )

    # Remove any previous install of this service so re-running is idempotent.
    if (Get-Service -Name $Name -ErrorAction SilentlyContinue) {
        nssm stop $Name confirm | Out-Null
        nssm remove $Name confirm | Out-Null
    }

    nssm install $Name $pythonExe $Arguments
    nssm set $Name AppDirectory $workDir
    nssm set $Name AppStdout "$workDir\$LogPrefix.out.log"
    nssm set $Name AppStderr "$workDir\$LogPrefix.err.log"
    nssm set $Name AppRotateFiles 1
    nssm set $Name AppRotateBytes 1048576
    nssm set $Name Start SERVICE_AUTO_START
    nssm set $Name AppExit Default Restart
    nssm set $Name AppRestartDelay 5000
    nssm set $Name Description $Description
    # The service runs as LocalSystem, whose profile is systemprofile - point
    # profile lookups (Mixxx DB, default output dir) at the installing user.
    nssm set $Name AppEnvironmentExtra "USERPROFILE=$env:USERPROFILE" "LOCALAPPDATA=$env:LOCALAPPDATA"

    Start-Service $Name
}

Install-AiDjService -Name "AIDJService" `
    -Arguments "-u -m ai_dj.server --port 8765" `
    -LogPrefix "service" `
    -Description "AI DJ workout-mix Flask service (port 8765) for the Running app."

Install-AiDjService -Name "AIDJWebService" `
    -Arguments "-u -m ai_dj.webapp --port 8766" `
    -LogPrefix "service-web" `
    -Description "AI DJ web GUI (http://localhost:8766)."

# Firewall exceptions so other devices on the LAN can reach the services.
foreach ($rule in @(@{Name = "AI DJ service (8765)"; Port = 8765}, @{Name = "AI DJ web GUI (8766)"; Port = 8766})) {
    if (-not (Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $rule.Port | Out-Null
        Write-Host "Added firewall rule '$($rule.Name)'."
    }
}

Start-Sleep -Seconds 2
Get-Service AIDJService, AIDJWebService | Format-List Name, Status, StartType
Write-Host "`nWeb GUI: http://localhost:8766"
Write-Host "Logs: $workDir\service.out.log / service.err.log and service-web.out.log / service-web.err.log"
