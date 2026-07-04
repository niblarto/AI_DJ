# Installs Icecast 2 as a Windows service (IcecastService) for Mixxx live
# broadcasting. Run elevated. Expects the official NSIS installer; download:
#   https://downloads.xiph.org/releases/icecast/icecast_win32_2.4.4.exe
# Usage: .\install-icecast.ps1 -Installer path\to\icecast_win32_2.4.4.exe
param(
    [string]$Installer = "",
    [string]$InstallDir = "C:\Icecast",
    [string]$SourcePassword = "hackme",
    [string]$AdminPassword = "hackme-admin",
    [int]$Port = 8000
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $InstallDir "icecast.xml.dist")) -and -not (Get-ChildItem $InstallDir -Filter "icecast*.exe" -Recurse -ErrorAction SilentlyContinue)) {
    if (-not $Installer -or -not (Test-Path $Installer)) { throw "Pass -Installer with the Icecast setup exe." }
    Start-Process $Installer -ArgumentList "/S", "/D=$InstallDir" -Wait
}

$exe = (Get-ChildItem $InstallDir -Filter "icecast*.exe" -Recurse |
        Where-Object { $_.Name -notmatch "uninst" } | Select-Object -First 1).FullName
if (-not $exe) { throw "icecast.exe not found under $InstallDir" }
Write-Host "Icecast binary: $exe"

New-Item -ItemType Directory -Force (Join-Path $InstallDir "logs") | Out-Null

# Minimal LAN config: one source (Mixxx), listener limit, rotating logs.
$config = @"
<icecast>
    <location>Home</location>
    <admin>icemaster@localhost</admin>
    <limits>
        <clients>16</clients>
        <sources>2</sources>
        <queue-size>524288</queue-size>
        <client-timeout>30</client-timeout>
        <header-timeout>15</header-timeout>
        <source-timeout>10</source-timeout>
        <burst-on-connect>1</burst-on-connect>
        <burst-size>65535</burst-size>
    </limits>
    <authentication>
        <source-password>$SourcePassword</source-password>
        <relay-password>$SourcePassword</relay-password>
        <admin-user>admin</admin-user>
        <admin-password>$AdminPassword</admin-password>
    </authentication>
    <hostname>localhost</hostname>
    <listen-socket>
        <port>$Port</port>
    </listen-socket>
    <http-headers>
        <header name="Access-Control-Allow-Origin" value="*" />
    </http-headers>
    <fileserve>1</fileserve>
    <paths>
        <basedir>$InstallDir</basedir>
        <logdir>$InstallDir\logs</logdir>
        <webroot>$InstallDir\web</webroot>
        <adminroot>$InstallDir\admin</adminroot>
    </paths>
    <logging>
        <accesslog>access.log</accesslog>
        <errorlog>error.log</errorlog>
        <loglevel>3</loglevel>
        <logsize>10000</logsize>
        <logarchive>1</logarchive>
    </logging>
</icecast>
"@
$configPath = Join-Path $InstallDir "icecast-mixxx.xml"
$config | Out-File $configPath -Encoding utf8
Write-Host "Wrote $configPath"

$nssm = Get-Command nssm -ErrorAction SilentlyContinue
if (-not $nssm) { throw "nssm not found - run install-service.ps1 first (it installs NSSM)." }

if (Get-Service -Name "IcecastService" -ErrorAction SilentlyContinue) {
    nssm stop IcecastService confirm | Out-Null
    nssm remove IcecastService confirm | Out-Null
}
nssm install IcecastService $exe "-c `"$configPath`""
nssm set IcecastService AppDirectory $InstallDir
nssm set IcecastService AppStdout "$InstallDir\logs\service.out.log"
nssm set IcecastService AppStderr "$InstallDir\logs\service.err.log"
nssm set IcecastService AppRotateFiles 1
nssm set IcecastService AppRotateBytes 1048576
nssm set IcecastService Start SERVICE_AUTO_START
nssm set IcecastService AppExit Default Restart
nssm set IcecastService AppRestartDelay 5000
nssm set IcecastService Description "Icecast 2 streaming server (port $Port) - Mixxx live broadcast target."

if (-not (Get-NetFirewallRule -DisplayName "Icecast ($Port)" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName "Icecast ($Port)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port | Out-Null
    Write-Host "Added firewall rule 'Icecast ($Port)'."
}

Start-Service IcecastService
Start-Sleep -Seconds 2
Get-Service IcecastService | Format-List Name, Status, StartType
Write-Host "Status page: http://localhost:$Port/  (admin / your admin password)"
