# Point Windows :2222 at the current WSL sshd (:22) so the LAN can ssh in.
# The WSL IP changes on every wsl --shutdown / reboot, so this must run again
# each time. Two ways it does:
#   - double-click wsl-ssh-tunnel.cmd next to this file (one UAC click)
#   - the scheduled task "WSL ssh tunnel", which this script registers on its
#     first elevated run: at every logon of this user, highest privileges, 30 s
#     delay so the network is up, then this file with -NoPause.
# Either way it also STARTS WSL if it is down, so the scheduled task doubles
# as "start WSL at logon".
param([switch]$NoPause)

if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  Start-Process powershell -Verb RunAs -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$PSCommandPath`""
  exit
}

$taskName = "WSL ssh tunnel"
if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue)) {
  $action  = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$PSCommandPath`" -NoPause"
  $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
  $trigger.Delay = "PT30S"
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
  $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings | Out-Null
  Write-Host "registered scheduled task '$taskName' (at logon, 30 s delay, highest privileges)"
}

# Start WSL if it is not running (this is what boots it at logon), then wait
# for the distro to have an address. systemd inside it brings sshd up and keeps
# the VM alive afterwards.
wsl.exe --exec /bin/true 2>$null
$wslIp = ""
foreach ($i in 1..20) {
  $wslIp = ((wsl.exe hostname -I 2>$null) -join " ").Trim().Split(' ')[0]
  if ($wslIp) { break }
  Start-Sleep 3
}
if (-not $wslIp) { Write-Host "WSL did not come up with an IP in 60 s (wsl --status?)."; if (-not $NoPause) { Read-Host "Enter to close" }; exit 1 }

netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=2222 2>$null | Out-Null
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=2222 connectaddress=$wslIp connectport=22 | Out-Null
if (-not (Get-NetFirewallRule -DisplayName "WSL ssh 2222" -ErrorAction SilentlyContinue)) {
  New-NetFirewallRule -DisplayName "WSL ssh 2222" -Direction Inbound -Protocol TCP -LocalPort 2222 -Action Allow | Out-Null
}

$hostIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like '10.0.1.*' }).IPAddress
Write-Host ""
Write-Host "tunnel: ${hostIp}:2222 -> WSL ${wslIp}:22"
netsh interface portproxy show v4tov4
$ok = Test-NetConnection -ComputerName 127.0.0.1 -Port 2222 -InformationLevel Quiet -WarningAction SilentlyContinue
Write-Host ("sshd reachable through the tunnel: " + $(if ($ok) { "YES" } else { "NO -- is sshd running in WSL? (sudo systemctl start ssh)" }))
if (-not $NoPause) { Read-Host "Enter to close" }
