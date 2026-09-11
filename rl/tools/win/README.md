# ssh into a WSL training box from the LAN

WSL2 sits behind NAT on its Windows host and gets a new address on every
`wsl --shutdown` / reboot. These two files keep `host:2222` pointed at it.

One-time, inside WSL:

    sudo apt install -y openssh-server
    # /etc/ssh/sshd_config: PubkeyAuthentication yes, PasswordAuthentication no
    sudo systemctl enable --now ssh
    # the mac's public key into ~/.ssh/authorized_keys (dir 700, file 600)

One-time, on Windows: copy `wsl-ssh-portproxy.ps1` and `wsl-ssh-tunnel.cmd`
side by side (e.g. `C:\Users\<you>\`) and double-click the `.cmd`. One UAC
click. It adds the portproxy and the firewall rule, and registers the
scheduled task "WSL ssh tunnel": at every logon, highest privileges, it starts
WSL, waits for its IP and re-points the proxy. So the box is reachable a
minute after it boots, without anyone logging into WSL.

From the mac: `ssh -p 2222 <user>@<host-lan-ip>`.  The `.ps1` prints the pair.
