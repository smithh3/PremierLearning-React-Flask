#!/bin/bash
set -euxo pipefail
dnf install -y iptables
echo 'net.ipv4.ip_forward = 1' > /etc/sysctl.d/10-ip-forward.conf
sysctl -p /etc/sysctl.d/10-ip-forward.conf

cat > /usr/local/sbin/enable-nat.sh <<'SCRIPT'
#!/bin/bash
set -e
sysctl -w net.ipv4.ip_forward=1
IFACE=$(ip -4 route show default | awk '{print $5; exit}')
iptables -t nat -C POSTROUTING -o "$IFACE" -j MASQUERADE 2>/dev/null \
  || iptables -t nat -A POSTROUTING -o "$IFACE" -j MASQUERADE
iptables -P FORWARD ACCEPT
SCRIPT
chmod +x /usr/local/sbin/enable-nat.sh

cat > /etc/systemd/system/enable-nat.service <<'UNIT'
[Unit]
Description=Enable NAT masquerade for Fargate
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/sbin/enable-nat.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now enable-nat.service
