#!/bin/bash
echo "== docker =="
docker info --format '  server={{.ServerVersion}} containers={{.Containers}} running={{.ContainersRunning}}' 2>&1 | head -3
echo "  -- containers --"
docker ps -a --format '  {{.Names}}|{{.Image}}|{{.Status}}' 2>&1 | head -20
echo "== listening ports =="
ss -lntp 2>/dev/null | tail -n +2 | awk '{print "  "$4"  "$6}' | head -25
echo "== cuda =="
ls /usr/local | grep -i cuda || echo "  no /usr/local/cuda"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== systemd services (non-standard) =="
systemctl list-units --type=service --state=running --no-legend 2>/dev/null | awk '{print $1}' | grep -vE '^(ssh|chrony|crond|dbus|NetworkManager|firewalld|polkit|rsyslog|sshd|systemd|tuned|irqbalance|auditd|gssproxy|rpcbind|nfs|postfix|atd|kdump)' | head -20
echo "== reverse proxy =="
(nginx -v 2>&1 || echo "  no nginx")
ls -1 /etc/nginx/conf.d /etc/nginx/sites-enabled 2>/dev/null | head -20
echo "== gpu processes =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | head -10
echo "== /opt detail =="
du -sh /opt/* 2>/dev/null
echo "== python env =="
python3 -c "import sys; print('  ', sys.version.split()[0])"
pip3 list 2>/dev/null | grep -iE 'torch|vllm|fastapi|uvicorn|transformers|paddle|ultralytics' | sed 's/^/  /'
echo "== free space on /data =="
mkdir -p /data/tmp 2>/dev/null && touch /data/tmp/.w && echo "  /data writable" && rm -f /data/tmp/.w
