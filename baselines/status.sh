#!/bin/bash
# Quick status overview for the server training queue.
cd /root/mds || exit 1

echo "===== GPU ====="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/  /'

echo
echo "===== Automation status ====="
if [ -f automation_status.json ]; then
  cat automation_status.json
else
  echo "(no automation_status.json yet)"
fi

echo
echo "===== Davis training ====="
if pgrep -f "train_re\.py" >/dev/null; then
  latest=$(ls -dt results/CombinedDTALSTMDrop_davis_* 2>/dev/null | head -1)
  echo "Davis training is RUNNING. Latest run: $latest"
  tail -n 3 "$latest/history.csv" 2>/dev/null | cut -d, -f1,3,4
else
  echo "Davis training is not running."
fi

for ds in kiba bindingdb; do
  echo
  echo "===== $ds ====="
  log=$(ls -t automation_logs/${ds}_*.log 2>/dev/null | head -1)
  if [ -n "$log" ]; then
    echo "Log: $log"
    tail -n 3 "$log"
  else
    echo "(no log yet)"
  fi
  latest=$(ls -dt results/CombinedDTALSTMDrop_${ds}_* 2>/dev/null | head -1)
  if [ -n "$latest" ]; then
    echo "Latest run dir: $latest"
    tail -n 3 "$latest/history.csv" 2>/dev/null | cut -d, -f1,3,4
  fi
done
