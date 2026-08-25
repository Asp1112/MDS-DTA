#!/bin/bash
# Automated formal training of CombinedDTALSTMDrop on KIBA and BindingDB with
# the exact Davis configuration (fixed 4/1/1 six-part holdout, seed 42,
# batch 256, lr 1e-4, early stopping 150, top-3 averaging, etc.).
#
# Behaviour:
#   1. Prepares kiba_sixfold_all.pt / bindingdb_sixfold_all.pt if missing.
#   2. Waits for the running Davis training (train_re.py) to finish so the GPU
#      is never shared between two formal runs (avoids OOM / slowdowns).
#   3. Trains KIBA, then BindingDB, sequentially in the background with logs.
#
# Usage: bash run_automation.sh [--start-now]
#   --start-now   skip waiting for Davis and start KIBA immediately

set -u

cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

PY="${MDS_PYTHON:-python}"
MODEL="${MODEL:-combined_dta_lstmdrop}"
LOG_DIR="$ROOT/automation_logs"
STATUS_FILE="$ROOT/automation_status.json"
LOCK_DIR="$ROOT/.automation.lock"

mkdir -p "$LOG_DIR"

if [ -d "$LOCK_DIR" ]; then
  echo "Another automation run is already active ($LOCK_DIR). Exiting."
  exit 1
fi
mkdir "$LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

START_NOW=0
for arg in "$@"; do
  case "$arg" in
    --start-now) START_NOW=1 ;;
  esac
done

write_status() {
  "$PY" - "$1" "$2" "$3" <<'PY'
import json, os, sys, time
phase, detail, pid = sys.argv[1], sys.argv[2], sys.argv[3]
status = {
    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "phase": phase,
    "detail": detail,
    "pid": pid,
}
with open(os.environ["AUTOMATION_STATUS"], "w") as fh:
    json.dump(status, fh, indent=2)
print(phase, "-", detail)
PY
}

dataset_done() {
  local ds="$1"
  compgen -G "results/CombinedDTALSTMDrop_${ds}_*/test_metrics.json" >/dev/null && return 0
  return 1
}

wait_for_davis() {
  if [ "$START_NOW" -eq 1 ]; then
    echo "Skipping the Davis wait (--start-now)."
    return
  fi
  davis_running() {
    pgrep -f "train_re\.py" >/dev/null && return 0
    pgrep -f "train\.py --dataset davis" >/dev/null && return 0
    pgrep -f "python train\.py$" >/dev/null && return 0
    return 1
  }
  if ! davis_running; then
    echo "Davis training is not running; proceeding."
    return
  fi
  echo "Davis training is running; waiting for it to finish ..."
  write_status "waiting_for_davis" "Davis training still running" ""
  while davis_running; do
    sleep 60
  done
  echo "Davis training finished."
}

run_dataset() {
  local ds="$1"
  if dataset_done "$ds"; then
    echo "$ds: a completed run already exists in results/; skipping."
    write_status "skipped" "$ds already has a completed run" ""
    return
  fi
  local log="$LOG_DIR/${ds}_$(date +%Y%m%d-%H%M%S).log"
  echo "$ds: starting training (model=$MODEL), log: $log"
  write_status "training_$ds" "model=$MODEL, log: $log" ""
  nohup "$PY" train.py --dataset "$ds" --model "$MODEL" >"$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG_DIR/${ds}.pid"
  wait "$pid"
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "$ds: FAILED with exit code $rc (see $log)"
    write_status "failed_$ds" "exit code $rc, log: $log" "$pid"
    exit $rc
  fi
  echo "$ds: finished successfully (see $log)"
  write_status "finished_$ds" "log: $log" "$pid"
}

echo "=== CombinedDTA-family automation (model=$MODEL; KIBA + BindingDB) ==="
echo "Start: $(date)"

# 1. Data preparation
# Big artifacts live on /root/autodl-tmp (large data disk); only symlinks
# are placed in data/processed so the training script can load them normally.
ln -sf /root/autodl-tmp/mds_data/bindingdb_train.pt data/processed/bindingdb_train.pt
ln -sf /root/autodl-tmp/mds_data/bindingdb_test.pt data/processed/bindingdb_test.pt
for ds in kiba bindingdb; do
  echo "Checking ${ds}_sixfold_all.pt ..."
  write_status "preparing_data_$ds" "" ""
  "$PY" prepare_sixfold_data.py --datasets "$ds" \
    --output-dir /root/autodl-tmp/mds_data || {
    echo "Data preparation failed for $ds."
    write_status "data_failed_$ds" "" ""
    exit 1
  }
done

# 2. Wait for Davis to free the GPU
wait_for_davis

# 3. Train KIBA, then BindingDB (sequentially)
run_dataset kiba
run_dataset bindingdb

write_status "done" "all datasets completed" ""
echo "All done: $(date)"
