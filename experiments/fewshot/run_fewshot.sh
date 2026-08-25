#!/bin/bash
# Run the few-shot experiments (six-fold, 50% / 25% / 10% training data).
#
# Usage:
#   bash run_fewshot.sh --datasets davis kiba bindingdb
#   bash run_fewshot.sh --datasets davis --dry

set -u

cd "$(dirname "$0")" || exit 1
PY="${PY:-/root/mds/.venv_dta/bin/python}"
MODEL="combined_dta"
DATASETS=(davis)
SETTINGS=(fs50 fs25 fs10)
FOLDS=(0 1 2 3 4 5)
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) shift; DATASETS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --folds) shift; FOLDS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do FOLDS+=("$1"); shift; done ;;
    --dry) DRY=1; shift ;;
    --*) EXTRA+=("$1"); if [ $# -gt 1 ] && [[ "$2" != --* ]]; then EXTRA+=("$2"); shift; fi; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p logs
[ "$DRY" -eq 1 ] && EXTRA+=(--dry)

echo "=== few-shot (model=$MODEL, datasets=${DATASETS[*]}, settings=${SETTINGS[*]}, folds=${FOLDS[*]}, dry=$DRY) ==="
for ds in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      log="logs/${ds}_${setting}_fold${fold}_$(date +%Y%m%d-%H%M%S).log"
      echo ">>> $ds $setting fold $fold (log: $log)"
      "$PY" train_fewshot.py --dataset "$ds" --setting "$setting" --fold "$fold" \
        --model "$MODEL" --skip-done "${EXTRA[@]}" >"$log" 2>&1
      rc=$?
      if [ $rc -ne 0 ]; then
        echo "[FAILED] $ds $setting fold $fold (exit $rc, see $log)"
        tail -20 "$log"
        exit $rc
      fi
      echo "[OK] $ds $setting fold $fold"
    done
  done
done
echo "=== few-shot done (model=$MODEL) ==="
