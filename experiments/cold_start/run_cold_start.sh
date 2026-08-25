#!/bin/bash
# Run the cold-start experiments (entity-level six-fold) for the selected
# model and datasets.  Defaults to the MDS family model; pass one of
# deepdta / graphdta / deepdtagen to run the corresponding baseline.
#
# Usage:
#   bash run_cold_start.sh --model deepdta --datasets davis kiba
#   bash run_cold_start.sh --model graphdta_gcn --datasets davis kiba bindingdb
#   bash run_cold_start.sh --model deepdtagen --datasets davis kiba
#   bash run_cold_start.sh --model MDS_dta --datasets davis --dry

set -u

cd "$(dirname "$0")" || exit 1
PY="${PY:-/root/mds/.venv_dta/bin/python}"
MODEL="MDS_dta"
DATASETS=(davis)
SETTINGS=(cold_drug cold_target cold_both)
FOLDS=(0 1 2 3 4 5)
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) shift; DATASETS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
    --folds) shift; FOLDS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do FOLDS+=("$1"); shift; done ;;
    --settings) shift; SETTINGS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SETTINGS+=("$1"); shift; done ;;
    --dry) DRY=1; shift ;;
    --*) EXTRA+=("$1"); if [ $# -gt 1 ] && [[ "$2" != --* ]]; then EXTRA+=("$2"); shift; fi; shift ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
done

mkdir -p logs
[ "$DRY" -eq 1 ] && EXTRA+=(--dry)

echo "=== cold-start (model=$MODEL, datasets=${DATASETS[*]}, settings=${SETTINGS[*]}, folds=${FOLDS[*]}, dry=$DRY) ==="
for ds in "${DATASETS[@]}"; do
  for setting in "${SETTINGS[@]}"; do
    for fold in "${FOLDS[@]}"; do
      log="logs/${ds}_${setting}_fold${fold}_$(date +%Y%m%d-%H%M%S).log"
      echo ">>> $ds $setting fold $fold (log: $log)"
      "$PY" train_cold.py --dataset "$ds" --setting "$setting" --fold "$fold" \
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
echo "=== cold-start done (model=$MODEL) ==="

