set -u

cd "$(dirname "$0")" || exit 1
PY="${PY:-/root/mds/.venv_dta/bin/python}"
MODEL="MDS_dta"
DATASETS=(davis)
SETTINGS=(cold_drug cold_target cold_both)
FOLDS=(0 1 2 3 4 5)
DRY=0
EXTRA=()

while [ $
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --datasets) shift; DATASETS=(); while [ $
    --folds) shift; FOLDS=(); while [ $
    --settings) shift; SETTINGS=(); while [ $
    --dry) DRY=1; shift ;;
    --*) EXTRA+=("$1"); if [ $
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
