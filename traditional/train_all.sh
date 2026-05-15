#!/bin/bash
# Run from the traditional/ directory:
#
#   bash train_all.sh            - start all models from scratch, stop on error
#   bash train_all.sh --continue - pick up where you left off:
#                                    * skips already-finished models
#                                    * auto-resumes interrupted models
#                                    * skips failed models and moves on
#
# You can Ctrl+C or close the terminal at any time.
# Just run  bash train_all.sh --continue  to carry on.

CONTINUE=0
if [[ "$1" == "--continue" ]]; then
    CONTINUE=1
fi

# ── Helpers ──────────────────────────────────────────────────────────────────

# Read a value from a YAML config using Python (already a dependency).
yaml_get() {
    python -c "
import yaml, sys
keys = sys.argv[2].split('.')
c = yaml.safe_load(open(sys.argv[1]))
for k in keys:
    c = c[k]
print(c)
" "$1" "$2"
}

# Return 'yes' if experiment.json exists and finished_at is not null.
is_finished() {
    local json="$1/experiment.json"
    [ -f "$json" ] || { echo "no"; return; }
    python -c "
import json, sys
d = json.load(open(sys.argv[1]))
print('yes' if d.get('finished_at') else 'no')
" "$json"
}

# ── Configs to train (comment out any you want to skip) ──────────────────────

CONFIGS=(
    # Custom CNN (CBC)
    # "CBC_config/config_CBC_10k.yml"
    # "CBC_config/config_CBC_50k.yml"
    # "CBC_config/config_CBC_100k.yml"

    # Custom CNN + Skeletonization (CBC-ZS)
    # "CBC_config/config_CBC_ZS_10k.yml"
    # "CBC_config/config_CBC_ZS_50k.yml"
    # "CBC_config/config_CBC_ZS_100k.yml"

    # ResNet (RBC)
    "CBC_config/config_RBC_10k.yml"
    "CBC_config/config_RBC_50k.yml"
    "CBC_config/config_RBC_100k.yml"

    # ResNet + Skeletonization (RBC-ZS)
    "CBC_config/config_RBC_ZS_10k.yml"
    "CBC_config/config_RBC_ZS_50k.yml"
    "CBC_config/config_RBC_ZS_100k.yml"

    # # DenseNet BiLSTM (DBC)
    # "CBC_config/config_DBC_10k.yml"
    # "CBC_config/config_DBC_50k.yml"
    # "CBC_config/config_DBC_100k.yml"

    # # DenseNet BiLSTM + BiGRU (original densenet)
    # "CBC_config/config_densenet_10k.yml"
    # "CBC_config/config_densenet_50k.yml"
    # "CBC_config/config_densenet_100k.yml"
)

# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR="logs/train_all"
mkdir -p "$LOG_DIR"

TOTAL=${#CONFIGS[@]}
PASSED=0
SKIPPED=0
RESUMED=0
FAILED=0
FAILED_NAMES=()

echo "=============================="
echo " Sequential training"
echo " Total configs    : $TOTAL"
echo " Log dir          : $LOG_DIR"
echo " Mode             : $( [[ $CONTINUE -eq 1 ]] && echo '--continue (resume + skip errors)' || echo 'normal (stop on error)' )"
echo "=============================="

START_ALL=$(date +%s)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME=$(basename "$CONFIG" .yml)
    LOG_FILE="$LOG_DIR/${NAME}.log"
    NUM=$((i + 1))

    # Read checkpoint_dir from the config
    CKPT_DIR=$(yaml_get "$CONFIG" "checkpoint.checkpoint_dir")
    LAST_CKPT="$CKPT_DIR/last_model.pth"

    # ── Skip already-finished models (both modes) ────────────────────────────
    if [[ "$(is_finished "$CKPT_DIR")" == "yes" ]]; then
        SKIPPED=$((SKIPPED + 1))
        echo ""
        echo "[$NUM/$TOTAL] SKIP (already finished): $NAME"
        continue
    fi

    # ── Detect interrupted run and auto-resume (both modes) ──────────────────
    RESUME_FLAG=""
    if [ -f "$LAST_CKPT" ]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        RESUMED=$((RESUMED + 1))
        echo ""
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') RESUMING: $NAME"
        echo "  Checkpoint : $LAST_CKPT"
    else
        echo ""
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') Starting: $NAME"
    fi

    echo "  Config : $CONFIG"
    echo "  Log    : $LOG_FILE"
    echo "------------------------------"

    START=$(date +%s)
    python tools/train.py --config "$CONFIG" $RESUME_FLAG 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    ELAPSED=$(( $(date +%s) - START ))
    ELAPSED_FMT=$(printf '%02dh %02dm %02ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

    if [ $EXIT_CODE -eq 0 ]; then
        PASSED=$((PASSED + 1))
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') DONE: $NAME  ($ELAPSED_FMT)"
    else
        FAILED=$((FAILED + 1))
        FAILED_NAMES+=("$NAME")
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') FAILED: $NAME (exit $EXIT_CODE, $ELAPSED_FMT)"
        echo "  Check log: $LOG_FILE"

        if [ $CONTINUE -eq 0 ]; then
            echo ""
            echo "Stopping. Re-run with  bash train_all.sh --continue  to skip and carry on."
            break
        fi
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
TOTAL_FMT=$(printf '%02dh %02dm %02ds' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60)))

echo ""
echo "=============================="
echo " Summary"
echo " Done    : $PASSED"
echo " Resumed : $RESUMED  (auto-resumed from checkpoint)"
echo " Skipped : $SKIPPED  (already finished)"
echo " Failed  : $FAILED"
echo " Total   : $TOTAL_FMT"
if [ ${#FAILED_NAMES[@]} -gt 0 ]; then
    echo " Failed configs:"
    for NAME in "${FAILED_NAMES[@]}"; do
        echo "   - $NAME"
    done
fi
echo "=============================="
