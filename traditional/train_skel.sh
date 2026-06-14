#!/bin/bash
# =============================================================================
# CRNN Training Runner — SKELETON images
#
# Uses pre-computed skeleton-converted images (saved on disk) and its own config
# folder (CRNN_skel/). Unlike train.sh (plain grayscale), these configs set
# preprocessing.binarize=True, so the input is thresholded to black-and-white.
# The other differences from train.sh are the input dataset and output/log paths.
#
# Run from the traditional/ directory:
#
#   bash train_skel.sh             — run all configs, stop on first error
#   bash train_skel.sh --continue  — resume interrupted; skip finished & failed
#
# Ctrl+C any time, then re-run with --continue to pick up where you left off.
# =============================================================================

# ─── SET YOUR DATASET BASE PATH HERE ─────────────────────────────────────────
DATASET_BASE="/home/thareah/Desktop/text_img/skeleton"
# Dataset subdirectories are derived automatically:
#   ${DATASET_BASE}/khmer_10k/
#   ${DATASET_BASE}/khmer_50k/
#   ${DATASET_BASE}/khmer_100k/
# ─────────────────────────────────────────────────────────────────────────────

# ─── SET YOUR OUTPUT BASE PATH HERE ──────────────────────────────────────────
# All run outputs (checkpoints, logs, metrics) are written under here, replacing
# the default "outputs_crnn" prefix in each config. Per-run subdirs are kept:
#   ${OUTPUT_BASE}/vgg_bilstm_ctc_10k/checkpoints/
OUTPUT_BASE="/home/thareah/Desktop/server_config/reah_ocr/traditional/outputs_crnn_skel"
# ─────────────────────────────────────────────────────────────────────────────

# ── Parse flags ───────────────────────────────────────────────────────────────
CONTINUE=0
for arg in "$@"; do
    case "$arg" in
        --continue|--cont) CONTINUE=1 ;;
    esac
done

# ── Configs ───────────────────────────────────────────────────────────────────
CONFIGS=(
    # VGG
    CRNN_skel/config_CRNN_skel_vgg_bilstm_ctc_10k.yml
    CRNN_skel/config_CRNN_skel_vgg_bilstm_ctc_50k.yml
    CRNN_skel/config_CRNN_skel_vgg_bilstm_ctc_100k.yml
    CRNN_skel/config_CRNN_skel_vgg_bilstm_attention_10k.yml
    CRNN_skel/config_CRNN_skel_vgg_bilstm_attention_50k.yml
    CRNN_skel/config_CRNN_skel_vgg_bilstm_attention_100k.yml

    # ResNet
    CRNN_skel/config_CRNN_skel_resnet_bilstm_ctc_10k.yml
    CRNN_skel/config_CRNN_skel_resnet_bilstm_ctc_50k.yml
    CRNN_skel/config_CRNN_skel_resnet_bilstm_ctc_100k.yml
    CRNN_skel/config_CRNN_skel_resnet_bilstm_attention_10k.yml
    CRNN_skel/config_CRNN_skel_resnet_bilstm_attention_50k.yml
    CRNN_skel/config_CRNN_skel_resnet_bilstm_attention_100k.yml

    # DenseNet
    CRNN_skel/config_CRNN_skel_densenet_bilstm_ctc_10k.yml
    CRNN_skel/config_CRNN_skel_densenet_bilstm_ctc_50k.yml
    CRNN_skel/config_CRNN_skel_densenet_bilstm_ctc_100k.yml
    CRNN_skel/config_CRNN_skel_densenet_bilstm_attention_10k.yml
    CRNN_skel/config_CRNN_skel_densenet_bilstm_attention_50k.yml
    CRNN_skel/config_CRNN_skel_densenet_bilstm_attention_100k.yml
)

# ── Helpers ───────────────────────────────────────────────────────────────────

yaml_get() {
    python3 -c "
import yaml, sys
keys = sys.argv[2].split('.')
c = yaml.safe_load(open(sys.argv[1]))
for k in keys: c = c[k]
print(c)
" "$1" "$2"
}

is_finished() {
    python3 -c "
import yaml, csv, os, sys
config = yaml.safe_load(open(sys.argv[1]))
total_epochs = config['training']['epochs']
metrics = os.path.join(config['checkpoint']['checkpoint_dir'], 'metrics.csv')
if not os.path.exists(metrics):
    print('no'); sys.exit()
with open(metrics) as f:
    rows = list(csv.DictReader(f))
last_epoch = int(rows[-1]['epoch']) if rows else 0
print('yes' if last_epoch >= total_epochs else 'no')
" "$1"
}

make_patched_config() {
    python3 -c "
import yaml, sys, os, tempfile
config_path  = sys.argv[1]
dataset_base = sys.argv[2]
output_base  = sys.argv[3]
config = yaml.safe_load(open(config_path))
size = config['project']['name'].split('_')[-1]
new_path = os.path.join(dataset_base, f'khmer_{size}') + '/'
config['dataset']['path']    = new_path
config['dataset']['charset'] = os.path.join(new_path, 'char.json')

# Redirect outputs under OUTPUT_BASE, dropping the config's leading prefix
# (e.g. 'outputs_crnn') but keeping the per-run subpath.
def remap(p):
    parts = p.split('/', 1)
    rest = parts[1] if len(parts) > 1 else parts[0]
    return os.path.join(output_base, rest)
config['project']['save_dir']          = remap(config['project']['save_dir'])
config['checkpoint']['checkpoint_dir'] = remap(config['checkpoint']['checkpoint_dir'])

tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False, prefix='crnn_run_')
yaml.dump(config, tmp)
tmp.close()
print(tmp.name)
" "$1" "$2" "$3"
}

# ── Main ──────────────────────────────────────────────────────────────────────
LOG_DIR="logs/crnn_skel"
mkdir -p "$LOG_DIR"

TOTAL=${#CONFIGS[@]}
PASSED=0 SKIPPED=0 FAILED=0
FAILED_NAMES=()

echo "======================================================"
echo "  CRNN Training (SKELETON)"
echo "  Configs  : $TOTAL"
echo "  Dataset  : $DATASET_BASE"
echo "  Output   : $OUTPUT_BASE"
echo "  Logs     : $LOG_DIR"
echo "  Mode     : $( [[ $CONTINUE -eq 1 ]] && echo '--continue' || echo 'normal' )"
echo "======================================================"

START_ALL=$(date +%s)

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME=$(basename "$CONFIG" .yml)
    LOG_FILE="$LOG_DIR/${NAME}.log"
    NUM=$((i + 1))

    echo ""

    if [[ ! -f "$CONFIG" ]]; then
        echo "[$NUM/$TOTAL] MISSING: $CONFIG"
        FAILED=$((FAILED + 1)); FAILED_NAMES+=("$NAME [missing]")
        [[ $CONTINUE -eq 0 ]] && { echo "  Stopping. Re-run with --continue to skip."; break; }
        continue
    fi

    TMP_CONFIG=$(make_patched_config "$CONFIG" "$DATASET_BASE" "$OUTPUT_BASE")

    if [[ "$(is_finished "$TMP_CONFIG")" == "yes" ]]; then
        echo "[$NUM/$TOTAL] SKIP (already finished): $NAME"
        SKIPPED=$((SKIPPED + 1)); rm -f "$TMP_CONFIG"; continue
    fi

    CKPT_DIR=$(yaml_get "$TMP_CONFIG" "checkpoint.checkpoint_dir")
    LAST_CKPT="$CKPT_DIR/last_model.pth"
    RESUME_FLAG=""
    if [[ -f "$LAST_CKPT" ]]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') RESUMING : $NAME"
        echo "  checkpoint : $LAST_CKPT"
    else
        echo "[$NUM/$TOTAL] $(date '+%Y-%m-%d %H:%M:%S') START     : $NAME"
    fi
    echo "  log        : $LOG_FILE"
    echo "──────────────────────────────────────────────────────"

    START_T=$(date +%s)

    # shellcheck disable=SC2086
    python3 tools/train.py --config "$TMP_CONFIG" $RESUME_FLAG 2>&1 | tee -a "$LOG_FILE"
    EXIT_CODE=${PIPESTATUS[0]}
    rm -f "$TMP_CONFIG"

    ELAPSED=$(( $(date +%s) - START_T ))
    FMT=$(printf '%02dh %02dm %02ds' $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

    if [[ $EXIT_CODE -eq 0 ]]; then
        PASSED=$((PASSED + 1))
        echo "[$NUM/$TOTAL] DONE: $NAME  ($FMT)"
    else
        FAILED=$((FAILED + 1)); FAILED_NAMES+=("$NAME")
        echo "[$NUM/$TOTAL] FAILED: $NAME  (exit $EXIT_CODE, $FMT)"
        echo "  See log: $LOG_FILE"
        [[ $CONTINUE -eq 0 ]] && { echo "  Stopping. Re-run with --continue to skip failed models."; break; }
    fi
done

TOTAL_ELAPSED=$(( $(date +%s) - START_ALL ))
TOTAL_FMT=$(printf '%02dh %02dm %02ds' $((TOTAL_ELAPSED/3600)) $((TOTAL_ELAPSED%3600/60)) $((TOTAL_ELAPSED%60)))

echo ""
echo "======================================================"
echo "  Done    : $PASSED"
echo "  Skipped : $SKIPPED  (already finished)"
echo "  Failed  : $FAILED"
echo "  Total   : $TOTAL_FMT"
if [[ ${#FAILED_NAMES[@]} -gt 0 ]]; then
    echo "  Failed configs:"
    for N in "${FAILED_NAMES[@]}"; do echo "    - $N"; done
fi
echo "======================================================"
