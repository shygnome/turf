#!/usr/bin/env bash
# Submit the LEAK pipeline as a SLURM job array.
# One array task per match; each task runs the full 4-step pipeline.
#
# Usage:
#   bash scripts/submit_leak_pipeline.sh [OPTIONS]
#
# Options:
#   -d DATASET_ID       Dataset to process (default: pff/fifa-wc-2022)
#   -o OUTPUT_DIR       Output root (default: output)
#   -l LOG_DIR          Directory for SLURM logs (default: logs/leak)
#   -v EVENT_IDX        Events to visualize: number, "all", or "none" to
#                       skip visualization (default: first 10)
#   --min-line-gap G    Min metres between line means (default: 0)
#   --resume            Skip matches that already have labeled_metadata.csv
#   --time HH:MM:SS     Per-task wall time (default: 04:00:00)
#   --mem MEM           Memory per task, e.g. 8G (default: 8G)
#   --cpus N            CPUs per task (default: 1)
#   --partition NAME    SLURM partition (default: not set — cluster default)
#   --account NAME      SLURM account/project (default: not set)
#   --dry-run           Print the sbatch command without submitting
#   -h                  Show this help
#
# After submission, monitor progress with:
#   squeue -u $USER --name=turf-leak
#   tail -f logs/leak/slurm-<JOB_ID>_<TASK>.out

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────────────
DATASET_ID="pff/fifa-wc-2022"
OUTPUT_DIR="output"
LOG_DIR="logs/leak"
EVENT_IDX=""
MIN_LINE_GAP="0"
RESUME=0
WALL_TIME="04:00:00"
MEM="8G"
CPUS=1
PARTITION=""
ACCOUNT=""
DRY_RUN=0

usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | sed 's/^# \{0,2\}//' | head -n -1
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d)            DATASET_ID="$2";   shift 2 ;;
        -o)            OUTPUT_DIR="$2";   shift 2 ;;
        -l)            LOG_DIR="$2";      shift 2 ;;
        -v)            EVENT_IDX="$2";    shift 2 ;;
        --min-line-gap) MIN_LINE_GAP="$2"; shift 2 ;;
        --resume)      RESUME=1;          shift ;;
        --time)        WALL_TIME="$2";    shift 2 ;;
        --mem)         MEM="$2";          shift 2 ;;
        --cpus)        CPUS="$2";         shift 2 ;;
        --partition)   PARTITION="$2";    shift 2 ;;
        --account)     ACCOUNT="$2";      shift 2 ;;
        --dry-run)     DRY_RUN=1;         shift ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── discover matches ──────────────────────────────────────────────────────────
DATA_EVENT_DIR="data/preprocessed/${DATASET_ID}/event"
if [[ ! -d "$DATA_EVENT_DIR" ]]; then
    echo "ERROR: event data directory not found: $DATA_EVENT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

MATCHES_FILE="${LOG_DIR}/matches.txt"
ls "$DATA_EVENT_DIR"/event_data_*.csv 2>/dev/null \
    | sed 's|.*/event_data_||; s|\.csv$||' \
    | sort -n > "$MATCHES_FILE"

N_MATCHES=$(wc -l < "$MATCHES_FILE")
if [[ $N_MATCHES -eq 0 ]]; then
    echo "ERROR: no match files found in $DATA_EVENT_DIR" >&2
    exit 1
fi

ARRAY_SPEC="0-$((N_MATCHES - 1))"

# ── build sbatch command ──────────────────────────────────────────────────────
SBATCH_ARGS=(
    --job-name=turf-leak
    --time="$WALL_TIME"
    --mem="$MEM"
    --cpus-per-task="$CPUS"
    --output="${LOG_DIR}/slurm-%A_%a.out"
    --error="${LOG_DIR}/slurm-%A_%a.err"
    --array="$ARRAY_SPEC"
    --export="ALL,DATASET_ID=${DATASET_ID},OUTPUT_DIR=${OUTPUT_DIR},MATCHES_FILE=${MATCHES_FILE},MIN_LINE_GAP=${MIN_LINE_GAP},EVENT_IDX=${EVENT_IDX},RESUME=${RESUME}"
)

[[ -n "$PARTITION" ]] && SBATCH_ARGS+=(--partition="$PARTITION")
[[ -n "$ACCOUNT"   ]] && SBATCH_ARGS+=(--account="$ACCOUNT")

# ── summary ───────────────────────────────────────────────────────────────────
echo "Dataset:      $DATASET_ID"
echo "Matches:      $N_MATCHES  ($MATCHES_FILE)"
echo "Array:        $ARRAY_SPEC"
echo "Wall time:    $WALL_TIME  |  Memory: $MEM  |  CPUs: $CPUS"
[[ -n "$PARTITION" ]] && echo "Partition:    $PARTITION"
[[ -n "$ACCOUNT"   ]] && echo "Account:      $ACCOUNT"
[[ $RESUME -eq 1   ]] && echo "Resume:       ON"
echo ""

if [[ $DRY_RUN -eq 1 ]]; then
    echo "DRY RUN — sbatch command:"
    echo "  sbatch ${SBATCH_ARGS[*]} scripts/slurm_task.sh"
    exit 0
fi

JOB_OUTPUT=$(sbatch "${SBATCH_ARGS[@]}" scripts/slurm_task.sh)
echo "$JOB_OUTPUT"
JOB_ID=$(echo "$JOB_OUTPUT" | awk '{print $NF}')

echo ""
echo "Monitor:  squeue -u \$USER --name=turf-leak"
echo "Logs:     ${LOG_DIR}/slurm-${JOB_ID}_<task>.out"
echo "Cancel:   scancel $JOB_ID"
