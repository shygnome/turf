#!/usr/bin/env bash
# Run the full LEAK analysis pipeline for every match in a dataset.
#
# Steps per match (in order):
#   1. event extract    — clip tracking frames around every pass event
#   2. leak extract-line — detect defending unit lines per clip
#   3. leak label-pass  — label each pass as line-breaking or not
#   4. leak visualize-line — render unit-line GIF animations
#
# Usage:
#   ./scripts/run_leak_pipeline.sh [OPTIONS]
#
# Options:
#   -d DATASET_ID     Dataset to process (default: pff/fifa-wc-2022)
#   -o OUTPUT_DIR     Output root directory (default: output)
#   -l LOG_DIR        Directory for per-match log files (default: logs/leak)
#   -j N              Run N matches in parallel (default: 1 = serial)
#   -v EVENT_IDX      Which events to visualize: a number, "all", or
#                     "none" to skip visualization (default: first 10)
#   --min-line-gap G  Min metres between line means for extract-line (default: 0)
#   --resume          Skip matches that already have labeled_metadata.csv
#   -h                Show this help message
#
# Environment:
#   TURF_CMD   Override the turf invocation (default: uv run turf)
#
# Exit codes:
#   0  All matches succeeded
#   1  One or more matches had failures (details in per-match log files)

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
DATASET_ID="pff/fifa-wc-2022"
OUTPUT_DIR="output"
LOG_DIR="logs/leak"
PARALLEL=1
EVENT_IDX=""          # empty = default (first 10)
MIN_LINE_GAP="0"
RESUME=0
TURF_CMD="${TURF_CMD:-uv run turf}"

# ── argument parsing ──────────────────────────────────────────────────────────
usage() {
    sed -n '/^# Usage:/,/^[^#]/p' "$0" | sed 's/^# \{0,2\}//' | head -n -1
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d) DATASET_ID="$2"; shift 2 ;;
        -o) OUTPUT_DIR="$2"; shift 2 ;;
        -l) LOG_DIR="$2"; shift 2 ;;
        -j) PARALLEL="$2"; shift 2 ;;
        -v) EVENT_IDX="$2"; shift 2 ;;
        --min-line-gap) MIN_LINE_GAP="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ── derived paths ─────────────────────────────────────────────────────────────
# Match IDs are discovered from the preprocessed event CSV files.
# data/preprocessed/<dataset_id>/event/event_data_<match_id>.csv
DATA_EVENT_DIR="data/preprocessed/${DATASET_ID}/event"

if [[ ! -d "$DATA_EVENT_DIR" ]]; then
    echo "ERROR: event data directory not found: $DATA_EVENT_DIR" >&2
    exit 1
fi

mapfile -t MATCH_IDS < <(
    ls "$DATA_EVENT_DIR"/event_data_*.csv 2>/dev/null \
        | sed 's|.*/event_data_||; s|\.csv$||' \
        | sort -n
)

if [[ ${#MATCH_IDS[@]} -eq 0 ]]; then
    echo "ERROR: no match files found in $DATA_EVENT_DIR" >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# ── helpers ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

log()  { echo -e "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo -e "${GREEN}[OK]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
skip() { echo -e "${YELLOW}[SKIP]${NC} $*"; }

run_step() {
    local label="$1"; shift
    local log_file="$1"; shift
    if "$@" >> "$log_file" 2>&1; then
        ok "$label"
        return 0
    else
        fail "$label (see $log_file)"
        return 1
    fi
}

# ── per-match pipeline ────────────────────────────────────────────────────────
process_match() {
    local match_id="$1"
    local log_file="${LOG_DIR}/${match_id}.log"
    local pass_dir="${OUTPUT_DIR}/${DATASET_ID}/${match_id}/pass"
    local failed=0

    # Resume: skip if labeled output already exists
    if [[ $RESUME -eq 1 && -f "${pass_dir}/labeled_metadata.csv" ]]; then
        skip "match ${match_id} (already labeled)"
        return 0
    fi

    log "=== match ${match_id} ==="
    : > "$log_file"   # truncate / create log

    # Step 1 — event extract
    run_step "  [1/4] event extract    ${match_id}" "$log_file" \
        $TURF_CMD event extract "$DATASET_ID" "$match_id" pass \
        || { ((failed++)); }

    # Step 2 — extract-line (only if step 1 produced metadata.csv)
    if [[ -f "${pass_dir}/metadata.csv" ]]; then
        run_step "  [2/4] extract-line     ${match_id}" "$log_file" \
            $TURF_CMD analyze leak extract-line "$DATASET_ID" "$match_id" \
            --min-line-gap "$MIN_LINE_GAP" \
            || { ((failed++)); }
    else
        fail "  [2/4] extract-line     ${match_id} — skipped (no metadata.csv)"
        ((failed++))
    fi

    # Step 3 — label-pass (only if extract-line produced any lines.csv)
    if [[ -f "${pass_dir}/metadata.csv" ]]; then
        run_step "  [3/4] label-pass      ${match_id}" "$log_file" \
            $TURF_CMD analyze leak label-pass "$DATASET_ID" "$match_id" \
            || { ((failed++)); }
    else
        fail "  [3/4] label-pass      ${match_id} — skipped"
        ((failed++))
    fi

    # Step 4 — visualize-line
    if [[ -f "${pass_dir}/labeled_metadata.csv" ]]; then
        local viz_args=()
        if [[ -n "$EVENT_IDX" && "$EVENT_IDX" != "none" ]]; then
            viz_args+=(--event-idx "$EVENT_IDX")
        fi
        if [[ "$EVENT_IDX" == "none" ]]; then
            ok "  [4/4] visualize-line  ${match_id} — skipped (EVENT_IDX=none)"
        else
            run_step "  [4/4] visualize-line  ${match_id}" "$log_file" \
                $TURF_CMD analyze leak visualize-line "$DATASET_ID" "$match_id" \
                --show-labels "${viz_args[@]}" \
                || { ((failed++)); }
        fi
    else
        fail "  [4/4] visualize-line  ${match_id} — skipped (no labeled_metadata.csv)"
        ((failed++))
    fi

    return $failed
}

export -f process_match run_step log ok fail skip
export DATASET_ID OUTPUT_DIR LOG_DIR MIN_LINE_GAP RESUME TURF_CMD EVENT_IDX RED GREEN YELLOW NC

# ── main ─────────────────────────────────────────────────────────────────────
TOTAL=${#MATCH_IDS[@]}
log "Dataset:  $DATASET_ID"
log "Matches:  $TOTAL"
log "Parallel: $PARALLEL"
log "Logs:     $LOG_DIR"
log "Output:   $OUTPUT_DIR"
[[ $RESUME -eq 1 ]] && log "Resume:   ON (skipping matches with labeled_metadata.csv)"
echo ""

FAILED_MATCHES=()

if [[ $PARALLEL -le 1 ]]; then
    # Serial execution
    for match_id in "${MATCH_IDS[@]}"; do
        if ! process_match "$match_id"; then
            FAILED_MATCHES+=("$match_id")
        fi
    done
else
    # Parallel execution via xargs
    # Each match runs in a subshell; failures collected via a temp file
    FAIL_LOG=$(mktemp)
    printf '%s\n' "${MATCH_IDS[@]}" | xargs -P "$PARALLEL" -I{} bash -c '
        if ! process_match "{}"; then
            echo "{}" >> '"$FAIL_LOG"'
        fi
    '
    if [[ -s "$FAIL_LOG" ]]; then
        mapfile -t FAILED_MATCHES < "$FAIL_LOG"
    fi
    rm -f "$FAIL_LOG"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
log "═══════════════════════════════════════"
SUCCEEDED=$(( TOTAL - ${#FAILED_MATCHES[@]} ))
log "Done: ${SUCCEEDED}/${TOTAL} matches succeeded"

if [[ ${#FAILED_MATCHES[@]} -gt 0 ]]; then
    fail "Failed matches (${#FAILED_MATCHES[@]}):"
    for m in "${FAILED_MATCHES[@]}"; do
        echo "    $m  →  ${LOG_DIR}/${m}.log"
    done
    exit 1
fi

ok "All matches completed successfully."
