#!/usr/bin/env bash
# SLURM array task — runs the full LEAK pipeline for one match.
# Each array index maps to one line in the matches file.
# Submit via: bash scripts/submit_leak_pipeline.sh
#
# Resource defaults (override with sbatch flags or in submit script):
#SBATCH --job-name=turf-leak
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/leak/slurm-%A_%a.out
#SBATCH --error=logs/leak/slurm-%A_%a.err

set -euo pipefail

# ── read match ID for this task ───────────────────────────────────────────────
MATCHES_FILE="${MATCHES_FILE:-logs/leak/matches.txt}"
if [[ ! -f "$MATCHES_FILE" ]]; then
    echo "ERROR: matches file not found: $MATCHES_FILE" >&2
    exit 1
fi

MATCH_ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MATCHES_FILE")
if [[ -z "$MATCH_ID" ]]; then
    echo "ERROR: no match ID at line $((SLURM_ARRAY_TASK_ID + 1)) of $MATCHES_FILE" >&2
    exit 1
fi

# ── configurable env vars (set by submit script or sbatch --export) ───────────
DATASET_ID="${DATASET_ID:-pff/fifa-wc-2022}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
MIN_LINE_GAP="${MIN_LINE_GAP:-0}"
EVENT_IDX="${EVENT_IDX:-}"       # empty = default (first 10); "all"; "none"
RESUME="${RESUME:-0}"
TURF_CMD="${TURF_CMD:-uv run turf}"

# ── paths ─────────────────────────────────────────────────────────────────────
PASS_DIR="${OUTPUT_DIR}/${DATASET_ID}/${MATCH_ID}/pass"

echo "=== SLURM job ${SLURM_JOB_ID:-?} array ${SLURM_ARRAY_TASK_ID:-?} ==="
echo "Match:   ${MATCH_ID}"
echo "Dataset: ${DATASET_ID}"
echo "Node:    ${SLURMD_NODENAME:-local}"
date

# ── resume guard ──────────────────────────────────────────────────────────────
if [[ $RESUME -eq 1 && -f "${PASS_DIR}/labeled_metadata.csv" ]]; then
    echo "SKIP: labeled_metadata.csv already exists for match ${MATCH_ID}"
    exit 0
fi

# ── step 1: event extract ─────────────────────────────────────────────────────
echo ""
echo "[1/4] event extract — match ${MATCH_ID}"
$TURF_CMD event extract "$DATASET_ID" "$MATCH_ID" pass

# ── step 2: leak extract-line ─────────────────────────────────────────────────
if [[ ! -f "${PASS_DIR}/metadata.csv" ]]; then
    echo "ERROR: step 1 produced no metadata.csv — aborting match ${MATCH_ID}" >&2
    exit 1
fi

echo ""
echo "[2/4] leak extract-line — match ${MATCH_ID}"
$TURF_CMD analyze leak extract-line "$DATASET_ID" "$MATCH_ID" \
    --min-line-gap "$MIN_LINE_GAP"

# ── step 3: leak label-pass ───────────────────────────────────────────────────
echo ""
echo "[3/4] leak label-pass — match ${MATCH_ID}"
$TURF_CMD analyze leak label-pass "$DATASET_ID" "$MATCH_ID"

# ── step 4: leak visualize-line ───────────────────────────────────────────────
if [[ "$EVENT_IDX" == "none" ]]; then
    echo ""
    echo "[4/4] visualize-line — skipped (EVENT_IDX=none)"
else
    if [[ ! -f "${PASS_DIR}/labeled_metadata.csv" ]]; then
        echo "ERROR: step 3 produced no labeled_metadata.csv — aborting viz for ${MATCH_ID}" >&2
        exit 1
    fi

    echo ""
    echo "[4/4] leak visualize-line — match ${MATCH_ID}"
    viz_args=(--show-labels)
    if [[ -n "$EVENT_IDX" ]]; then
        viz_args+=(--event-idx "$EVENT_IDX")
    fi
    $TURF_CMD analyze leak visualize-line "$DATASET_ID" "$MATCH_ID" "${viz_args[@]}"
fi

echo ""
echo "Done — match ${MATCH_ID}"
date
