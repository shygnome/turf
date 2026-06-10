# TURF

CLI toolkit for soccer/football tracking and event research.

TURF loads preprocessed match data, extracts tracking clips around events, and
runs tactical analysis — currently focused on **LEAK unit line detection** for
the defending team.

---

## Requirements

- Python 3.10 or 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Install

```bash
git clone https://github.com/shygnome/turf
cd turf
uv sync
```

The `turf` command is available via `uv run turf`.

## Typical workflow

```bash
# 1. Point TURF at your data folder (saved to ~/.turf/config.toml)
uv run turf dataset set-root /path/to/data

# 2. See which datasets are in the catalog and present locally
uv run turf dataset ls

# 3. Preprocess a raw dataset
uv run turf dataset prepare pff/fifa-wc-2022

# 4. Browse matches
uv run turf match ls pff/fifa-wc-2022

# 5. Extract pass clips for a match
uv run turf event extract pff/fifa-wc-2022 3788741 pass

# 6. Detect defending-team unit lines for every pass clip
uv run turf analyze leak extract-line pff/fifa-wc-2022 3788741

# 7. Label each pass as line-breaking or not
uv run turf analyze leak label-pass pff/fifa-wc-2022 3788741

# 8. Print summary statistics
uv run turf analyze leak stats-pass pff/fifa-wc-2022 3788741
```

Output is written to `output/<dataset_id>/<match_id>/pass/`.

## Command reference

See [docs/commands.md](docs/commands.md) for the full list of commands,
arguments, and options.

## Development

```bash
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy src        # type-check
```