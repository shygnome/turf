# TURF — Command Reference

All commands are run with `uv run turf <command>`.

---

## Top-level

| Command | Description |
|---|---|
| `turf info` | Print version and project info |
| `turf --version` | Print version and exit |

---

## `turf dataset` — Manage datasets

### `dataset ls`

List all datasets in the catalog and whether they are present/prepared locally.

```
turf dataset ls
```

### `dataset set-root <path>`

Set the data root folder. Saved to `~/.turf/config.toml`.

```
turf dataset set-root /path/to/data
```

| Argument | Description |
|---|---|
| `path` | Path to the folder that contains raw dataset directories |

### `dataset prepare <dataset_id>`

Run `openstarlab-preprocessing` on a raw dataset and write the result to
`<root>/preprocessed/<dataset_id>/`. Only supported for datasets that ship with
a `PrepareSpec` (currently `pff/fifa-wc-2022`).

```
turf dataset prepare pff/fifa-wc-2022
```

| Argument | Description |
|---|---|
| `dataset_id` | ID from the catalog (see `dataset ls`) |

**Catalog IDs**

| ID | Provider | Competition |
|---|---|---|
| `pff/fifa-wc-2022` | PFF FC | FIFA World Cup 2022 |
| `sb/fifa-wc-2022` | StatsBomb | FIFA World Cup 2022 |
| `sb/laliga-2023-24` | StatsBomb | La Liga 2023/24 |

---

## `turf match` — Browse match data

### `match ls <dataset_id>`

List available matches for a prepared dataset, with optional pagination.

```
turf match ls pff/fifa-wc-2022 [--page N] [--per-page N]
```

| Argument / Option | Default | Description |
|---|---|---|
| `dataset_id` | — | Dataset ID from the catalog |
| `--page`, `-p` | `1` | Page number |
| `--per-page` | `20` | Matches per page |

### `match load <dataset_id> <match_id>`

Load a match and print a summary of the event and tracking DataFrames.

```
turf match load pff/fifa-wc-2022 3788741
```

| Argument | Description |
|---|---|
| `dataset_id` | Dataset ID from the catalog |
| `match_id` | Match ID (from `match ls`) |

---

## `turf event` — Extract and visualize events

### `event ls <dataset_id> <match_id>`

List all event types in a match and their counts.

```
turf event ls pff/fifa-wc-2022 3788741
```

| Argument | Description |
|---|---|
| `dataset_id` | Dataset ID from the catalog |
| `match_id` | Match ID |

### `event extract <dataset_id> <match_id> <label>`

Extract tracking frame clips for every occurrence of an event type.

Writes to `output/<dataset_id>/<match_id>/<label>/`:
- `metadata.csv` — one row per event with coordinates and match context
- `<event_idx>/frames_home.csv` — home-team tracking frames for the clip
- `<event_idx>/frames_away.csv` — away-team tracking frames for the clip

```
turf event extract pff/fifa-wc-2022 3788741 pass
```

| Argument / Option | Default | Description |
|---|---|---|
| `dataset_id` | — | Dataset ID from the catalog |
| `match_id` | — | Match ID |
| `label` | — | Event type (case-insensitive, e.g. `pass`, `cross`) |
| `--infer-endpoints` / `--no-infer-endpoints` | on | Infer pass/cross endpoint from the following event |

### `event visualize <dataset_id> <match_id> <label>`

Render extracted clips as freeze-frame PNGs and animated GIFs.

Writes to `output/<dataset_id>/<match_id>/<label>/<event_idx>/`:
- `freeze.png` — first-frame snapshot
- `clip.gif` — animated tracking clip

```
turf event visualize pff/fifa-wc-2022 3788741 pass
turf event visualize pff/fifa-wc-2022 3788741 pass --event-idx 42
turf event visualize pff/fifa-wc-2022 3788741 pass --event-idx all
```

| Argument / Option | Default | Description |
|---|---|---|
| `dataset_id` | — | Dataset ID from the catalog |
| `match_id` | — | Match ID |
| `label` | — | Event type label |
| `--event-idx`, `-i` | first 10 | Event index to visualize, or `all` |
| `--fps` | `25.0` | Frames per second for the GIF |
| `--smooth` | off | Apply Savitzky-Golay smoothing to positions |

---

## `turf analyze leak` — LEAK unit line detection

The LEAK pipeline runs in three steps that build on each other:
`extract-line` → `label-pass` → `stats-pass` / `visualize-line`.
All commands require `turf event extract ... pass` to have run first.

### `analyze leak extract-line <dataset_id> <match_id>`

Detect defending-team unit lines for each extracted pass clip using Ward
clustering. Writes `lines.csv` into every `<event_idx>/` directory.

```
turf analyze leak extract-line pff/fifa-wc-2022 3788741
turf analyze leak extract-line pff/fifa-wc-2022 3788741 --min-line-gap 2.0
```

| Argument / Option | Default | Description |
|---|---|---|
| `dataset_id` | — | Dataset ID from the catalog |
| `match_id` | — | Match ID |
| `--min-line-gap` | `0.0` | Minimum distance (metres) between adjacent line means |

### `analyze leak label-pass <dataset_id> <match_id>`

Label each pass event as line-breaking (`is_line_breaking`) and count how many
lines were broken (`lines_broken`). Writes
`output/<dataset_id>/<match_id>/pass/labeled_metadata.csv`.

Requires `extract-line` to have run.

```
turf analyze leak label-pass pff/fifa-wc-2022 3788741
```

| Argument | Description |
|---|---|
| `dataset_id` | Dataset ID from the catalog |
| `match_id` | Match ID |

### `analyze leak stats-pass <dataset_id> <match_id>`

Print a summary of the labeled pass data: total labeled, line-breaking rate,
breakdown by lines broken, outcome subtype, team, and period.

Requires `label-pass` to have run.

```
turf analyze leak stats-pass pff/fifa-wc-2022 3788741
```

| Argument | Description |
|---|---|
| `dataset_id` | Dataset ID from the catalog |
| `match_id` | Match ID |

### `analyze leak visualize-line <dataset_id> <match_id>`

Animate defending-team unit lines alongside ball trail and both teams' players.
Writes `linevis.gif` into every `<event_idx>/` directory.

Requires `extract-line` to have run.

```
turf analyze leak visualize-line pff/fifa-wc-2022 3788741
turf analyze leak visualize-line pff/fifa-wc-2022 3788741 --event-idx 42
turf analyze leak visualize-line pff/fifa-wc-2022 3788741 --event-idx all --no-smooth-lines
```

| Argument / Option | Default | Description |
|---|---|---|
| `dataset_id` | — | Dataset ID from the catalog |
| `match_id` | — | Match ID |
| `--event-idx`, `-i` | first 10 | Event index to visualize, or `all` |
| `--fps` | `25.0` | Frames per second for the GIF |
| `--smooth-lines` / `--no-smooth-lines` | on | Smooth line cluster assignments over time |
| `--debug` / `--no-debug` | off | Overlay inter-line gap labels on the animation |
