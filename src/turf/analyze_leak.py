from __future__ import annotations

from pathlib import Path

import typer

from turf.dataset import CATALOG

app = typer.Typer(help="Analyse match data.", no_args_is_help=True)
_leak_app = typer.Typer(help="LEAK unit line detection.", no_args_is_help=True)
app.add_typer(_leak_app, name="leak")

_DEFAULT_OUTPUT_ROOT = Path("output")
_DEFAULT_GOALS_PATH = Path("data/wc2022_goals.csv")
_DEFAULT_VISUALIZE_LIMIT = 10


def _has_any_around(direction_str: object) -> bool:
    """True if direction_per_line contains at least one 'Around'."""
    import ast

    try:
        return "Around" in ast.literal_eval(str(direction_str))
    except (ValueError, SyntaxError):
        return False


def _is_advanced_around(direction_str: object) -> bool:
    """True if the most-advanced line (first element) was beaten 'Around'."""
    import ast

    try:
        dirs = ast.literal_eval(str(direction_str))
        return bool(dirs) and dirs[0] == "Around"
    except (ValueError, SyntaxError):
        return False


def get_output_root() -> Path:
    return _DEFAULT_OUTPUT_ROOT


def _resolve_pass_dir(dataset_id: str, match_id: str) -> tuple[Path, Path]:
    output_root = get_output_root().resolve()
    out_dir = (output_root / dataset_id / match_id / "pass").resolve()
    if not out_dir.is_relative_to(output_root):
        typer.echo("Invalid output path.", err=True)
        raise typer.Exit(1)
    return output_root, out_dir


@_leak_app.command("extract-line")
def extract_line(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    min_line_gap: float = typer.Option(
        0.0, "--min-line-gap", help="Min metres between adjacent line means."
    ),
) -> None:
    """Detect unit lines for the defending team in every extracted pass clip."""
    if min_line_gap < 0:
        typer.echo("--min-line-gap must be a non-negative number.", err=True)
        raise typer.Exit(1)

    import pandas as pd  # type: ignore[import-untyped]

    from leak.lines import analyze_lines, fix_player_assignments, vote_line_count

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. Run 'turf event extract' first.",
            err=True,
        )
        raise typer.Exit(1)

    metadata = pd.read_csv(meta_path)
    count = 0
    for _, row in metadata.iterrows():
        event_idx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = out_dir / str(event_idx)
        frames_path = event_dir / f"frames_{defending_team.lower()}.csv"
        if not frames_path.exists():
            typer.echo(f"  skip {event_idx}: frames file not found", err=True)
            continue

        frames_df = pd.read_csv(frames_path)
        try:
            first_pass = analyze_lines(
                frames_df, defending_team, min_line_gap=min_line_gap
            )
            k_voted = vote_line_count(first_pass)
            lines_df = fix_player_assignments(
                analyze_lines(
                    frames_df,
                    defending_team,
                    min_line_gap=min_line_gap,
                    force_n_lines=k_voted,
                ),
                defending_team,
            )
        except ValueError as exc:
            typer.echo(f"  skip {event_idx}: {exc}", err=True)
            continue

        lines_df.to_csv(event_dir / "lines.csv", index=False)
        count += 1

    typer.echo(f"Extracted unit lines for {count} event(s) to {out_dir}")


@_leak_app.command("label-pass")
def label_pass(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
) -> None:
    """Label pass events with line-break attributes (is_line_breaking, lines_broken)."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    output_root, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. "
            "Run 'turf event extract' and 'turf analyze leak extract-line' first.",
            err=True,
        )
        raise typer.Exit(1)

    from leak.pass_label import label_all_passes

    labeled = label_all_passes(output_root, dataset_id, match_id)
    n_labeled = int(labeled["is_line_breaking"].notna().sum())
    n_breaking = int(labeled["is_line_breaking"].eq(True).sum())
    typer.echo(
        f"Labeled {n_labeled} pass event(s): "
        f"{n_breaking} line-breaking. "
        f"Written to {out_dir / 'labeled_metadata.csv'}"
    )


@_leak_app.command("stats-pass")
def stats_pass(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
) -> None:
    """Print summary statistics for labeled pass events."""
    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    labeled_path = out_dir / "labeled_metadata.csv"
    if not labeled_path.exists():
        typer.echo(
            f"labeled_metadata.csv not found at {labeled_path}. "
            "Run 'turf analyze leak label-pass' first.",
            err=True,
        )
        raise typer.Exit(1)

    import pandas as pd  # noqa: PLC0415

    from leak.pass_label import compute_pass_stats

    df = pd.read_csv(labeled_path)
    s = compute_pass_stats(df)

    def _pct(n: int, total: int) -> str:
        return f"{n / total * 100:.1f}%" if total else "—"

    sep = "-" * 50
    header = f"Pass labeling summary - {dataset_id} / {match_id}"
    typer.echo(f"\n{header}")
    typer.echo(sep)

    skip_note = f"  ({s['n_skipped']} skipped - no lines.csv)" if s["n_skipped"] else ""
    typer.echo(f"Total labeled:      {s['n_labeled']:>5}{skip_note}")
    n_tot = s["n_labeled"]
    typer.echo(
        f"Line-breaking:      {s['n_breaking']:>5}  ({_pct(s['n_breaking'], n_tot)})"
    )
    typer.echo(
        f"Not line-breaking:  {s['n_not_breaking']:>5}"
        f"  ({_pct(s['n_not_breaking'], n_tot)})"
    )

    if s["lines_broken_dist"]:
        typer.echo(f"\nLines broken (of {s['n_breaking']} line-breaking passes)")
        for k in sorted(s["lines_broken_dist"]):
            cnt = s["lines_broken_dist"][k]
            label = "line " if k == 1 else "lines"
            typer.echo(f"  {k} {label}:  {cnt:>4}  ({_pct(cnt, s['n_breaking'])})")

    if s["by_subtype"]:
        typer.echo("\nBy outcome (subtype)")
        for subtype, v in s["by_subtype"].items():
            typer.echo(
                f"  {subtype:<12} {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    if s["by_team"]:
        typer.echo("\nBy team")
        for team, v in s["by_team"].items():
            typer.echo(
                f"  {team:<8} {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    if s["by_period"]:
        typer.echo("\nBy period")
        for period, v in s["by_period"].items():
            typer.echo(
                f"  Period {period}  {v['breaking']:>4} / {v['total']:>4}"
                f"  ({_pct(v['breaking'], v['total'])})"
            )

    typer.echo("")


@_leak_app.command("around-stats")
def around_stats(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    goals_path: Path = typer.Option(  # noqa: B008
        _DEFAULT_GOALS_PATH,
        "--goals",
        help="Path to goals CSV.",
    ),
    output_root: Path = typer.Option(  # noqa: B008
        _DEFAULT_OUTPUT_ROOT,
        "--output-root",
        help="Root directory for output files.",
    ),
) -> None:
    """Cross-match around-break win-rate statistics (two scenarios) for DATASET_ID."""
    import pandas as pd

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    if not goals_path.exists():
        typer.echo(f"Goals CSV not found: {goals_path}", err=True)
        raise typer.Exit(1)

    goals_raw = pd.read_csv(goals_path, comment="#")
    score = (
        goals_raw.groupby(["match_id", "scoring_team"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Home", "Away"], fill_value=0)
        .rename(columns={"Home": "home_goals", "Away": "away_goals"})
    )

    labeled_dir = output_root.resolve() / dataset_id
    if not labeled_dir.exists():
        typer.echo("No labeled data found.", err=True)
        raise typer.Exit(1)

    records: list[dict[str, object]] = []
    n_loaded = 0
    for match_path in sorted(labeled_dir.iterdir()):
        csv_path = match_path / "pass" / "labeled_metadata.csv"
        if not csv_path.exists():
            continue
        try:
            match_id = int(match_path.name)
        except ValueError:
            continue

        lm = pd.read_csv(csv_path)
        lm = lm[lm["subtype"] == "success"].copy()
        is_breaking = lm["is_line_breaking"].astype(bool)
        lm["_all"] = is_breaking
        lm["_any"] = is_breaking & lm["direction_per_line"].apply(_has_any_around)
        lm["_adv"] = is_breaking & lm["direction_per_line"].apply(_is_advanced_around)

        poss_path = (
            output_root.resolve() / dataset_id / str(match_id)
            / "possession_summary.csv"
        )
        if not poss_path.exists():
            typer.echo(
                f"[skip] match {match_id}: possession summary missing "
                "(run `turf dataset possession` first).",
                err=True,
            )
            continue
        poss = pd.read_csv(poss_path)

        n_loaded += 1
        for team in ("Home", "Away"):
            t = lm[lm["team"] == team]
            col_sec = "home_sec" if team == "Home" else "away_sec"
            poss_sec = float(poss[col_sec].sum())
            records.append(
                {
                    "match_id": match_id,
                    "team": team,
                    "poss_min": poss_sec / 60.0,
                    "all_breaks": int(t["_all"].sum()),
                    "around_any": int(t["_any"].sum()),
                    "around_adv": int(t["_adv"].sum()),
                }
            )

    if not records:
        typer.echo("No labeled_metadata.csv files found.", err=True)
        raise typer.Exit(1)

    df = pd.DataFrame(records)
    for col in ("all_breaks", "around_any", "around_adv"):
        df[f"{col}_per_30"] = df[col] / (df["poss_min"] / 30)

    pivot = df.pivot(
        index="match_id",
        columns="team",
        values=["all_breaks_per_30", "around_any_per_30", "around_adv_per_30"],
    )
    pivot.columns = pd.Index([f"{c}_{t.lower()}" for c, t in pivot.columns])
    pivot = pivot.merge(
        score[["home_goals", "away_goals"]],
        left_index=True,
        right_index=True,
        how="inner",
    )

    def _win_stats(col: str) -> tuple[float, float, int]:
        wins = 0
        n_clear = 0
        diffs: list[float] = []
        for _, row in pivot.iterrows():
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            if hg > ag:
                winner, loser = "home", "away"
            elif ag > hg:
                winner, loser = "away", "home"
            else:
                continue
            h_rate: float = float(row[f"{col}_per_30_home"])
            a_rate: float = float(row[f"{col}_per_30_away"])
            w_rate = float(row[f"{col}_per_30_{winner}"])
            l_rate = float(row[f"{col}_per_30_{loser}"])
            diffs.append(w_rate - l_rate)
            if h_rate != a_rate:
                n_clear += 1
                more = "home" if h_rate > a_rate else "away"
                if more == winner:
                    wins += 1
        win_pct = wins / n_clear * 100 if n_clear else float("nan")
        avg_diff = sum(diffs) / len(diffs) if diffs else float("nan")
        return win_pct, avg_diff, n_clear

    n_decisive = sum(
        1 for _, r in pivot.iterrows()
        if int(r["home_goals"]) != int(r["away_goals"])
    )

    sep = "-" * 56
    typer.echo(f"\nAround-break analysis: {dataset_id}")
    typer.echo(f"Labeled matches: {n_loaded}  |  Decisive results: {n_decisive}")
    typer.echo(sep)

    for label, col in (
        ("Scenario 0 - all line breaks (any direction)", "all_breaks"),
        ("Scenario 1 - around breaks (any line)", "around_any"),
        ("Scenario 2 - around break (most advanced line only)", "around_adv"),
    ):
        win_pct, avg_diff, n_clear = _win_stats(col)
        typer.echo(f"\n{label}")
        typer.echo(
            f"  Win rate (more around => won) : {win_pct:>6.1f}%"
            f"  (of {n_clear} decisive matches with unequal rates)"
        )
        typer.echo(f"  Avg (winner - loser) /30 min  : {avg_diff:>6.2f}")

    typer.echo("")


@_leak_app.command("visualize-line")
def visualize_line(
    dataset_id: str = typer.Argument(..., help="Dataset ID from the catalog."),
    match_id: str = typer.Argument(..., help="Match ID."),
    event_idx: str | None = typer.Option(
        None,
        "--event-idx",
        "-i",
        help=(
            "Event index to visualize, or 'all' for every event. "
            f"Default: first {_DEFAULT_VISUALIZE_LIMIT}."
        ),
    ),
    fps: float = typer.Option(25.0, "--fps", help="Frames per second for animation."),
    smooth_lines: bool = typer.Option(
        True, "--smooth-lines/--no-smooth-lines", help="Smooth line assignments."
    ),
    debug: bool = typer.Option(
        False, "--debug/--no-debug", help="Overlay inter-line gap labels."
    ),
    show_labels: bool = typer.Option(
        False,
        "--show-labels/--no-show-labels",
        help="Overlay Phase 2 pass labels as sequential reveal animation.",
    ),
) -> None:
    """Visualize defending team unit lines as animated GIF clips."""
    if fps <= 0:
        typer.echo("--fps must be a positive number.", err=True)
        raise typer.Exit(1)

    entry = next((e for e in CATALOG if e.id == dataset_id), None)
    if entry is None:
        typer.echo(f"Unknown dataset: {dataset_id}", err=True)
        raise typer.Exit(1)

    _, out_dir = _resolve_pass_dir(dataset_id, match_id)

    meta_path = out_dir / "metadata.csv"
    if not meta_path.exists():
        typer.echo(
            f"metadata.csv not found at {meta_path}. Run 'turf event extract' first.",
            err=True,
        )
        raise typer.Exit(1)

    import ast

    import matplotlib.pyplot as plt
    import pandas as pd
    from tqdm import tqdm  # type: ignore[import-untyped]

    from turf.leak_lines_visualizer import LeakLinesVisualizer

    metadata = pd.read_csv(meta_path)

    # ── labeled metadata for --show-labels ───────────────────────────────────
    labeled_lookup: dict[int, dict[str, object]] = {}
    if show_labels:
        labeled_path = out_dir / "labeled_metadata.csv"
        if labeled_path.exists():
            ldf = pd.read_csv(labeled_path)
            for _, lr in ldf.iterrows():
                if pd.notna(lr.get("is_line_breaking")):
                    labeled_lookup[int(lr["event_idx"])] = dict(lr)
        else:
            typer.echo(
                f"  Warning: {labeled_path} not found — run 'label-pass' first.",
                err=True,
            )

    if event_idx is None:
        metadata = metadata.head(_DEFAULT_VISUALIZE_LIMIT)
    elif event_idx.strip().lower() != "all":
        try:
            idx = int(event_idx)
        except ValueError:
            typer.echo("--event-idx must be a number or 'all'.", err=True)
            raise typer.Exit(1) from None
        metadata = metadata[metadata["event_idx"] == idx]
        if metadata.empty:
            typer.echo(f"Event index {idx} not found.", err=True)
            raise typer.Exit(1)

    viz = LeakLinesVisualizer()
    count = 0
    for _, row in tqdm(
        metadata.iterrows(), total=len(metadata), desc="visualizing", unit="event"
    ):
        eidx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = out_dir / str(eidx)
        lines_path = event_dir / "lines.csv"
        atk_frames_path = event_dir / f"frames_{attacking_team.lower()}.csv"

        if not lines_path.exists():
            typer.echo(
                f"  skip {eidx}: lines file not found (run extract-line first)",
                err=True,
            )
            continue
        if not atk_frames_path.exists():
            typer.echo(f"  skip {eidx}: attacking frames not found", err=True)
            continue

        def_df = pd.read_csv(lines_path)
        atk_df = pd.read_csv(atk_frames_path)
        meta_dict: dict[str, object] = {
            "event_idx": eidx,
            "period": int(row["period"]),
            "team": attacking_team,
        }

        n_frames = min(len(def_df), len(atk_df))
        gif_path = event_dir / "linevis.gif"

        # ── build pass_labels for --show-labels ──────────────────────────────
        pass_labels: dict[str, object] | None = None
        if show_labels and eidx in labeled_lookup:
            lrow = labeled_lookup[eidx]
            if bool(lrow.get("is_line_breaking")):
                from leak.pass_label import (
                    compute_team_hull_vertices,
                    detect_attack_direction,
                )

                try:
                    attack_dir = detect_attack_direction(def_df, defending_team)
                    hull_verts = compute_team_hull_vertices(
                        def_df, defending_team, attack_dir
                    )

                    def _parse_list(raw: object) -> list[object]:
                        s = str(raw)
                        if s in ("nan", "None", ""):
                            return []
                        return ast.literal_eval(s)  # type: ignore[no-any-return]

                    lbc_raw = lrow.get("lines_broken_count")
                    lbc_ok = lbc_raw is not None and str(lbc_raw) != "nan"
                    lbc = int(float(str(lbc_raw))) if lbc_ok else 0
                    pass_labels = {
                        "is_line_breaking": bool(lrow["is_line_breaking"]),
                        "lines_broken_count": lbc,
                        "lines_broken": _parse_list(lrow.get("lines_broken", "[]")),
                        "direction_per_line": _parse_list(
                            lrow.get("direction_per_line", "[]")
                        ),
                        "location_after_break": lrow.get("location_after_break"),
                        "hull_vertices": hull_verts,
                    }
                except Exception:
                    pass_labels = None

        anim = viz.animate(
            def_df,
            atk_df,
            defending_team,
            attacking_team,
            meta_dict,
            fps=fps,
            smooth_lines=smooth_lines,
            debug=debug,
            pass_labels=pass_labels,
        )
        with tqdm(
            total=n_frames,
            desc=f"  gif {eidx}",
            unit="frame",
            leave=False,
        ) as frame_bar:

            def _on_frame(current: int, _total: int) -> None:
                frame_bar.update(1)

            anim.save(
                str(gif_path),
                writer="pillow",
                fps=int(fps),
                progress_callback=_on_frame,
            )

        plt.close("all")
        count += 1

    typer.echo(f"Visualized unit lines for {count} event(s) to {out_dir}")
