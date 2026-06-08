from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from typer.testing import CliRunner

from turf.cli import app

runner = CliRunner()

DATASET_ID = "pff/fifa-wc-2022"
MATCH_ID = "10502"


# ---------------------------------------------------------------------------
# helpers / data factories
# ---------------------------------------------------------------------------


def _make_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_idx": [0, 1, 2],
            "start_frame": [0, 5, 15],
            "end_frame": [4, 14, 24],
            "start_time": [0.0, 0.165, 0.495],
            "end_time": [0.132, 0.462, 0.792],
            "start_x": [0.1, 5.0, -5.0],
            "start_y": [0.2, 1.0, -1.0],
            "end_x": [1.0, 6.0, -4.0],
            "end_y": [1.2, 1.5, -0.5],
            "inferred_end_x": [None, None, None],
            "inferred_end_y": [None, None, None],
            "inferred_end_time": [None, None, None],
            "from_player": ["Alice", "Bob", "Charlie"],
            "to_player": ["Bob", "Alice", "Dave"],
            "team": ["Home", "Home", "Away"],
            "subtype": ["success", "success", "fail"],
            "period": [1, 1, 1],
        }
    )


def _make_away_frames_df() -> pd.DataFrame:
    rows = []
    for f in range(5):
        row: dict[str, object] = {
            "frame": f,
            "Period": 1,
            "Time [s]": float(f) * 0.033,
            "Away_3_x": 43.0,
            "Away_3_y": 0.0,
            "Away_2_x": 8.0,
            "Away_2_y": -12.0,
            "Away_11_x": 9.0,
            "Away_11_y": -5.0,
            "Away_6_x": 18.0,
            "Away_6_y": -6.0,
            "Away_7_x": 19.0,
            "Away_7_y": 6.0,
            "Away_9_x": 25.0,
            "Away_9_y": -4.0,
            "Away_10_x": 26.0,
            "Away_10_y": 4.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _make_home_frames_df() -> pd.DataFrame:
    rows = []
    for f in range(5):
        row: dict[str, object] = {
            "frame": f,
            "Period": 1,
            "Time [s]": float(f) * 0.033,
            "Home_1_x": -43.0,
            "Home_1_y": 0.0,
            "Home_2_x": -8.0,
            "Home_2_y": -12.0,
            "Home_11_x": -9.0,
            "Home_11_y": -5.0,
            "Home_6_x": -18.0,
            "Home_6_y": -6.0,
            "Home_7_x": -19.0,
            "Home_7_y": 6.0,
            "Home_9_x": -25.0,
            "Home_9_y": -4.0,
            "Home_10_x": -26.0,
            "Home_10_y": 4.0,
            "ball_x": 0.0,
            "ball_y": 0.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture()
def pass_output_dir(tmp_path: Path) -> Path:
    out_dir = tmp_path / "output" / Path(DATASET_ID) / MATCH_ID / "pass"
    out_dir.mkdir(parents=True, exist_ok=True)

    _make_metadata().to_csv(out_dir / "metadata.csv", index=False)

    away_df = _make_away_frames_df()
    home_df = _make_home_frames_df()

    # events 0,1: Home attacks → Away defends
    for idx in [0, 1]:
        event_dir = out_dir / str(idx)
        event_dir.mkdir(parents=True, exist_ok=True)
        away_df.to_csv(event_dir / "frames_away.csv", index=False)
        home_df.to_csv(event_dir / "frames_home.csv", index=False)

    # event 2: Away attacks → Home defends
    event_dir_2 = out_dir / "2"
    event_dir_2.mkdir(parents=True, exist_ok=True)
    home_df.to_csv(event_dir_2 / "frames_home.csv", index=False)
    away_df.to_csv(event_dir_2 / "frames_away.csv", index=False)

    return tmp_path / "output"


# ---------------------------------------------------------------------------
# extract-line — happy path
# ---------------------------------------------------------------------------


@pytest.fixture()
def extract_line_result(
    pass_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path]:
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: pass_output_dir)
    result = runner.invoke(
        app, ["analyze", "leak", "extract-line", DATASET_ID, MATCH_ID]
    )
    out_dir = pass_output_dir / Path(DATASET_ID) / MATCH_ID / "pass"
    return result, out_dir


def test_extract_line_exit_code(extract_line_result: tuple) -> None:
    result, _ = extract_line_result
    assert result.exit_code == 0


def test_extract_line_creates_lines_csv_for_home_attack(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    assert (out_dir / "0" / "lines.csv").exists()
    assert (out_dir / "1" / "lines.csv").exists()


def test_extract_line_creates_lines_csv_for_away_attack(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    assert (out_dir / "2" / "lines.csv").exists()


def test_extract_line_lines_csv_has_line_columns(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    df = pd.read_csv(out_dir / "0" / "lines.csv")
    line_cols = [c for c in df.columns if c.endswith("_line")]
    assert len(line_cols) > 0


def test_extract_line_lines_csv_has_line_count_column(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    df = pd.read_csv(out_dir / "0" / "lines.csv")
    assert "line_count" in df.columns


def test_extract_line_away_defending_has_away_line_cols(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    df = pd.read_csv(out_dir / "0" / "lines.csv")
    line_cols = [c for c in df.columns if c.endswith("_line")]
    assert all(c.startswith("Away_") for c in line_cols)


def test_extract_line_home_defending_has_home_line_cols(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    df = pd.read_csv(out_dir / "2" / "lines.csv")
    line_cols = [c for c in df.columns if c.endswith("_line")]
    assert all(c.startswith("Home_") for c in line_cols)


def test_extract_line_preserves_original_columns(
    extract_line_result: tuple,
) -> None:
    _, out_dir = extract_line_result
    df = pd.read_csv(out_dir / "0" / "lines.csv")
    for col in ["frame", "Period", "Time [s]", "ball_x", "ball_y"]:
        assert col in df.columns


def test_extract_line_shows_output_path(extract_line_result: tuple) -> None:
    result, out_dir = extract_line_result
    assert str(out_dir) in result.output


# ---------------------------------------------------------------------------
# extract-line — error cases
# ---------------------------------------------------------------------------


def test_extract_line_unknown_dataset_exits_nonzero(
    pass_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: pass_output_dir)
    result = runner.invoke(
        app, ["analyze", "leak", "extract-line", "unknown/dataset", MATCH_ID]
    )
    assert result.exit_code != 0


def test_extract_line_missing_metadata_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["analyze", "leak", "extract-line", DATASET_ID, MATCH_ID]
    )
    assert result.exit_code != 0


def test_extract_line_skips_missing_frames_file(
    pass_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Remove frames_away.csv for event 0 so it is skipped
    frames_path = (
        pass_output_dir / Path(DATASET_ID) / MATCH_ID / "pass" / "0" / "frames_away.csv"
    )
    frames_path.unlink()
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: pass_output_dir)
    result = runner.invoke(
        app, ["analyze", "leak", "extract-line", DATASET_ID, MATCH_ID]
    )
    # Should still succeed (exit 0), but only 2 events processed
    assert result.exit_code == 0
    out_dir = pass_output_dir / Path(DATASET_ID) / MATCH_ID / "pass"
    assert not (out_dir / "0" / "lines.csv").exists()
    assert (out_dir / "1" / "lines.csv").exists()


# ---------------------------------------------------------------------------
# visualize-line helpers
# ---------------------------------------------------------------------------


def _mock_leak_visualizer(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    mock_anim = MagicMock()
    mock_instance = MagicMock()
    mock_instance.animate.return_value = mock_anim
    mock_cls = MagicMock(return_value=mock_instance)
    monkeypatch.setattr("turf.leak_lines_visualizer.LeakLinesVisualizer", mock_cls)
    return mock_cls, mock_anim


def _make_lines_csv(event_dir: Path, team: str, frames_df: pd.DataFrame) -> None:
    from leak.lines import analyze_lines

    lines_df = analyze_lines(frames_df, team)
    lines_df.to_csv(event_dir / "lines.csv", index=False)


@pytest.fixture()
def pass_output_dir_with_lines(pass_output_dir: Path) -> Path:
    out_dir = pass_output_dir / Path(DATASET_ID) / MATCH_ID / "pass"
    away_df = _make_away_frames_df()
    home_df = _make_home_frames_df()
    # events 0,1 → Away defends
    for idx in [0, 1]:
        _make_lines_csv(out_dir / str(idx), "Away", away_df)
    # event 2 → Home defends
    _make_lines_csv(out_dir / "2", "Home", home_df)
    return pass_output_dir


# ---------------------------------------------------------------------------
# visualize-line — happy path
# ---------------------------------------------------------------------------


@pytest.fixture()
def visualize_line_result(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path]:
    output_root = pass_output_dir_with_lines
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: output_root)
    _mock_leak_visualizer(monkeypatch)
    result = runner.invoke(
        app, ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID]
    )
    out_dir = output_root / Path(DATASET_ID) / MATCH_ID / "pass"
    return result, out_dir


def test_visualize_line_exit_code(visualize_line_result: tuple) -> None:
    result, _ = visualize_line_result
    assert result.exit_code == 0


def test_visualize_line_animate_called_per_event(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    mock_cls, _ = _mock_leak_visualizer(monkeypatch)
    runner.invoke(app, ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID])
    # default limit=10, 3 events available → animate called 3 times
    assert mock_cls.return_value.animate.call_count == 3


def test_visualize_line_anim_save_called_per_event(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    _, mock_anim = _mock_leak_visualizer(monkeypatch)
    runner.invoke(app, ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID])
    assert mock_anim.save.call_count == 3


def test_visualize_line_shows_output_path(visualize_line_result: tuple) -> None:
    result, out_dir = visualize_line_result
    assert str(out_dir) in result.output


def test_visualize_line_event_idx_filter(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    mock_cls, _ = _mock_leak_visualizer(monkeypatch)
    runner.invoke(
        app,
        ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID, "--event-idx", "0"],
    )
    assert mock_cls.return_value.animate.call_count == 1


def test_visualize_line_event_idx_all(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    mock_cls, _ = _mock_leak_visualizer(monkeypatch)
    runner.invoke(
        app,
        [
            "analyze",
            "leak",
            "visualize-line",
            DATASET_ID,
            MATCH_ID,
            "--event-idx",
            "all",
        ],
    )
    assert mock_cls.return_value.animate.call_count == 3


def test_visualize_line_skips_missing_lines_csv(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = pass_output_dir_with_lines / Path(DATASET_ID) / MATCH_ID / "pass"
    (out_dir / "0" / "lines.csv").unlink()
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    mock_cls, _ = _mock_leak_visualizer(monkeypatch)
    result = runner.invoke(
        app, ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID]
    )
    assert result.exit_code == 0
    assert mock_cls.return_value.animate.call_count == 2


# ---------------------------------------------------------------------------
# visualize-line — error cases
# ---------------------------------------------------------------------------


def test_visualize_line_unknown_dataset_exits_nonzero(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    result = runner.invoke(
        app, ["analyze", "leak", "visualize-line", "unknown/dataset", MATCH_ID]
    )
    assert result.exit_code != 0


def test_visualize_line_missing_metadata_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID]
    )
    assert result.exit_code != 0


def test_visualize_line_invalid_event_idx_exits_nonzero(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    _mock_leak_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        [
            "analyze",
            "leak",
            "visualize-line",
            DATASET_ID,
            MATCH_ID,
            "--event-idx",
            "foo",
        ],
    )
    assert result.exit_code != 0


def test_visualize_line_nonexistent_event_idx_exits_nonzero(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    _mock_leak_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        [
            "analyze",
            "leak",
            "visualize-line",
            DATASET_ID,
            MATCH_ID,
            "--event-idx",
            "99",
        ],
    )
    assert result.exit_code != 0


def test_visualize_line_zero_fps_exits_nonzero(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    _mock_leak_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        ["analyze", "leak", "visualize-line", DATASET_ID, MATCH_ID, "--fps", "0"],
    )
    assert result.exit_code != 0


def test_extract_line_min_line_gap_flag_accepted(
    pass_output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.analyze_leak.get_output_root", lambda: pass_output_dir)
    result = runner.invoke(
        app,
        [
            "analyze",
            "leak",
            "extract-line",
            DATASET_ID,
            MATCH_ID,
            "--min-line-gap",
            "5.0",
        ],
    )
    assert result.exit_code == 0


def test_visualize_line_no_smooth_lines_passes_false(
    pass_output_dir_with_lines: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "turf.analyze_leak.get_output_root", lambda: pass_output_dir_with_lines
    )
    mock_cls, _ = _mock_leak_visualizer(monkeypatch)
    runner.invoke(
        app,
        [
            "analyze",
            "leak",
            "visualize-line",
            DATASET_ID,
            MATCH_ID,
            "--no-smooth-lines",
        ],
    )
    for call in mock_cls.return_value.animate.call_args_list:
        assert call.kwargs.get("smooth_lines") is False
