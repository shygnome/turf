from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from turf.cli import app

runner = CliRunner()

EVENT_COLS = [
    "Team",
    "Type",
    "Subtype",
    "Period",
    "Start Frame",
    "Start Time [s]",
    "End Frame",
    "End Time [s]",
    "From",
    "To",
    "Start X",
    "Start Y",
    "End X",
    "End Y",
]
TRACKING_HOME_COLS = ["Period", "Time [s]", "Home_1_x", "Home_1_y", "ball_x", "ball_y"]
TRACKING_AWAY_COLS = ["Period", "Time [s]", "Away_1_x", "Away_1_y", "ball_x", "ball_y"]

N_TRACKING_ROWS = 30


def _make_events() -> pd.DataFrame:
    """Three pass events and one other, with realistic frame numbers."""
    return pd.DataFrame(
        {
            "Team": ["Home", "Home", "Away", "Home"],
            "Type": ["pass", "other", "pass", "pass"],
            "Subtype": ["success", None, "fail", "success"],
            "Period": [1, 1, 1, 1],
            "Start Frame": [2, 5, 10, 20],
            "Start Time [s]": [2.0, 5.0, 10.0, 20.0],
            "End Frame": [4, 9, 14, 24],
            "End Time [s]": [4.0, 9.0, 14.0, 24.0],
            "From": ["Alice", "Bob", "Charlie", "Alice"],
            "To": ["Bob", None, "Dave", "Charlie"],
            "Start X": [0.1, -5.0, 10.0, 3.0],
            "Start Y": [0.2, -3.0, 5.0, 1.0],
            "End X": [1.0, -4.0, 11.0, 4.0],
            "End Y": [1.2, -2.5, 5.5, 2.0],
        }
    )


def _make_tracking(home: bool) -> pd.DataFrame:
    prefix = "Home" if home else "Away"
    sign = 1.0 if home else -1.0
    n = N_TRACKING_ROWS
    return pd.DataFrame(
        {
            "Period": [1] * n,
            "Time [s]": [float(i) for i in range(n)],
            f"{prefix}_1_x": [sign * float(i) for i in range(n)],
            f"{prefix}_1_y": [sign * float(i) * 0.5 for i in range(n)],
            "ball_x": [float(i) * 0.1 for i in range(n)],
            "ball_y": [float(i) * 0.1 for i in range(n)],
        }
    )


@pytest.fixture()
def preprocessed_root(tmp_path: Path) -> Path:
    dataset_id = "pff/fifa-wc-2022"
    base = tmp_path / "preprocessed" / Path(dataset_id)
    match_id = "10502"

    events = _make_events()
    home_t = _make_tracking(home=True)
    away_t = _make_tracking(home=False)

    for subdir, filename, df in [
        ("event", f"event_data_{match_id}.csv", events),
        ("home_tracking", f"home_tracking_{match_id}.csv", home_t),
        ("away_tracking", f"away_tracking_{match_id}.csv", away_t),
    ]:
        d = base / subdir
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / filename, index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# turf event ls — happy path
# ---------------------------------------------------------------------------


def test_event_ls_exit_code(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "10502"])
    assert result.exit_code == 0


def test_event_ls_shows_pass_label(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "10502"])
    assert "pass" in result.output


def test_event_ls_shows_other_label(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "10502"])
    assert "other" in result.output


def test_event_ls_shows_event_counts(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "10502"])
    # 3 pass events
    assert "3" in result.output


def test_event_ls_shows_match_id(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "10502"])
    assert "10502" in result.output


# ---------------------------------------------------------------------------
# turf event ls — error cases
# ---------------------------------------------------------------------------


def test_event_ls_unknown_dataset_exits_nonzero(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "unknown/dataset", "10502"])
    assert result.exit_code != 0


def test_event_ls_unknown_match_exits_nonzero(
    preprocessed_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    result = runner.invoke(app, ["event", "ls", "pff/fifa-wc-2022", "99999"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# turf event extract — happy path
# ---------------------------------------------------------------------------


@pytest.fixture()
def extract_result(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path]:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "extract", "pff/fifa-wc-2022", "10502", "pass"]
    )
    out_dir = output_root / "pff" / "fifa-wc-2022" / "10502" / "pass"
    return result, out_dir


def test_event_extract_exit_code(extract_result: tuple) -> None:
    result, _ = extract_result
    assert result.exit_code == 0


def test_event_extract_creates_output_dir(extract_result: tuple) -> None:
    _, out_dir = extract_result
    assert out_dir.exists()


def test_event_extract_creates_metadata_csv(extract_result: tuple) -> None:
    _, out_dir = extract_result
    assert (out_dir / "metadata.csv").exists()


def test_event_extract_metadata_row_count(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "metadata.csv")
    assert len(df) == 3  # three pass events


def test_event_extract_metadata_columns(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "metadata.csv")
    expected = {
        "event_idx",
        "start_frame",
        "end_frame",
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "inferred_end_x",
        "inferred_end_y",
        "inferred_end_time",
        "from_player",
        "to_player",
        "team",
        "subtype",
        "period",
    }
    assert expected.issubset(set(df.columns))


def test_event_extract_creates_home_frame_csvs(extract_result: tuple) -> None:
    _, out_dir = extract_result
    for i in range(3):
        assert (out_dir / f"frames_home_{i}.csv").exists()


def test_event_extract_creates_away_frame_csvs(extract_result: tuple) -> None:
    _, out_dir = extract_result
    for i in range(3):
        assert (out_dir / f"frames_away_{i}.csv").exists()


def test_event_extract_frames_have_frame_column(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_home_0.csv")
    assert "frame" in df.columns


def test_event_extract_first_home_frames_row_count(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_home_0.csv")
    # event 0: inferred end = next event (other) at t=5.0 → rows 2,3,4,5 = 4
    assert len(df) == 4


def test_event_extract_first_away_frames_row_count(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_away_0.csv")
    assert len(df) == 4


def test_event_extract_frame_column_values(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_home_0.csv")
    # event 0: inferred end t=5.0 → frame column [2, 3, 4, 5]
    assert df["frame"].tolist() == [2, 3, 4, 5]


def test_event_extract_shows_output_path(extract_result: tuple) -> None:
    result, out_dir = extract_result
    assert str(out_dir) in result.output


def test_event_extract_no_infer_endpoints_flag(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    runner.invoke(
        app,
        [
            "event",
            "extract",
            "pff/fifa-wc-2022",
            "10502",
            "pass",
            "--no-infer-endpoints",
        ],
    )
    out_dir = output_root / "pff" / "fifa-wc-2022" / "10502" / "pass"
    df = pd.read_csv(out_dir / "frames_home_0.csv")
    # without inference, event 0 uses raw end t=4.0 → rows 2,3,4 = 3
    assert df["frame"].tolist() == [2, 3, 4]


# ---------------------------------------------------------------------------
# turf event extract — case insensitive
# ---------------------------------------------------------------------------


def test_event_extract_case_insensitive(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "extract", "pff/fifa-wc-2022", "10502", "PASS"]
    )
    assert result.exit_code == 0
    out_dir = output_root / "pff" / "fifa-wc-2022" / "10502" / "pass"
    assert (out_dir / "metadata.csv").exists()


# ---------------------------------------------------------------------------
# turf event extract — error cases
# ---------------------------------------------------------------------------


def test_event_extract_unknown_dataset_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "extract", "unknown/dataset", "10502", "pass"]
    )
    assert result.exit_code != 0


def test_event_extract_unknown_match_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "extract", "pff/fifa-wc-2022", "99999", "pass"]
    )
    assert result.exit_code != 0


def test_event_extract_unknown_label_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "extract", "pff/fifa-wc-2022", "10502", "tackle"]
    )
    assert result.exit_code != 0


def test_event_extract_dotdot_label_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(app, ["event", "extract", "pff/fifa-wc-2022", "10502", ".."])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# turf event visualize — helpers
# ---------------------------------------------------------------------------


def _mock_visualizer(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object, object]:
    """Patch EventVisualizer so no matplotlib rendering happens.

    Returns (mock_class, mock_fig, mock_anim).
    """
    from unittest.mock import MagicMock

    mock_fig = MagicMock()
    mock_anim = MagicMock()
    mock_instance = MagicMock()
    mock_instance.freeze_frame.return_value = mock_fig
    mock_instance.animate.return_value = mock_anim
    mock_cls = MagicMock(return_value=mock_instance)
    monkeypatch.setattr("turf.event_visualizer.EventVisualizer", mock_cls)
    return mock_cls, mock_fig, mock_anim


@pytest.fixture()
def visualize_result(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, Path]:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"]
    )
    out_dir = output_root / "pff" / "fifa-wc-2022" / "10502" / "pass"
    return result, out_dir


# ---------------------------------------------------------------------------
# turf event visualize — happy path
# ---------------------------------------------------------------------------


def test_event_visualize_exit_code(visualize_result: tuple) -> None:
    result, _ = visualize_result
    assert result.exit_code == 0


def test_event_visualize_creates_output_dir(visualize_result: tuple) -> None:
    _, out_dir = visualize_result
    assert out_dir.exists()


def test_event_visualize_freeze_called_per_event(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    # 3 pass events → freeze_frame called 3 times
    assert mock_cls.return_value.freeze_frame.call_count == 3


def test_event_visualize_animate_called_per_event(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    assert mock_cls.return_value.animate.call_count == 3


def test_event_visualize_savefig_called_per_event(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _, mock_fig, _ = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    assert mock_fig.savefig.call_count == 3


def test_event_visualize_anim_save_called_per_event(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _, _, mock_anim = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    assert mock_anim.save.call_count == 3


def test_event_visualize_shows_output_path(visualize_result: tuple) -> None:
    result, out_dir = visualize_result
    assert str(out_dir) in result.output


def test_event_visualize_case_insensitive(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "PASS"]
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# turf event visualize — --event-idx filter
# ---------------------------------------------------------------------------


def test_event_visualize_event_idx_filter(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(
        app,
        ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass", "--event-idx", "0"],
    )
    # only 1 event → freeze_frame called once
    assert mock_cls.return_value.freeze_frame.call_count == 1


def test_event_visualize_event_idx_all_uses_all_events(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(
        app,
        [
            "event",
            "visualize",
            "pff/fifa-wc-2022",
            "10502",
            "pass",
            "--event-idx",
            "all",
        ],
    )
    # fixture has 3 pass events → all 3 visualized
    assert mock_cls.return_value.freeze_frame.call_count == 3


def test_event_visualize_invalid_event_idx_str_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        [
            "event",
            "visualize",
            "pff/fifa-wc-2022",
            "10502",
            "pass",
            "--event-idx",
            "foo",
        ],
    )
    assert result.exit_code != 0


def test_event_visualize_invalid_event_idx_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        [
            "event",
            "visualize",
            "pff/fifa-wc-2022",
            "10502",
            "pass",
            "--event-idx",
            "99",
        ],
    )
    assert result.exit_code != 0


def test_event_visualize_default_caps_at_10(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import MagicMock

    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    # Return 15 mock clips from the extractor
    mock_clips = [MagicMock(event_idx=i) for i in range(15)]
    mock_extractor_cls = MagicMock()
    mock_extractor_cls.return_value.extract.return_value = mock_clips
    monkeypatch.setattr("turf.event.EventExtractor", mock_extractor_cls)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    assert mock_cls.return_value.freeze_frame.call_count == 10


# ---------------------------------------------------------------------------
# turf event visualize — --smooth flag
# ---------------------------------------------------------------------------


def test_event_visualize_smooth_flag_passed_to_visualizer(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(
        app,
        ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass", "--smooth"],
    )
    # verify smooth=True was forwarded to freeze_frame
    call_kwargs = mock_cls.return_value.freeze_frame.call_args_list[0]
    assert call_kwargs.kwargs.get("smooth") is True


def test_event_visualize_no_smooth_by_default(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    mock_cls, _, _ = _mock_visualizer(monkeypatch)
    runner.invoke(app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass"])
    call_kwargs = mock_cls.return_value.freeze_frame.call_args_list[0]
    assert call_kwargs.kwargs.get("smooth") is False


# ---------------------------------------------------------------------------
# turf event visualize — error cases
# ---------------------------------------------------------------------------


def test_event_visualize_unknown_dataset_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "visualize", "unknown/dataset", "10502", "pass"]
    )
    assert result.exit_code != 0


def test_event_visualize_unknown_match_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "visualize", "pff/fifa-wc-2022", "99999", "pass"]
    )
    assert result.exit_code != 0


def test_event_visualize_unknown_label_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app, ["event", "visualize", "pff/fifa-wc-2022", "10502", "tackle"]
    )
    assert result.exit_code != 0


def test_event_visualize_dotdot_label_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    result = runner.invoke(
        app, ["event", "visualize", "pff/fifa-wc-2022", "10502", ".."]
    )
    assert result.exit_code != 0


def test_event_visualize_zero_fps_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass", "--fps", "0"],
    )
    assert result.exit_code != 0


def test_event_visualize_negative_fps_exits_nonzero(
    preprocessed_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "output"
    monkeypatch.setattr("turf.event.get_root", lambda: preprocessed_root)
    monkeypatch.setattr("turf.event.get_output_root", lambda: output_root)
    _mock_visualizer(monkeypatch)
    result = runner.invoke(
        app,
        ["event", "visualize", "pff/fifa-wc-2022", "10502", "pass", "--fps", "-5"],
    )
    assert result.exit_code != 0
