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
            "Start Time [s]": [0.06, 0.15, 0.30, 0.60],
            "End Frame": [4, 9, 14, 24],
            "End Time [s]": [0.12, 0.28, 0.44, 0.76],
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
            "Time [s]": [i * 0.04 for i in range(n)],
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
    # event 0: start_frame=2, end_frame=4 → 3 rows
    assert len(df) == 3


def test_event_extract_first_away_frames_row_count(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_away_0.csv")
    assert len(df) == 3


def test_event_extract_frame_column_values(extract_result: tuple) -> None:
    _, out_dir = extract_result
    df = pd.read_csv(out_dir / "frames_home_0.csv")
    # event 0: start_frame=2 → frame column should be [2, 3, 4]
    assert df["frame"].tolist() == [2, 3, 4]


def test_event_extract_shows_output_path(extract_result: tuple) -> None:
    result, out_dir = extract_result
    assert str(out_dir) in result.output


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
    result = runner.invoke(
        app, ["event", "extract", "pff/fifa-wc-2022", "10502", ".."]
    )
    assert result.exit_code != 0
