from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from turf.cli import app
from turf.dataset import CATALOG, DatasetEntry
from turf.prepare import _extract_metadata

runner = CliRunner()

METADATA_COLUMNS = [
    "match_id",
    "home_team_id",
    "home_team_name",
    "home_team_short_name",
    "away_team_id",
    "away_team_name",
    "away_team_short_name",
    "date",
    "stadium",
]


def _make_metadata_json(path: Path, match_id: int = 10502) -> None:
    data = [
        {
            "id": match_id,
            "homeTeam": {"id": "1", "name": "Home FC", "shortName": "HOM"},
            "awayTeam": {"id": "2", "name": "Away FC", "shortName": "AWY"},
            "date": "2022-12-03T15:00:00",
            "stadium": {"id": "99", "name": "Test Stadium"},
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")


def _setup_dataset(tmp_path: Path, entry: DatasetEntry) -> None:
    dataset_path = tmp_path / entry.path
    dataset_path.mkdir(parents=True)
    assert entry.prepare_spec is not None
    for subdir in entry.prepare_spec.input_paths.values():
        (dataset_path / subdir).mkdir()


def test_prepare_exit_code_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.prepare._run_preprocessing", lambda *a, **kw: None)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code == 0


def test_prepare_calls_preprocessing_with_correct_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    calls: list[tuple[object, dict[str, str], str]] = []
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)

    def _capture(spec: object, input_kwargs: dict[str, str], out_path: str) -> None:
        calls.append((spec, input_kwargs, out_path))

    monkeypatch.setattr("turf.prepare._run_preprocessing", _capture)
    runner.invoke(app, ["dataset", "prepare", entry.id])
    assert len(calls) == 1
    spec, input_kwargs, _ = calls[0]
    assert spec is entry.prepare_spec
    assert entry.prepare_spec is not None
    dataset_path = tmp_path / entry.path
    for kwarg, subdir in entry.prepare_spec.input_paths.items():
        assert input_kwargs[kwarg] == str(dataset_path / subdir)


def test_prepare_does_not_create_output_dir_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)

    def _fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("preprocessing failed")

    monkeypatch.setattr("turf.prepare._run_preprocessing", _fail)
    runner.invoke(app, ["dataset", "prepare", entry.id])
    assert not (tmp_path / "preprocessed" / Path(entry.id)).exists()


def test_prepare_creates_output_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.prepare._run_preprocessing", lambda *a, **kw: None)
    runner.invoke(app, ["dataset", "prepare", entry.id])
    assert (tmp_path / "preprocessed" / Path(entry.id)).exists()


def test_prepare_errors_on_unknown_dataset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", "unknown/dataset"])
    assert result.exit_code != 0


def test_prepare_errors_when_dataset_not_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code != 0


def test_prepare_errors_when_no_prepare_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is None)
    (tmp_path / entry.path).mkdir(parents=True)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# _extract_metadata
# ---------------------------------------------------------------------------


def test_extract_metadata_creates_csv(tmp_path: Path) -> None:
    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _make_metadata_json(meta_dir / "10502.json")
    _extract_metadata(meta_dir, tmp_path)
    assert (tmp_path / "metadata.csv").exists()


def test_extract_metadata_columns(tmp_path: Path) -> None:
    import pandas as pd

    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _make_metadata_json(meta_dir / "10502.json")
    _extract_metadata(meta_dir, tmp_path)
    df = pd.read_csv(tmp_path / "metadata.csv")
    assert list(df.columns) == METADATA_COLUMNS


def test_extract_metadata_row_count(tmp_path: Path) -> None:
    import pandas as pd

    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _make_metadata_json(meta_dir / "10502.json")
    _make_metadata_json(meta_dir / "10503.json", match_id=10503)
    _extract_metadata(meta_dir, tmp_path)
    df = pd.read_csv(tmp_path / "metadata.csv")
    assert len(df) == 2


def test_extract_metadata_stadium_is_name_string(tmp_path: Path) -> None:
    import pandas as pd

    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _make_metadata_json(meta_dir / "10502.json")
    _extract_metadata(meta_dir, tmp_path)
    df = pd.read_csv(tmp_path / "metadata.csv")
    assert df.loc[0, "stadium"] == "Test Stadium"


def test_extract_metadata_correct_values(tmp_path: Path) -> None:
    import pandas as pd

    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _make_metadata_json(meta_dir / "10502.json", match_id=10502)
    _extract_metadata(meta_dir, tmp_path)
    df = pd.read_csv(tmp_path / "metadata.csv")
    row = df.iloc[0]
    assert row["match_id"] == 10502
    assert row["home_team_name"] == "Home FC"
    assert row["home_team_short_name"] == "HOM"
    assert row["away_team_name"] == "Away FC"
    assert row["date"] == "2022-12-03T15:00:00"


def test_prepare_creates_metadata_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec and e.prepare_spec.metadata_path)
    _setup_dataset(tmp_path, entry)
    assert entry.prepare_spec is not None
    meta_dir = tmp_path / entry.path / entry.prepare_spec.metadata_path
    meta_dir.mkdir(parents=True, exist_ok=True)
    _make_metadata_json(meta_dir / "10502.json")
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.prepare._run_preprocessing", lambda *a, **kw: None)
    runner.invoke(app, ["dataset", "prepare", entry.id])
    out = tmp_path / "preprocessed" / Path(entry.id) / "metadata.csv"
    assert out.exists()


def test_prepare_skips_metadata_when_dir_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    monkeypatch.setattr("turf.prepare.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.prepare._run_preprocessing", lambda *a, **kw: None)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code == 0


def test_extract_metadata_empty_dir_does_not_create_csv(tmp_path: Path) -> None:
    meta_dir = tmp_path / "Metadata"
    meta_dir.mkdir()
    _extract_metadata(meta_dir, tmp_path)
    assert not (tmp_path / "metadata.csv").exists()
