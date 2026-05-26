from pathlib import Path

import pytest
from typer.testing import CliRunner

from turf.cli import app
from turf.dataset import CATALOG, DatasetEntry

runner = CliRunner()


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
    spec, input_kwargs, out_path = calls[0]
    assert spec is entry.prepare_spec
    expected_out = str(tmp_path / "preprocessed" / Path(entry.id))
    assert out_path == expected_out


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
