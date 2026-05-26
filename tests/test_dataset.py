from pathlib import Path

import pytest
from typer.testing import CliRunner

from turf.cli import app
from turf.dataset import CATALOG, DatasetEntry, get_root

runner = CliRunner()


def test_catalog_is_not_empty() -> None:
    assert len(CATALOG) > 0


def test_catalog_entries_have_required_fields() -> None:
    for entry in CATALOG:
        assert isinstance(entry, DatasetEntry)
        assert entry.id
        assert entry.name
        assert entry.provider
        assert entry.path
        assert entry.description


def test_ls_exit_code() -> None:
    result = runner.invoke(app, ["dataset", "ls"])
    assert result.exit_code == 0


def test_ls_shows_all_catalog_ids() -> None:
    result = runner.invoke(app, ["dataset", "ls"])
    for entry in CATALOG:
        assert entry.id in result.output


def test_ls_shows_present_when_path_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = CATALOG[0]
    (tmp_path / entry.path).mkdir(parents=True)
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "ls"])
    assert result.exit_code == 0
    assert "[+]" in result.output


def test_ls_marks_absent_when_path_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "ls"])
    assert result.exit_code == 0
    assert "[ ]" in result.output


def test_ls_shows_dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "ls"])
    assert str(tmp_path) in result.output


def test_set_root_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    result = runner.invoke(app, ["dataset", "set-root", str(tmp_path)])
    assert result.exit_code == 0


def test_set_root_writes_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    runner.invoke(app, ["dataset", "set-root", str(tmp_path)])
    assert config_file.exists()


def test_set_root_creates_parent_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "nested" / "dir" / "config.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    result = runner.invoke(app, ["dataset", "set-root", str(tmp_path)])
    assert result.exit_code == 0
    assert config_file.exists()


def test_get_root_reads_written_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    new_root = tmp_path / "datasets"
    runner.invoke(app, ["dataset", "set-root", str(new_root)])
    assert get_root() == new_root


def test_get_root_defaults_when_no_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "nonexistent.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    assert get_root() == Path("data")
