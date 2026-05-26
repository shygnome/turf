from pathlib import Path

import pytest
from typer.testing import CliRunner

from turf.cli import app
from turf.dataset import CATALOG, DatasetEntry, _config_path, get_root

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


def test_ls_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "ls"])
    assert result.exit_code == 0


def test_ls_shows_all_catalog_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
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


def test_get_root_falls_back_on_corrupt_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("not valid toml ][", encoding="utf-8")
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    assert get_root() == Path("data")


def test_get_root_falls_back_when_dataset_root_not_a_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text("dataset_root = 42\n", encoding="utf-8")
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    assert get_root() == Path("data")


def test_get_root_falls_back_on_empty_dataset_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text('dataset_root = ""\n', encoding="utf-8")
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    assert get_root() == Path("data")


def test_config_path_falls_back_when_home_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoHomePath(type(Path())):
        def expanduser(self) -> Path:
            raise RuntimeError("no home directory")

    monkeypatch.setattr("turf.dataset.CONFIG_PATH", _NoHomePath("~/.turf/config.toml"))
    result = _config_path()
    assert result == Path("~/.turf/config.toml")


def test_set_root_expands_tilde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("turf.dataset.CONFIG_PATH", config_file)
    result = runner.invoke(app, ["dataset", "set-root", "~/data"])
    assert result.exit_code == 0
    root = get_root()
    assert "~" not in str(root)
    assert root.is_absolute()


# --- prepare command ---


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
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.dataset._run_preprocessing", lambda *a, **kw: None)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code == 0


def test_prepare_calls_preprocessing_with_correct_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    _setup_dataset(tmp_path, entry)
    calls: list[tuple[object, dict[str, str], str]] = []
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)

    def _capture(spec: object, input_kwargs: dict[str, str], out_path: str) -> None:
        calls.append((spec, input_kwargs, out_path))

    monkeypatch.setattr("turf.dataset._run_preprocessing", _capture)
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
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    monkeypatch.setattr("turf.dataset._run_preprocessing", lambda *a, **kw: None)
    runner.invoke(app, ["dataset", "prepare", entry.id])
    assert (tmp_path / "preprocessed" / Path(entry.id)).exists()


def test_prepare_errors_on_unknown_dataset_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", "unknown/dataset"])
    assert result.exit_code != 0


def test_prepare_errors_when_dataset_not_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is not None)
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code != 0


def test_prepare_errors_when_no_prepare_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = next(e for e in CATALOG if e.prepare_spec is None)
    (tmp_path / entry.path).mkdir(parents=True)
    monkeypatch.setattr("turf.dataset.get_root", lambda: tmp_path)
    result = runner.invoke(app, ["dataset", "prepare", entry.id])
    assert result.exit_code != 0
