# TURF — Claude Code Guide

Soccer/football research toolkit. CLI-first Python project managed with `uv`.

## Project layout

```
src/turf/
  __init__.py     # version string only
  cli.py          # typer app — all subcommands live here (or imported here)
data/             # gitignored — raw research data
output/           # gitignored — generated artefacts
```

## Running commands

Always prefix with `uv run` — do not assume the venv is activated.

```bash
uv run turf --help
uv run pytest
uv run ruff check .
uv run ruff format .
uv run mypy src
```

## Development workflow — red/green TDD

1. Write a failing test first (`pytest` marks it red).
2. Write the minimum code to make it pass (green).
3. Refactor if needed, keeping tests green.
4. Never write implementation code without a corresponding test driving it.

Tests live in `tests/` mirroring `src/turf/`:

```
tests/
  test_cli.py
  test_<module>.py
```

Run tests:

```bash
uv run pytest
uv run pytest -x          # stop on first failure
uv run pytest -k "name"   # filter by name
```

## Code style — ruff

Linting and formatting are handled by `ruff`.

```bash
uv run ruff check .        # lint
uv run ruff check . --fix  # auto-fix
uv run ruff format .       # format
```

`ruff` must pass with zero errors before committing.

## Type hints — mypy

All new code must be fully type-annotated. Run:

```bash
uv run mypy src
```

Exceptions may be granted per-case with an inline `# type: ignore[<code>]` comment that explains why.

## Commit message format — Conventional Commits

```
<type>(<optional scope>): <short description>
```

| Type       | When to use                                      |
|------------|--------------------------------------------------|
| `feat`     | New feature or capability                        |
| `fix`      | Bug fix                                          |
| `test`     | Adding or updating tests                         |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `docs`     | Documentation only                               |
| `chore`    | Tooling, deps, config, CI — no production code   |
| `style`    | Formatting, whitespace — no logic change         |
| `perf`     | Performance improvement                          |
| `revert`   | Reverts a previous commit                        |

Examples:

```
feat(cli): add fixtures subcommand
fix(parser): handle missing date field in match data
test(fixtures): add red/green tests for fixture fetch
refactor(cli): extract data-fetch logic into loader module
docs: update README with turf install steps
chore: add ruff and mypy to dev dependencies
```

Use imperative mood in the description ("add", not "added" or "adds").

Commit messages are **title-only** — no body, no bullet points, no Co-Authored-By. Detailed descriptions belong in the pull request body, not the commit.

## Branch naming

```
<type>/<ref>/short-description
<type>/no-ref/short-description   # when there is no issue/ticket reference
```

| Prefix     | When to use                         |
|------------|-------------------------------------|
| `feature/` | New feature work                    |
| `bugfix/`  | Non-urgent bug fix                  |
| `hotfix/`  | Urgent production fix               |
| `refactor/`| Refactoring without behaviour change|
| `chore/`   | Tooling, deps, config               |
| `docs/`    | Documentation only                  |
| `test/`    | Test-only changes                   |

Examples:

```
feature/12/add-fixtures-subcommand
bugfix/34/fix-date-parsing
chore/no-ref/add-ruff-config
docs/no-ref/update-readme
```

## Adding a new CLI subcommand

1. Create `src/turf/<module>.py` with the logic.
2. Add the command to `src/turf/cli.py` via `@app.command()` (or an `app.add_typer()` sub-app).
3. Write tests in `tests/test_<module>.py` before implementing.
4. Run `uv run turf --help` to confirm the command appears.
