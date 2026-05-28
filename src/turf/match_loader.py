from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


@dataclass
class MatchData:
    match_id: str
    events: pd.DataFrame
    home_tracking: pd.DataFrame
    away_tracking: pd.DataFrame


class MatchLoader:
    def __init__(self, dataset_id: str, root: Path) -> None:
        self.dataset_id = dataset_id
        self.preprocessed_path: Path = root / "preprocessed" / Path(dataset_id)

    def load(self, match_id: str) -> MatchData:
        import pandas as pd

        if not self.preprocessed_path.exists():
            raise FileNotFoundError(
                f"Preprocessed data not found at {self.preprocessed_path}. "
                f"Run `turf dataset prepare {self.dataset_id}` first."
            )

        event_file = self.preprocessed_path / "event" / f"event_data_{match_id}.csv"
        home_file = (
            self.preprocessed_path / "home_tracking" / f"home_tracking_{match_id}.csv"
        )
        away_file = (
            self.preprocessed_path / "away_tracking" / f"away_tracking_{match_id}.csv"
        )

        for path in (event_file, home_file, away_file):
            if not path.exists():
                raise FileNotFoundError(f"Match file not found: {path}")

        return MatchData(
            match_id=match_id,
            events=pd.read_csv(event_file),
            home_tracking=pd.read_csv(home_file),
            away_tracking=pd.read_csv(away_file),
        )
