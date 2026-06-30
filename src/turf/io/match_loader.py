from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

# Standard kickoff time in seconds for each period.
# Period 1 is always 0 and never adjusted.
# PFF tracking data continues time across period boundaries rather than resetting,
# so periods 2-4 need to be shifted back to their standard kickoff times.
_PERIOD_KICKOFF_SECONDS: dict[int, float] = {
    2: 2700.0,
    3: 5400.0,
    4: 8100.0,
}


def normalize_period_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize tracking 'Time [s]' so each period starts at its standard kickoff time.

    PFF tracking data continues time from the end of the previous period instead of
    resetting. This shifts Period 2 to start at 2700s, Period 3 at 5400s,
    Period 4 at 8100s. Period 1 and any period beyond 4 are left unchanged.
    """
    result = df.copy()
    for period_int, expected in _PERIOD_KICKOFF_SECONDS.items():
        mask = result["Period"] == period_int
        if not mask.any():
            continue
        actual_start = float(result.loc[mask, "Time [s]"].iloc[0])
        offset = actual_start - expected
        if offset:
            result.loc[mask, "Time [s]"] = result.loc[mask, "Time [s]"] - offset
    return result


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
            home_tracking=normalize_period_times(pd.read_csv(home_file)),
            away_tracking=normalize_period_times(pd.read_csv(away_file)),
        )
