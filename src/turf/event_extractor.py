from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from turf.match_loader import MatchData


@dataclass
class EventClip:
    event_idx: int
    start_frame: int
    end_frame: int
    metadata: dict[str, object]
    home_frames: pd.DataFrame
    away_frames: pd.DataFrame


_TIME_COL = "Time [s]"
_PERIOD_COL = "Period"


class EventExtractor:
    def list_labels(self, events: pd.DataFrame) -> list[str]:
        return sorted(events["Type"].str.lower().dropna().unique().tolist())

    def extract(self, match_data: MatchData, label: str) -> list[EventClip]:
        mask = match_data.events["Type"].str.lower() == label.lower()
        matching = match_data.events[mask].reset_index(drop=True)
        clips: list[EventClip] = []
        for event_idx, (_, row) in enumerate(matching.iterrows()):
            start_time = float(row["Start Time [s]"])
            end_time = float(row["End Time [s]"])
            period = int(row[_PERIOD_COL])

            # Filter to matching period so we don't snap across half-time boundary
            home_period = match_data.home_tracking[
                match_data.home_tracking[_PERIOD_COL] == period
            ]
            away_period = match_data.away_tracking[
                match_data.away_tracking[_PERIOD_COL] == period
            ]
            if len(home_period) == 0 or len(away_period) == 0:
                raise ValueError(
                    f"No tracking frames found for event {event_idx} "
                    f"in period {period}."
                )

            # Snap to the nearest tracking frame for start and end times
            start_label = int((home_period[_TIME_COL] - start_time).abs().idxmin())
            end_label = int((home_period[_TIME_COL] - end_time).abs().idxmin())

            home_frames = match_data.home_tracking.loc[start_label:end_label].copy()
            away_frames = match_data.away_tracking.loc[start_label:end_label].copy()

            if len(home_frames) == 0 or len(away_frames) == 0:
                raise ValueError(
                    f"No tracking frames found for event {event_idx} "
                    f"in time range [{start_time}, {end_time}]."
                )

            start = int(home_frames.index[0])
            end = int(home_frames.index[-1])
            meta: dict[str, object] = {
                "event_idx": event_idx,
                "start_frame": start,
                "end_frame": end,
                "start_time": start_time,
                "end_time": end_time,
                "start_x": row["Start X"],
                "start_y": row["Start Y"],
                "end_x": row["End X"],
                "end_y": row["End Y"],
                "from_player": row["From"],
                "to_player": row["To"],
                "team": row["Team"],
                "subtype": row["Subtype"],
                "period": period,
            }
            clips.append(
                EventClip(
                    event_idx=event_idx,
                    start_frame=start,
                    end_frame=end,
                    metadata=meta,
                    home_frames=home_frames,
                    away_frames=away_frames,
                )
            )
        return clips
