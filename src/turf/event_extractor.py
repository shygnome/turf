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


class EventExtractor:
    def list_labels(self, events: pd.DataFrame) -> list[str]:
        return sorted(events["Type"].str.lower().dropna().unique().tolist())

    def extract(self, match_data: MatchData, label: str) -> list[EventClip]:
        mask = match_data.events["Type"].str.lower() == label.lower()
        matching = match_data.events[mask].reset_index(drop=True)
        clips: list[EventClip] = []
        for event_idx, (_, row) in enumerate(matching.iterrows()):
            start = int(row["Start Frame"])
            end = int(row["End Frame"])
            meta: dict[str, object] = {
                "event_idx": event_idx,
                "start_frame": start,
                "end_frame": end,
                "start_x": row["Start X"],
                "start_y": row["Start Y"],
                "end_x": row["End X"],
                "end_y": row["End Y"],
                "from_player": row["From"],
                "to_player": row["To"],
                "team": row["Team"],
                "subtype": row["Subtype"],
                "period": int(row["Period"]),
            }
            clips.append(
                EventClip(
                    event_idx=event_idx,
                    start_frame=start,
                    end_frame=end,
                    metadata=meta,
                    home_frames=match_data.home_tracking.iloc[start : end + 1].copy(),
                    away_frames=match_data.away_tracking.iloc[start : end + 1].copy(),
                )
            )
        return clips
