from __future__ import annotations

import warnings
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
_START_TIME_COL = "Start Time [s]"
_INFER_LABELS = frozenset({"pass", "cross"})


class EventExtractor:
    def list_labels(self, events: pd.DataFrame) -> list[str]:
        return sorted(events["Type"].str.lower().dropna().unique().tolist())

    def _compute_inferred_endpoints(
        self,
        matching: pd.DataFrame,
        all_events: pd.DataFrame,
    ) -> dict[int, dict[str, object]]:
        """Return inferred end values for each event in *matching*.

        For each event at position *event_idx*, finds the first event in
        *all_events* within the same period that starts strictly later.
        Events with no successor are omitted from the result (callers keep
        original values for those).
        """
        result: dict[int, dict[str, object]] = {}
        for event_idx, (_, row) in enumerate(matching.iterrows()):
            period = int(row[_PERIOD_COL])
            start_time = float(row[_START_TIME_COL])
            mask = (all_events[_PERIOD_COL] == period) & (
                all_events[_START_TIME_COL] > start_time
            )
            next_events = all_events[mask]
            if next_events.empty:
                continue
            nxt = next_events.iloc[0]
            result[event_idx] = {
                "end_time": float(nxt[_START_TIME_COL]),
                "end_x": float(nxt["Start X"]),
                "end_y": float(nxt["Start Y"]),
            }
        return result

    def extract(
        self,
        match_data: MatchData,
        label: str,
        infer_endpoints: bool = True,
    ) -> list[EventClip]:
        mask = match_data.events["Type"].str.lower() == label.lower()
        matching = match_data.events[mask].reset_index(drop=True)

        covered_periods = set(
            match_data.home_tracking[_PERIOD_COL].unique()
        ) & set(match_data.away_tracking[_PERIOD_COL].unique())
        missing_periods = sorted(set(matching[_PERIOD_COL].unique()) - covered_periods)
        if missing_periods:
            n_skipped = int(matching[_PERIOD_COL].isin(missing_periods).sum())
            warnings.warn(
                f"Periods {missing_periods} have events but no tracking data — "
                f"{n_skipped} event(s) skipped.",
                UserWarning,
                stacklevel=2,
            )
            matching = matching[
                matching[_PERIOD_COL].isin(covered_periods)
            ].reset_index(drop=True)

        inferred: dict[int, dict[str, object]] = {}
        if infer_endpoints and label.lower() in _INFER_LABELS:
            inferred = self._compute_inferred_endpoints(matching, match_data.events)

        clips: list[EventClip] = []
        for event_idx, (_, row) in enumerate(matching.iterrows()):
            start_time = float(row[_START_TIME_COL])
            raw_end_time = float(row["End Time [s]"])
            inf = inferred.get(event_idx)
            # Use inferred end time for clip slicing when available; raw otherwise
            snap_end_time = float(inf["end_time"]) if inf else raw_end_time  # type: ignore[arg-type]
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
            end_label = int((home_period[_TIME_COL] - snap_end_time).abs().idxmin())
            if end_label < start_label:
                end_label = start_label

            home_frames = home_period.loc[start_label:end_label].copy()
            away_frames = away_period.loc[start_label:end_label].copy()

            if len(home_frames) == 0 or len(away_frames) == 0:
                raise ValueError(
                    f"No tracking frames found for event {event_idx} "
                    f"in time range [{start_time}, {snap_end_time}]."
                )

            start = int(home_frames.index[0])
            end = int(home_frames.index[-1])
            meta: dict[str, object] = {
                "event_idx": event_idx,
                "start_frame": start,
                "end_frame": end,
                "start_time": start_time,
                "end_time": raw_end_time,
                "start_x": row["Start X"],
                "start_y": row["Start Y"],
                "end_x": row["End X"],
                "end_y": row["End Y"],
                "inferred_end_x": inf["end_x"] if inf else None,
                "inferred_end_y": inf["end_y"] if inf else None,
                "inferred_end_time": inf["end_time"] if inf else None,
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
