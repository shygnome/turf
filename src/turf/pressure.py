"""Pressure lookup from raw PFF event JSON files.

Reads the vendor PFF JSON format (one flat array of events per match) and
builds a lookup from (period, game_clock_s, team) → under_pressure bool.

Pressure is recorded on the possessionEvents object as pressureType:
  "N" = no pressure, "P" = pressure, "L" = loose, "A" = aggressive.
Binary: any value other than "N" is considered under pressure.

The match key uses startGameClock (integer game-clock seconds), which is the
same value stored as "Start Time [s]" in the preprocessed Metrica CSVs and
as "start_time" in labeled_metadata.csv / pass_features.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["load_match_pressure"]

PressureLookup = dict[tuple[int, int, str], bool]


def load_match_pressure(json_path: Path) -> PressureLookup:
    """Return pressure flags for all pass events in one PFF match JSON.

    Parameters
    ----------
    json_path:
        Path to the raw PFF JSON file (e.g. ``data/FIFA_WC_2022/Event Data/3833.json``).

    Returns
    -------
    dict
        Keys are ``(period, game_clock_s, team)`` where team is ``"Home"`` or
        ``"Away"`` and ``game_clock_s`` is the integer startGameClock value.
        Values are ``True`` (under pressure) or ``False`` (not under pressure).
        Only pass events (``possessionEventType == "PA"``) are included.
    """
    with json_path.open(encoding="utf-8") as fh:
        events: list[dict[str, Any]] = json.load(fh)

    lookup: PressureLookup = {}
    for ev in events:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}

        if pe.get("possessionEventType") != "PA":
            continue

        period = ge.get("period")
        clock = ge.get("startGameClock")
        if period is None or clock is None:
            continue

        home_team: bool = bool(ge.get("homeTeam", False))
        team = "Home" if home_team else "Away"
        under_pressure = pe.get("pressureType", "N") != "N"

        lookup[(int(period), int(clock), team)] = under_pressure

    return lookup
