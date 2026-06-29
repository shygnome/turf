"""OBPV gain for a single pass: two-snapshot difference."""

from __future__ import annotations

from obpv._state import GameState
from obpv.analyzer import OBPVAnalyzer


def compute_pass_obpv(
    start_state: GameState,
    end_state: GameState,
    analyzer: OBPVAnalyzer,
) -> float:
    """Return the OBPV gain of a pass.

    Computes the total OBPV surface (sum of all grid cells) at the pass start
    and end frames, then returns end - start.

    Parameters
    ----------
    start_state:
        GameState snapshot at the moment the pass is played.
        ``ball_position`` should be the pass origin, players at their positions
        at that frame.
    end_state:
        GameState snapshot when the pass is received (or the nearest tracked
        frame to the pass end).  ``ball_position`` should be the pass
        destination.
    analyzer:
        A configured OBPVAnalyzer (PPCF + transition + weight models).

    Returns
    -------
    float
        OBPV gain = sum(OBPV surface at end) - sum(OBPV surface at start).
        Positive = pass moved the ball to a more threatening position.
    """
    obpv_start = analyzer.analyze(start_state).get_layer("obpv").values.sum()
    obpv_end = analyzer.analyze(end_state).get_layer("obpv").values.sum()
    return float(obpv_end - obpv_start)

