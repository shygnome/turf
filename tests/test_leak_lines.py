from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from leak.lines import analyze_lines, detect_lines_frame, find_goalkeeper

# ── fixtures ──────────────────────────────────────────────────────────────────


def _away_df(**players: float) -> pd.DataFrame:
    """Single-row DataFrame with Away_N_x columns from keyword args."""
    return pd.DataFrame([{f"Away_{n}_x": v for n, v in players.items()}])


def _event_df() -> pd.DataFrame:
    """Five-frame event with GK (player 3) and two clear defensive lines."""
    rows = []
    for f in range(5):
        row: dict[str, object] = {"frame": f, "Period": 1, "Time [s]": f * 0.033}
        row["Away_3_x"] = 43.0 + f * 0.01
        row["Away_3_y"] = 0.0
        for pid, x, y in [
            ("2", 8.0, -12.0),
            ("11", 9.0, -5.0),
            ("6", 18.0, -6.0),
            ("7", 19.0, 6.0),
            ("9", 25.0, -4.0),
            ("10", 26.0, 4.0),
        ]:
            row[f"Away_{pid}_x"] = x + f * 0.1
            row[f"Away_{pid}_y"] = float(y)
        rows.append(row)
    return pd.DataFrame(rows)


# ── find_goalkeeper ───────────────────────────────────────────────────────────


class TestFindGoalkeeper:
    def test_returns_player_with_highest_abs_x(self) -> None:
        df = _away_df(**{"2": 10.0, "3": 44.0, "4": 18.0})
        assert find_goalkeeper(df, "Away") == "3"

    def test_uses_first_row_only(self) -> None:
        df = pd.DataFrame(
            [{"Away_2_x": 10.0, "Away_3_x": 44.0}, {"Away_2_x": 55.0, "Away_3_x": 5.0}]
        )
        assert find_goalkeeper(df, "Away") == "3"

    def test_negative_x_side(self) -> None:
        df = _away_df(**{"2": -5.0, "3": -43.0, "4": -12.0})
        assert find_goalkeeper(df, "Away") == "3"


# ── detect_lines_frame ────────────────────────────────────────────────────────


class TestDetectLinesFrame:
    def test_empty_returns_empty(self) -> None:
        assert detect_lines_frame({}) == {}

    def test_three_clear_groups_assigns_correct_lines(self) -> None:
        positions = {
            "2": 8.0, "11": 9.0,
            "6": 18.0, "7": 19.0,
            "9": 27.0, "10": 28.0,
        }
        result = detect_lines_frame(positions)
        assert result["2"] == result["11"] == 1
        assert result["6"] == result["7"] == 2
        assert result["9"] == result["10"] == 3

    def test_lines_ordered_deepest_first(self) -> None:
        positions = {
            "2": 28.0, "11": 29.0,
            "6": 10.0, "7": 11.0,
        }
        result = detect_lines_frame(positions)
        assert result["6"] < result["2"]

    def test_never_more_than_four_lines(self) -> None:
        positions = {str(i): float(i * 3) for i in range(1, 11)}
        assert len(set(detect_lines_frame(positions).values())) <= 4

    def test_at_least_two_lines_when_enough_players(self) -> None:
        positions = {str(i): float(i) for i in range(1, 9)}
        assert len(set(detect_lines_frame(positions).values())) >= 2

    def test_singleton_merged_all_lines_min_two_players(self) -> None:
        positions = {
            "2": 8.0, "11": 9.0,
            "6": 18.0, "7": 19.0, "8": 19.5,
            "5": 22.5,  # isolated between groups
            "9": 30.0, "10": 31.0,
        }
        result = detect_lines_frame(positions)
        counts = Counter(result.values())
        assert all(c >= 2 for c in counts.values())

    def test_all_players_assigned(self) -> None:
        positions = {"2": 8.0, "11": 9.0, "6": 18.0, "7": 19.0}
        result = detect_lines_frame(positions)
        assert set(result.keys()) == set(positions.keys())

    def test_fallback_preserves_min_lines_when_merge_would_collapse(self) -> None:
        # 3 tightly packed + 1 outlier: Ward k=2 gives 3+1,
        # naive merge would collapse to 1 line
        positions = {"2": 1.0, "11": 1.1, "6": 1.2, "7": 10.0}
        result = detect_lines_frame(positions)
        assert len(set(result.values())) >= 2


# ── analyze_lines ─────────────────────────────────────────────────────────────


class TestAnalyzeLines:
    def test_adds_line_column_for_each_player(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away")
        for pid in ["2", "3", "6", "7", "9", "10", "11"]:
            assert f"Away_{pid}_line" in result.columns

    def test_adds_line_count_column(self) -> None:
        assert "line_count" in analyze_lines(_event_df(), "Away").columns

    def test_gk_always_line_zero(self) -> None:
        result = analyze_lines(_event_df(), "Away")
        assert (result["Away_3_line"] == 0).all()

    def test_outfield_lines_between_one_and_four(self) -> None:
        result = analyze_lines(_event_df(), "Away")
        for pid in ["2", "6", "7", "9", "10", "11"]:
            vals = result[f"Away_{pid}_line"].dropna()
            assert vals.between(1, 4).all(), f"Player {pid} out-of-range line values"

    def test_absent_player_x_gives_nan_line(self) -> None:
        df = _event_df()
        df["Away_14_x"] = np.nan
        df["Away_14_y"] = np.nan
        result = analyze_lines(df, "Away")
        assert result["Away_14_line"].isna().all()

    def test_partially_present_player_nan_for_missing_frames(self) -> None:
        df = _event_df()
        df["Away_14_x"] = np.nan
        df["Away_14_y"] = np.nan
        df.loc[0:1, "Away_14_x"] = [12.0, 12.1]
        result = analyze_lines(df, "Away")
        assert not pd.isna(result.loc[0, "Away_14_line"])
        assert pd.isna(result.loc[2, "Away_14_line"])

    def test_original_columns_preserved(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away")
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self) -> None:
        df = _event_df()
        assert len(analyze_lines(df, "Away")) == len(df)

    def test_absent_gk_gets_nan_line(self) -> None:
        df = _event_df()
        df.loc[2, "Away_3_x"] = np.nan
        result = analyze_lines(df, "Away")
        assert pd.isna(result.loc[2, "Away_3_line"])
        assert result.loc[0, "Away_3_line"] == 0
