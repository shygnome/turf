from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turf.pass_label import (
    assign_zone,
    compute_conceptual_line_xs,
    compute_pass_stats,
    detect_attack_direction,
    detect_line_break,
    label_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lines_df(
    gk_x: float,
    line_player_xs: dict[int, list[float]],
    ball_x: float = 0.0,
    ball_y: float = 0.0,
    team: str = "Away",
    n_frames: int = 1,
) -> pd.DataFrame:
    """
    Build a minimal lines.csv-shaped DataFrame.

    gk_x: x position of the GK (assigned line=0)
    line_player_xs: {LEAK_line_num -> [x positions of players in that line]}
    """
    rows = []
    for _ in range(n_frames):
        row: dict[str, object] = {"frame": 0, "Period": 1, "Time [s]": 0.0}
        # GK
        row[f"{team}_0_x"] = gk_x
        row[f"{team}_0_y"] = 0.0
        row[f"{team}_0_line"] = 0.0
        # Outfield players
        player_num = 1
        for leak_line, xs in line_player_xs.items():
            for x in xs:
                row[f"{team}_{player_num}_x"] = x
                row[f"{team}_{player_num}_y"] = 0.0
                row[f"{team}_{player_num}_line"] = float(leak_line)
                player_num += 1
        row["ball_x"] = ball_x
        row["ball_y"] = ball_y
        row["line_count"] = float(len(line_player_xs))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# detect_attack_direction
# ---------------------------------------------------------------------------


class TestDetectAttackDirection:
    def test_gk_at_positive_x_returns_plus_one(self) -> None:
        df = _make_lines_df(gk_x=40.0, line_player_xs={1: [5.0], 2: [20.0]})
        assert detect_attack_direction(df, "Away") == 1

    def test_gk_at_negative_x_returns_minus_one(self) -> None:
        df = _make_lines_df(gk_x=-40.0, line_player_xs={1: [-20.0], 2: [-5.0]})
        assert detect_attack_direction(df, "Away") == -1

    def test_median_used_over_multiple_frames(self) -> None:
        # GK mostly at positive x but one outlier frame at negative
        df = _make_lines_df(gk_x=38.0, line_player_xs={1: [5.0], 2: [18.0]}, n_frames=3)
        # Override one frame to have GK at negative x
        df.loc[0, "Away_0_x"] = -38.0
        assert detect_attack_direction(df, "Away") == 1

    def test_raises_when_no_gk_column(self) -> None:
        df = pd.DataFrame({"Away_1_x": [5.0], "Away_1_line": [1.0], "ball_x": [0.0]})
        with pytest.raises(ValueError, match="GK"):
            detect_attack_direction(df, "Away")


# ---------------------------------------------------------------------------
# compute_conceptual_line_xs
# ---------------------------------------------------------------------------


class TestComputeConceptualLineXs:
    def test_attack_dir_plus_one_ascending_order(self) -> None:
        # attack_dir=+1: conceptual L1 = lowest x (first encountered moving +x)
        df = _make_lines_df(
            gk_x=40.0,
            line_player_xs={1: [4.0, 6.0], 2: [12.0, 14.0], 3: [22.0, 24.0]},
        )
        xs = compute_conceptual_line_xs(df.iloc[0], "Away", attack_dir=1)
        # Conceptual L1=5.0, L2=13.0, L3=23.0 (means, ascending)
        assert xs == pytest.approx([5.0, 13.0, 23.0])

    def test_attack_dir_minus_one_descending_x_order(self) -> None:
        # attack_dir=-1: conceptual L1 = highest x (first encountered moving -x)
        df = _make_lines_df(
            gk_x=-40.0,
            line_player_xs={1: [-22.0, -24.0], 2: [-12.0, -14.0], 3: [-4.0, -6.0]},
        )
        xs = compute_conceptual_line_xs(df.iloc[0], "Away", attack_dir=-1)
        # Means: LEAK L1=-23, L2=-13, L3=-5
        # Sorted by ascending norm (x * -1): -5→norm=5, -13→norm=13, -23→norm=23
        # So conceptual order = [-5.0, -13.0, -23.0] (descending x)
        assert xs == pytest.approx([-5.0, -13.0, -23.0])

    def test_nan_players_excluded_from_mean(self) -> None:
        df = _make_lines_df(gk_x=40.0, line_player_xs={1: [4.0, 6.0], 2: [20.0, 22.0]})
        # Manually inject NaN for one player in line 1
        df.loc[0, "Away_1_x"] = float("nan")
        xs = compute_conceptual_line_xs(df.iloc[0], "Away", attack_dir=1)
        # Line 1: only 6.0 valid → mean=6.0; line 2: 21.0
        assert xs[0] == pytest.approx(6.0)
        assert xs[1] == pytest.approx(21.0)

    def test_excludes_gk_from_line_xs(self) -> None:
        # GK (line=0) at x=40 should NOT appear in conceptual line xs
        df = _make_lines_df(gk_x=40.0, line_player_xs={1: [5.0], 2: [20.0]})
        xs = compute_conceptual_line_xs(df.iloc[0], "Away", attack_dir=1)
        assert all(x < 40.0 for x in xs)


# ---------------------------------------------------------------------------
# assign_zone
# ---------------------------------------------------------------------------


class TestAssignZone:
    # Lines at x=[10, 20, 30], attack_dir=+1
    # conceptual L1=10 (most advanced press), L2=20, L3=30
    # Zones (attack_dir=+1, lines=[10,20,30]):
    #   L1-zone(x<10) | L1 | L2-zone(10..20) | L2 | L3-zone(20..30) | L3 | danger-zone

    def test_before_first_line_is_l1_zone(self) -> None:
        assert assign_zone(5.0, [10.0, 20.0, 30.0], attack_dir=1) == "L1-zone"

    def test_exactly_at_first_line_is_l2_zone(self) -> None:
        assert assign_zone(10.0, [10.0, 20.0, 30.0], attack_dir=1) == "L2-zone"

    def test_between_first_and_second_line_is_l2_zone(self) -> None:
        assert assign_zone(15.0, [10.0, 20.0, 30.0], attack_dir=1) == "L2-zone"

    def test_between_second_and_third_line_is_l3_zone(self) -> None:
        assert assign_zone(25.0, [10.0, 20.0, 30.0], attack_dir=1) == "L3-zone"

    def test_past_all_lines_is_danger_zone(self) -> None:
        assert assign_zone(35.0, [10.0, 20.0, 30.0], attack_dir=1) == "danger-zone"

    def test_attack_dir_minus_one_before_press_is_l1_zone(self) -> None:
        # Lines (conceptual order, descending x): [30, 20, 10]
        # L1=30 (highest x = most advanced press when moving -x)
        # L1-zone: x > 30 (haven't beaten press yet)
        assert assign_zone(35.0, [30.0, 20.0, 10.0], attack_dir=-1) == "L1-zone"

    def test_attack_dir_minus_one_between_l1_and_l2_is_l2_zone(self) -> None:
        assert assign_zone(25.0, [30.0, 20.0, 10.0], attack_dir=-1) == "L2-zone"

    def test_attack_dir_minus_one_past_all_is_danger_zone(self) -> None:
        assert assign_zone(5.0, [30.0, 20.0, 10.0], attack_dir=-1) == "danger-zone"

    def test_two_line_case(self) -> None:
        # Only two lines
        assert assign_zone(0.0, [10.0, 20.0], attack_dir=1) == "L1-zone"
        assert assign_zone(15.0, [10.0, 20.0], attack_dir=1) == "L2-zone"
        assert assign_zone(25.0, [10.0, 20.0], attack_dir=1) == "danger-zone"


# ---------------------------------------------------------------------------
# detect_line_break
# ---------------------------------------------------------------------------


class TestDetectLineBreak:
    # Helpers: lines at conceptual xs [10, 20, 30], attack_dir=+1

    _lines = [10.0, 20.0, 30.0]

    def _label(
        self,
        start_x: float,
        end_x: float,
        start_lines: list[float] | None = None,
        end_lines: list[float] | None = None,
    ) -> dict:
        sl = start_lines if start_lines is not None else self._lines
        el = end_lines if end_lines is not None else self._lines
        return detect_line_break(start_x, end_x, sl, el, attack_dir=1)

    def test_no_break_same_zone(self) -> None:
        result = self._label(start_x=5.0, end_x=8.0)
        assert result["is_line_breaking"] is False
        assert result["lines_broken_count"] == 0
        assert result["lines_broken"] == []

    def test_no_break_backward_movement(self) -> None:
        result = self._label(start_x=15.0, end_x=5.0)
        assert result["is_line_breaking"] is False

    def test_one_line_broken(self) -> None:
        # L1-zone → L2-zone: beat conceptual L1
        result = self._label(start_x=5.0, end_x=15.0)
        assert result["is_line_breaking"] is True
        assert result["lines_broken_count"] == 1
        assert result["lines_broken"] == [1]

    def test_two_lines_broken(self) -> None:
        # L1-zone → L3-zone: beat L1 and L2
        result = self._label(start_x=5.0, end_x=25.0)
        assert result["is_line_breaking"] is True
        assert result["lines_broken_count"] == 2
        assert result["lines_broken"] == [1, 2]

    def test_all_lines_broken_into_danger_zone(self) -> None:
        result = self._label(start_x=5.0, end_x=35.0)
        assert result["is_line_breaking"] is True
        assert result["lines_broken_count"] == 3
        assert result["lines_broken"] == [1, 2, 3]

    def test_break_starting_from_l2_zone(self) -> None:
        # L2-zone → L3-zone: beat conceptual L2 (not L1)
        result = self._label(start_x=15.0, end_x=25.0)
        assert result["is_line_breaking"] is True
        assert result["lines_broken_count"] == 1
        assert result["lines_broken"] == [2]

    def test_line_positions_shift_between_frames(self) -> None:
        # Start frame: lines at [10, 20, 30]; ball starts in L1-zone (x=5)
        # End frame: lines shifted forward to [15, 25, 35]; ball at x=20
        # End zone relative to end lines: 15<=20<25 → L2-zone (beat conceptual L1)
        result = self._label(
            start_x=5.0,
            end_x=20.0,
            start_lines=[10.0, 20.0, 30.0],
            end_lines=[15.0, 25.0, 35.0],
        )
        assert result["is_line_breaking"] is True
        assert result["lines_broken"] == [1]

    def test_attack_dir_minus_one(self) -> None:
        # Lines conceptual (descending x): [30, 20, 10], attack_dir=-1
        # Ball starts at x=35 (L1-zone), ends at x=15 (L3-zone)
        result = detect_line_break(
            start_ball_x=35.0,
            end_ball_x=15.0,
            start_line_xs=[30.0, 20.0, 10.0],
            end_line_xs=[30.0, 20.0, 10.0],
            attack_dir=-1,
        )
        assert result["is_line_breaking"] is True
        assert result["lines_broken_count"] == 2
        assert result["lines_broken"] == [1, 2]


# ---------------------------------------------------------------------------
# label_event (integration — uses a real directory structure)
# ---------------------------------------------------------------------------


class TestLabelEvent:
    def _build_event_dir(
        self, tmp_path: Path, gk_x: float, line_xs: dict[int, list[float]]
    ) -> tuple[Path, str]:
        """Write a minimal lines.csv and return (event_dir, defending_team)."""
        team = "Away"
        n_frames = 5
        rows = []
        for f in range(n_frames):
            row: dict[str, object] = {
                "frame": float(f),
                "Period": 1.0,
                "Time [s]": float(f) * 0.04,
            }
            row[f"{team}_0_x"] = gk_x
            row[f"{team}_0_y"] = 0.0
            row[f"{team}_0_line"] = 0.0
            player_num = 1
            for leak_line, xs in line_xs.items():
                for x in xs:
                    row[f"{team}_{player_num}_x"] = x
                    row[f"{team}_{player_num}_y"] = 0.0
                    row[f"{team}_{player_num}_line"] = float(leak_line)
                    player_num += 1
            row["ball_x"] = float(f)  # ball moves +x each frame
            row["ball_y"] = 0.0
            row["line_count"] = float(len(line_xs))
            rows.append(row)

        df = pd.DataFrame(rows)
        event_dir = tmp_path / "0"
        event_dir.mkdir()
        df.to_csv(event_dir / "lines.csv", index=False)
        return event_dir, team

    def test_label_event_line_breaking(self, tmp_path: Path) -> None:
        # GK at +40: attack_dir=+1
        # Conceptual L1=10 (line 1 players at x=10), L2=20
        # Ball moves from x=0 (frame 0) to x=4 (frame 4)
        # Start zone: x=0 < 10 → L1-zone; End zone: x=4 < 10 → L1-zone → NO break
        event_dir, team = self._build_event_dir(
            tmp_path, gk_x=40.0, line_xs={1: [10.0, 10.0], 2: [20.0, 20.0]}
        )
        result = label_event(event_dir, team)
        assert result["is_line_breaking"] is False

    def test_label_event_break_detected(self, tmp_path: Path) -> None:
        # Ball moves from x=0 (frame 0) to x=4 (frame 4)
        # Lines: L1 at x=3, L2 at x=15
        # Start: x=0 < 3 → L1-zone; End: x=4 between 3 and 15 → L2-zone → break
        event_dir, team = self._build_event_dir(
            tmp_path, gk_x=40.0, line_xs={1: [3.0, 3.0], 2: [15.0, 15.0]}
        )
        result = label_event(event_dir, team)
        assert result["is_line_breaking"] is True
        assert result["lines_broken"] == [1]


# ---------------------------------------------------------------------------
# compute_pass_stats
# ---------------------------------------------------------------------------


def _make_labeled_df() -> pd.DataFrame:
    """Minimal labeled_metadata DataFrame for stats tests."""
    return pd.DataFrame(
        {
            "event_idx": range(10),
            "team": ["Home"] * 6 + ["Away"] * 4,
            "period": [1, 1, 1, 2, 2, 2, 1, 1, 2, 2],
            "subtype": ["success"] * 7 + ["fail"] * 3,
            "is_line_breaking": [
                True,
                True,
                False,
                True,
                False,
                False,  # Home
                True,
                False,
                True,
                None,  # Away (last = skipped)
            ],
            "lines_broken_count": [1, 2, 0, 1, 0, 0, 2, 0, 1, None],
            "lines_broken": [
                "[1]",
                "[1, 2]",
                "[]",
                "[1]",
                "[]",
                "[]",
                "[1, 2]",
                "[]",
                "[1]",
                None,
            ],
        }
    )


class TestComputePassStats:
    def test_totals(self) -> None:
        stats = compute_pass_stats(_make_labeled_df())
        assert stats["total"] == 10
        assert stats["n_labeled"] == 9  # one None row
        assert stats["n_skipped"] == 1
        assert stats["n_breaking"] == 5
        assert stats["n_not_breaking"] == 4

    def test_lines_broken_distribution(self) -> None:
        stats = compute_pass_stats(_make_labeled_df())
        dist = stats["lines_broken_dist"]
        assert dist[1] == 3  # three passes broke exactly 1 line
        assert dist[2] == 2  # two passes broke exactly 2 lines
        assert dist.get(3, 0) == 0

    def test_by_team(self) -> None:
        stats = compute_pass_stats(_make_labeled_df())
        # Home: 6 rows, 1 skipped=0, breaking = True,True,False,True,False,False -> 3
        assert stats["by_team"]["Home"]["breaking"] == 3
        assert stats["by_team"]["Home"]["total"] == 6
        # Away: 4 rows, 1 skipped (None), labeled=3, breaking=True,False,True -> 2
        assert stats["by_team"]["Away"]["breaking"] == 2
        assert stats["by_team"]["Away"]["total"] == 3  # labeled only

    def test_by_period(self) -> None:
        stats = compute_pass_stats(_make_labeled_df())
        # Period 1 labeled: idx 0,1,2,6,7 = 5 rows; breaking: T,T,F,T,F = 3
        assert stats["by_period"][1]["breaking"] == 3
        assert stats["by_period"][1]["total"] == 5
        # Period 2 labeled: idx 3,4,5,8 = 4 rows (idx9 skipped); breaking: T,F,F,T = 2
        assert stats["by_period"][2]["breaking"] == 2
        assert stats["by_period"][2]["total"] == 4

    def test_by_subtype(self) -> None:
        stats = compute_pass_stats(_make_labeled_df())
        # success: idx 0-6 = 7 rows, all labeled; breaking: T,T,F,T,F,F,T = 4
        assert stats["by_subtype"]["success"]["breaking"] == 4
        assert stats["by_subtype"]["success"]["total"] == 7
        # fail: idx 7,8,9 = 3 rows, idx9 skipped; labeled=2; breaking: F,T = 1
        assert stats["by_subtype"]["fail"]["breaking"] == 1
        assert stats["by_subtype"]["fail"]["total"] == 2

    def test_all_skipped_returns_zeros(self) -> None:
        df = pd.DataFrame(
            {
                "event_idx": [0, 1],
                "team": ["Home", "Away"],
                "period": [1, 1],
                "subtype": ["success", "fail"],
                "is_line_breaking": [None, None],
                "lines_broken_count": [None, None],
                "lines_broken": [None, None],
            }
        )
        stats = compute_pass_stats(df)
        assert stats["n_labeled"] == 0
        assert stats["n_breaking"] == 0
        assert stats["lines_broken_dist"] == {}
