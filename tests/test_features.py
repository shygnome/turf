"""Tests for turf.features — pass covariate extraction."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from turf.features import (
    build_score_timeline,
    extract_pass_features,
    goal_to_period_seconds,
    normalize_coords,
    pass_angle,
    pass_distance,
    score_at,
    score_diff,
    territory_entry,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def goals_df() -> pd.DataFrame:
    # Germany 1-2 Japan (match 3821)
    return pd.DataFrame(
        {
            "match_id": [3821, 3821, 3821],
            "scoring_team": ["Home", "Away", "Away"],
            "period": [1, 2, 2],
            "minute": [33, 83, 89],
            "added_time": [0, 0, 7],
        }
    )


@pytest.fixture()
def timeline(goals_df: pd.DataFrame) -> dict:
    return build_score_timeline(goals_df)


# ── goal_to_period_seconds ────────────────────────────────────────────────────

class TestGoalToPeriodSeconds:
    def test_period1_regular_time(self) -> None:
        # minute 33 in P1 → (33-1)*60 = 1920s from P1 start
        assert goal_to_period_seconds(1, 33, 0) == pytest.approx(1920.0)

    def test_period2_regular_time(self) -> None:
        # minute 83 in P2 → (83-46)*60 = 2220s from P2 start
        assert goal_to_period_seconds(2, 83, 0) == pytest.approx(2220.0)

    def test_period2_stoppage_time(self) -> None:
        # minute 89+7 in P2 → (89-46)*60 + 7*60 = 2580+420 = 3000s
        assert goal_to_period_seconds(2, 89, 7) == pytest.approx(3000.0)

    def test_et1(self) -> None:
        # ET1 starts at minute 91 (P3); minute 93 → (93-91)*60 = 120s
        assert goal_to_period_seconds(3, 93, 0) == pytest.approx(120.0)

    def test_et2(self) -> None:
        # ET2 starts at minute 106 (P4); minute 108 → (108-106)*60 = 120s
        assert goal_to_period_seconds(4, 108, 0) == pytest.approx(120.0)


# ── score_at ─────────────────────────────────────────────────────────────────

class TestScoreAt:
    def test_before_any_goal(self, timeline: dict) -> None:
        assert score_at(timeline, 3821, 1, 900.0) == (0, 0)

    def test_exactly_at_goal_time_not_counted(self, timeline: dict) -> None:
        # goal at 1920s — pass at exactly the same time should not count it
        assert score_at(timeline, 3821, 1, 1920.0) == (0, 0)

    def test_just_after_first_goal(self, timeline: dict) -> None:
        assert score_at(timeline, 3821, 1, 1921.0) == (1, 0)

    def test_between_goals(self, timeline: dict) -> None:
        # After Germany goal, before Japan's second-half goals
        assert score_at(timeline, 3821, 2, 1000.0) == (1, 0)

    def test_after_second_goal(self, timeline: dict) -> None:
        # Doan's goal at P2 2220s
        assert score_at(timeline, 3821, 2, 2221.0) == (1, 1)

    def test_after_third_goal(self, timeline: dict) -> None:
        # Asano's goal at P2 3000s
        assert score_at(timeline, 3821, 2, 3001.0) == (1, 2)

    def test_unknown_match_returns_zeros(self, timeline: dict) -> None:
        assert score_at(timeline, 9999, 1, 100.0) == (0, 0)


# ── score_diff ────────────────────────────────────────────────────────────────

class TestScoreDiff:
    def test_home_passer_winning(self, timeline: dict) -> None:
        # After Germany goal, Home passer: 1-0 → diff = +1
        assert score_diff(timeline, 3821, 1, 1921.0, "Home") == 1

    def test_away_passer_losing(self, timeline: dict) -> None:
        # After Germany goal, Away passer: 0-1 → diff = -1
        assert score_diff(timeline, 3821, 1, 1921.0, "Away") == -1

    def test_level(self, timeline: dict) -> None:
        assert score_diff(timeline, 3821, 1, 0.0, "Home") == 0
        assert score_diff(timeline, 3821, 1, 0.0, "Away") == 0

    def test_away_passer_winning(self, timeline: dict) -> None:
        # After Asano's goal, Away leads 2-1 → Away diff = +1
        assert score_diff(timeline, 3821, 2, 3001.0, "Away") == 1


# ── pass_distance ─────────────────────────────────────────────────────────────

class TestPassDistance:
    def test_3_4_5_triangle(self) -> None:
        assert pass_distance(0.0, 0.0, 3.0, 4.0) == pytest.approx(5.0)

    def test_same_point_is_zero(self) -> None:
        assert pass_distance(10.0, 5.0, 10.0, 5.0) == pytest.approx(0.0)

    def test_horizontal_pass(self) -> None:
        assert pass_distance(-10.0, 0.0, 10.0, 0.0) == pytest.approx(20.0)


# ── pass_angle ────────────────────────────────────────────────────────────────

class TestPassAngle:
    def test_forward_pass_is_zero(self) -> None:
        assert pass_angle(0.0, 0.0, 10.0, 0.0) == pytest.approx(0.0)

    def test_backward_pass_is_pi(self) -> None:
        assert abs(pass_angle(0.0, 0.0, -10.0, 0.0)) == pytest.approx(math.pi)

    def test_square_pass_upfield(self) -> None:
        assert pass_angle(0.0, 0.0, 0.0, 10.0) == pytest.approx(math.pi / 2)

    def test_diagonal(self) -> None:
        assert pass_angle(0.0, 0.0, 5.0, 5.0) == pytest.approx(math.pi / 4)


# ── territory_entry ───────────────────────────────────────────────────────────

class TestTerritoryEntry:
    def test_in_final_third(self) -> None:
        assert territory_entry(20.0) is True

    def test_on_boundary(self) -> None:
        # boundary at 52.5/3 = 17.5 — falls into attacking third
        assert territory_entry(17.5) is True

    def test_just_below_boundary(self) -> None:
        assert territory_entry(17.4) is False

    def test_deep_own_half(self) -> None:
        assert territory_entry(-30.0) is False


# ── normalize_coords ──────────────────────────────────────────────────────────

class TestNormalizeCoords:
    def test_positive_direction_unchanged(self) -> None:
        assert normalize_coords(10.0, 5.0, 1) == (10.0, 5.0)

    def test_negative_direction_flips_both_axes(self) -> None:
        assert normalize_coords(10.0, 5.0, -1) == (-10.0, -5.0)

    def test_negative_direction_flips_defensive_coords(self) -> None:
        # A team attacking in -x direction: their own goal is at +x.
        # After flip, their own goal lands at -x (defensive end) — correct.
        assert normalize_coords(-30.0, -10.0, -1) == (30.0, 10.0)


# ── extract_pass_features (integration) ───────────────────────────────────────

class TestExtractPassFeatures:
    @pytest.fixture()
    def labeled_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "event_idx": 0,
                    "period": 1,
                    "team": "Home",
                    "start_time": 500.0,
                    "start_x": 5.0,
                    "start_y": 0.0,
                    "inferred_end_x": 25.0,  # enters final third (>17.5)
                    "inferred_end_y": 0.0,
                    "subtype": "success",
                    "is_line_breaking": True,
                    "lines_broken_count": 1,
                },
                {
                    "event_idx": 1,
                    "period": 1,
                    "team": "Home",
                    "start_time": 600.0,
                    "start_x": -20.0,
                    "start_y": 5.0,
                    "inferred_end_x": -10.0,
                    "inferred_end_y": 5.0,
                    "subtype": "fail",
                    "is_line_breaking": False,
                    "lines_broken_count": 0,
                },
            ]
        )

    @pytest.fixture()
    def attack_dirs(self) -> dict:
        # Home attacks toward +x in P1
        return {(1, "Home"): 1}

    def test_output_columns(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        expected = {
            "event_idx", "match_id", "team", "period", "start_time",
            "under_pressure",
            "is_line_breaking", "lines_broken_count", "pass_outcome",
            "zone_thirds", "zone_van_gaal", "zone_guardiola",
            "pass_distance", "pass_angle", "territory_entry", "score_diff",
            "expected_xt_gain", "actual_xt_gain",
            "expected_obpv_gain", "actual_obpv_gain",
        }
        assert expected.issubset(set(result.columns))

    def test_row_count(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert len(result) == 2

    def test_pass_outcome_maps_subtype(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert bool(result.iloc[0]["pass_outcome"]) is True
        assert bool(result.iloc[1]["pass_outcome"]) is False

    def test_territory_entry_correct(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        # First pass ends at norm_end_x=25.0 → in final third
        assert bool(result.iloc[0]["territory_entry"]) is True
        # Second pass ends at norm_end_x=-10.0 → not in final third
        assert bool(result.iloc[1]["territory_entry"]) is False

    def test_zone_thirds_from_start(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        # start_x=5.0 with attack_dir=1 → norm_x=5.0 → t_mid
        assert result.iloc[0]["zone_thirds"] == "t_mid"
        # start_x=-20.0 with attack_dir=1 → norm_x=-20.0 → t_def
        assert result.iloc[1]["zone_thirds"] == "t_def"

    def test_score_diff_before_goals(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        # Both passes at t=500 and t=600, before Germany goal at t=1920
        assert result.iloc[0]["score_diff"] == 0
        assert result.iloc[1]["score_diff"] == 0

    def test_actual_xt_gain_positive_for_completed_forward_pass(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert result.iloc[0]["actual_xt_gain"] > 0

    def test_actual_xt_gain_zero_for_incomplete_pass(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert result.iloc[1]["actual_xt_gain"] == pytest.approx(0.0)

    def test_expected_xt_gain_nonzero_for_incomplete_pass(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        # Second pass is incomplete but ball moved backward → expected gain is non-zero
        assert result.iloc[1]["expected_xt_gain"] != pytest.approx(0.0)

    def test_obpv_columns_nan_without_pass_dir(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        import math
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert math.isnan(result.iloc[0]["expected_obpv_gain"])
        assert math.isnan(result.iloc[0]["actual_obpv_gain"])

    def test_skips_rows_with_no_attack_dir(
        self,
        labeled_df: pd.DataFrame,
        timeline: dict,
    ) -> None:
        # Only P1 Home available; Away rows should be skipped
        attack_dirs: dict = {(1, "Home"): 1}
        labeled_with_away = pd.concat(
            [
                labeled_df,
                pd.DataFrame(
                    [
                        {
                            "event_idx": 99,
                            "period": 1,
                            "team": "Away",
                            "start_time": 700.0,
                            "start_x": 0.0,
                            "start_y": 0.0,
                            "inferred_end_x": 10.0,
                            "inferred_end_y": 0.0,
                            "subtype": "success",
                            "is_line_breaking": False,
                            "lines_broken_count": 0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        result = extract_pass_features(
            labeled_with_away, timeline, attack_dirs, match_id=3821
        )
        assert len(result) == 2  # Away row skipped

    def test_under_pressure_false_without_lookup(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        result = extract_pass_features(labeled_df, timeline, attack_dirs, match_id=3821)
        assert bool(result.iloc[0]["under_pressure"]) is False
        assert bool(result.iloc[1]["under_pressure"]) is False

    def test_under_pressure_from_lookup(
        self,
        labeled_df: pd.DataFrame,
        attack_dirs: dict,
        timeline: dict,
    ) -> None:
        from turf.pressure import PressureLookup
        # Row 0: period=1, start_time=500 → key (1, 500, "Home") → True
        # Row 1: period=1, start_time=600 → key (1, 600, "Home") → absent → False
        lookup: PressureLookup = {(1, 500, "Home"): True}
        result = extract_pass_features(
            labeled_df, timeline, attack_dirs, match_id=3821, pressure_lookup=lookup
        )
        assert bool(result.iloc[0]["under_pressure"]) is True
        assert bool(result.iloc[1]["under_pressure"]) is False


class TestExtractPassFeaturesWithTracking:
    """Integration tests that provide synthetic tracking frames."""

    @pytest.fixture()
    def pass_dir(self, tmp_path: Path) -> Path:
        """Write minimal frames_home.csv and frames_away.csv for event_idx 0."""
        import math

        event_dir = tmp_path / "0"
        event_dir.mkdir()

        home_rows = [
            {
                "frame": 0, "Period": 1, "Time [s]": 0.0,
                "Home_1_x": 5.0, "Home_1_y": 0.0,
                "Home_2_x": -10.0, "Home_2_y": 3.0,
                "ball_x": 5.0, "ball_y": 0.0,
            },
            {
                "frame": 1, "Period": 1, "Time [s]": 0.04,
                "Home_1_x": 5.2, "Home_1_y": 0.0,
                "Home_2_x": -9.8, "Home_2_y": 3.0,
                "ball_x": 10.0, "ball_y": 0.0,
            },
        ]
        away_rows = [
            {
                "frame": 0, "Period": 1, "Time [s]": 0.0,
                "Away_1_x": -15.0, "Away_1_y": 0.0,
                "ball_x": 5.0, "ball_y": 0.0,
            },
            {
                "frame": 1, "Period": 1, "Time [s]": 0.04,
                "Away_1_x": -14.8, "Away_1_y": 0.0,
                "ball_x": 10.0, "ball_y": 0.0,
            },
        ]
        pd.DataFrame(home_rows).to_csv(event_dir / "frames_home.csv", index=False)
        pd.DataFrame(away_rows).to_csv(event_dir / "frames_away.csv", index=False)

        assert not math.isnan(0.0)  # sanity
        return tmp_path

    @pytest.fixture()
    def single_pass_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "event_idx": 0,
            "period": 1,
            "team": "Home",
            "start_time": 0.0,
            "start_x": 5.0,
            "start_y": 0.0,
            "inferred_end_x": 10.0,
            "inferred_end_y": 0.0,
            "subtype": "success",
            "is_line_breaking": True,
            "lines_broken_count": 1,
        }])

    def test_obpv_columns_finite_with_pass_dir(
        self,
        single_pass_df: pd.DataFrame,
        pass_dir: Path,
        timeline: dict,
    ) -> None:
        import math
        attack_dirs = {(1, "Home"): 1}
        result = extract_pass_features(
            single_pass_df, timeline, attack_dirs, match_id=3821,
            pass_dir=pass_dir,
        )
        assert not math.isnan(result.iloc[0]["expected_obpv_gain"])
        assert not math.isnan(result.iloc[0]["actual_obpv_gain"])

    def test_actual_obpv_gain_zero_for_incomplete(
        self,
        single_pass_df: pd.DataFrame,
        pass_dir: Path,
        timeline: dict,
    ) -> None:
        single_pass_df = single_pass_df.copy()
        single_pass_df.loc[0, "subtype"] = "fail"
        attack_dirs = {(1, "Home"): 1}
        result = extract_pass_features(
            single_pass_df, timeline, attack_dirs, match_id=3821,
            pass_dir=pass_dir,
        )
        assert result.iloc[0]["actual_obpv_gain"] == pytest.approx(0.0)
