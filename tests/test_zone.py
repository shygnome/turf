"""Tests for turf.zone — pitch zone assignment."""

import pytest

from turf.zone import ZoneScheme, assign_zone


class TestSimpleThirds:
    def test_assigns_correct_third(self) -> None:
        assert assign_zone(-40.0, 0.0, ZoneScheme.THIRDS) == "t_def"
        assert assign_zone(0.0, 0.0, ZoneScheme.THIRDS) == "t_mid"
        assert assign_zone(40.0, 0.0, ZoneScheme.THIRDS) == "t_att"

    def test_boundary_falls_into_right_zone(self) -> None:
        # exact boundary x = -17.5 goes to middle, not defensive
        assert assign_zone(-17.5, 0.0, ZoneScheme.THIRDS) == "t_mid"
        # exact boundary x = 17.5 goes to attacking, not middle
        assert assign_zone(17.5, 0.0, ZoneScheme.THIRDS) == "t_att"

    def test_pitch_corners_are_assigned(self) -> None:
        assert assign_zone(-52.5, -34.0, ZoneScheme.THIRDS) == "t_def"
        assert assign_zone(52.5, 34.0, ZoneScheme.THIRDS) == "t_att"


class TestVanGaal:
    def test_centre_of_pitch(self) -> None:
        # x=0 is on the halfway boundary → goes to attacking half (m_att)
        # y=0 is within penalty area width → centre row (ctr)
        assert assign_zone(0.0, 0.0, ZoneScheme.VAN_GAAL) == "vg_m_att_ctr"

    def test_penalty_area_columns(self) -> None:
        # x=-36 is the defensive penalty depth boundary → d_mid
        assert assign_zone(-36.0, 0.0, ZoneScheme.VAN_GAAL) == "vg_d_mid_ctr"
        # x=36 is the attacking penalty depth boundary → a_box
        assert assign_zone(36.0, 0.0, ZoneScheme.VAN_GAAL) == "vg_a_box_ctr"

    def test_flank_rows(self) -> None:
        # y=-25 is outside penalty area width (-34 to -20.16) → bottom flank (fl)
        assert assign_zone(0.0, -25.0, ZoneScheme.VAN_GAAL) == "vg_m_att_fl"
        # y=25 is outside penalty area width (20.16 to 34) → top flank (fr)
        assert assign_zone(0.0, 25.0, ZoneScheme.VAN_GAAL) == "vg_m_att_fr"

    def test_penalty_area_row_boundary(self) -> None:
        # y=-20.16 is the row boundary → goes right into ctr, not fl
        assert assign_zone(0.0, -20.16, ZoneScheme.VAN_GAAL) == "vg_m_att_ctr"

    def test_pitch_corners(self) -> None:
        assert assign_zone(-52.5, -34.0, ZoneScheme.VAN_GAAL) == "vg_d_box_fl"
        assert assign_zone(52.5, 34.0, ZoneScheme.VAN_GAAL) == "vg_a_box_fr"

    def test_inner_columns(self) -> None:
        assert assign_zone(-18.0, 0.0, ZoneScheme.VAN_GAAL) == "vg_m_def_ctr"
        assert assign_zone(18.0, 0.0, ZoneScheme.VAN_GAAL) == "vg_a_mid_ctr"


class TestGuardiola:
    def test_gk_boxes(self) -> None:
        # deep defensive, centre lane → GK def box
        assert assign_zone(-40.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_gk_def"
        # deep attacking, centre lane → GK att box
        assert assign_zone(40.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_gk_att"
        # GK box spans all inner lanes (half-spaces + centre)
        assert assign_zone(-40.0, -15.0, ZoneScheme.GUARDIOLA) == "pg_gk_def"
        assert assign_zone(40.0, 15.0, ZoneScheme.GUARDIOLA) == "pg_gk_att"

    def test_gk_box_boundary(self) -> None:
        # x=-36 is NOT inside GK box (boundary goes to inner zone)
        assert assign_zone(-36.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_ctr_def"
        # x=36 IS the GK att boundary → pg_gk_att
        assert assign_zone(36.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_gk_att"

    def test_inner_lane_halves(self) -> None:
        # centre channel, defensive half
        assert assign_zone(-10.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_ctr_def"
        # centre channel, attacking half (x=0 boundary → att)
        assert assign_zone(0.0, 0.0, ZoneScheme.GUARDIOLA) == "pg_ctr_att"
        # left half-space, defensive half
        assert assign_zone(-10.0, -15.0, ZoneScheme.GUARDIOLA) == "pg_lhs_def"
        # right half-space, attacking half
        assert assign_zone(10.0, 15.0, ZoneScheme.GUARDIOLA) == "pg_rhs_att"

    def test_flank_zones(self) -> None:
        # bottom flank, near defensive goal
        assert assign_zone(-40.0, -30.0, ZoneScheme.GUARDIOLA) == "pg_lf_d_box"
        # top flank, build-up area
        assert assign_zone(-20.0, 25.0, ZoneScheme.GUARDIOLA) == "pg_rf_d_mid"
        # bottom flank, attacking mid
        assert assign_zone(20.0, -25.0, ZoneScheme.GUARDIOLA) == "pg_lf_a_mid"

    def test_flank_does_not_produce_gk_zone(self) -> None:
        # deep defensive x, but in flank lane → NOT a GK box zone
        assert assign_zone(-40.0, -30.0, ZoneScheme.GUARDIOLA) == "pg_lf_d_box"

    def test_half_space_lane_boundary(self) -> None:
        # y=-20.16 is the boundary between lf and lhs → goes into lhs
        assert assign_zone(-10.0, -20.16, ZoneScheme.GUARDIOLA) == "pg_lhs_def"
        # y=-9.15 is the boundary between lhs and ctr → goes into ctr
        assert assign_zone(-10.0, -9.15, ZoneScheme.GUARDIOLA) == "pg_ctr_def"

    def test_pitch_corners_are_flank_zones(self) -> None:
        assert assign_zone(-52.5, -34.0, ZoneScheme.GUARDIOLA) == "pg_lf_d_box"
        assert assign_zone(52.5, 34.0, ZoneScheme.GUARDIOLA) == "pg_rf_a_box"


def test_unknown_scheme_raises() -> None:
    with pytest.raises((ValueError, AttributeError)):
        assign_zone(0.0, 0.0, "bad_scheme")  # type: ignore[arg-type]


class TestGuardiolaZoneNumber:
    def test_all_20_zones_have_a_number(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_NUMBER
        assert len(GUARDIOLA_ZONE_NUMBER) == 20

    def test_numbers_are_1_to_20(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_NUMBER
        assert set(GUARDIOLA_ZONE_NUMBER.values()) == set(range(1, 21))

    def test_known_assignments_match_draw_zones_order(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_NUMBER
        # L-Flank first 6 zones
        assert GUARDIOLA_ZONE_NUMBER["pg_lf_d_box"] == 1
        assert GUARDIOLA_ZONE_NUMBER["pg_lf_a_box"] == 6
        # R-Flank next 6 zones
        assert GUARDIOLA_ZONE_NUMBER["pg_rf_d_box"] == 7
        assert GUARDIOLA_ZONE_NUMBER["pg_rf_a_box"] == 12
        # GK boxes last
        assert GUARDIOLA_ZONE_NUMBER["pg_gk_def"] == 19
        assert GUARDIOLA_ZONE_NUMBER["pg_gk_att"] == 20


class TestGuardiolaZoneBounds:
    def test_all_20_zones_have_bounds(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_BOUNDS
        assert len(GUARDIOLA_ZONE_BOUNDS) == 20

    def test_bounds_are_4_tuples(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_BOUNDS
        for label, bounds in GUARDIOLA_ZONE_BOUNDS.items():
            assert len(bounds) == 4, f"{label} should have 4 bounds"
            x_min, x_max, y_min, y_max = bounds
            assert x_min < x_max, f"{label}: x_min >= x_max"
            assert y_min < y_max, f"{label}: y_min >= y_max"

    def test_gk_def_spans_inner_lanes(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_BOUNDS
        x_min, x_max, y_min, y_max = GUARDIOLA_ZONE_BOUNDS["pg_gk_def"]
        assert x_min == pytest.approx(-52.5)
        assert x_max == pytest.approx(-36.0)
        assert y_min == pytest.approx(-20.16)
        assert y_max == pytest.approx(20.16)

    def test_bounds_keys_match_zone_number_keys(self) -> None:
        from turf.zone import GUARDIOLA_ZONE_BOUNDS, GUARDIOLA_ZONE_NUMBER
        assert set(GUARDIOLA_ZONE_BOUNDS.keys()) == set(GUARDIOLA_ZONE_NUMBER.keys())
