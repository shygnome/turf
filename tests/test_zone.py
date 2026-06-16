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
