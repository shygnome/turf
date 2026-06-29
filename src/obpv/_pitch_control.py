from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from obpv._base import PitchSurfaceModel
from obpv._grid import PitchGrid
from obpv._state import GameState, PlayerState
from obpv._surface import PitchSurface


@dataclass(frozen=True)
class PPCFParameters:
    max_player_acceleration: float = 7.0
    max_player_speed: float = 5.0
    reaction_time: float = 0.7
    tti_sigma: float = 0.45
    kappa_def: float = 1.0
    lambda_att: float = 4.3
    lambda_def: float = 4.3 * 1.0   # 4.3 * kappa_def
    lambda_gk: float = 4.3 * 3.0    # lambda_def * 3.0
    average_ball_speed: float = 15.0
    int_dt: float = 0.04
    max_int_time: float = 10.0
    model_converge_tol: float = 0.01
    time_to_control_veto: float = 3.0

    @property
    def time_to_control_att(self) -> float:
        return float(
            self.time_to_control_veto * np.log(10) * (
                np.sqrt(3.0) * self.tti_sigma / np.pi + 1.0 / self.lambda_att
            )
        )

    @property
    def time_to_control_def(self) -> float:
        return float(
            self.time_to_control_veto * np.log(10) * (
                np.sqrt(3.0) * self.tti_sigma / np.pi + 1.0 / self.lambda_def
            )
        )


class PPCFPitchControlModel(PitchSurfaceModel):
    def __init__(
        self,
        params: PPCFParameters | None = None,
        apply_offside_check: bool = False,
    ) -> None:
        self.params = params or PPCFParameters()
        self.apply_offside_check = apply_offside_check

    @property
    def name(self) -> str:
        return "ppcf_pitch_control"

    def get_config(self) -> dict[str, object]:
        return {
            "max_player_speed": self.params.max_player_speed,
            "reaction_time": self.params.reaction_time,
            "tti_sigma": self.params.tti_sigma,
            "lambda_att": self.params.lambda_att,
            "lambda_def": self.params.lambda_def,
            "average_ball_speed": self.params.average_ball_speed,
            "int_dt": self.params.int_dt,
            "max_int_time": self.params.max_int_time,
            "model_converge_tol": self.params.model_converge_tol,
            "time_to_control_veto": self.params.time_to_control_veto,
            "apply_offside_check": self.apply_offside_check,
        }

    def compute(self, state: GameState, grid: PitchGrid) -> PitchSurface:
        attacking_players = state.attacking_players
        defending_players = state.defending_players

        if len(attacking_players) == 0:
            raise ValueError("GameState has no attacking players")
        if len(defending_players) == 0:
            raise ValueError("GameState has no defending players")

        if self.apply_offside_check:
            attacking_players = self._filter_offside_attackers(
                attacking_players=attacking_players,
                defending_players=defending_players,
                ball_position=state.ball_position,
            )

        values = self._compute_vectorized(
            attacking_players=attacking_players,
            defending_players=defending_players,
            ball_start_pos=state.ball_position,
            grid=grid,
        )

        return PitchSurface(
            values=values,
            grid=grid,
            name=self.name,
            metadata={
                "frame_id": state.frame_id,
                "attacking_team_id": state.attacking_team_id,
                "defending_team_id": state.defending_team_id,
            },
        )

    def _ttis_all_cells(
        self,
        players: list[PlayerState],
        targets: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return (n_players, n_cells) TTI array using vectorised distance calc."""
        n = len(players)
        n_cells = targets.shape[0]
        if n == 0:
            return np.empty((0, n_cells), dtype=float)
        react_positions = np.stack([
            p.position + p.velocity * self.params.reaction_time for p in players
        ])  # (n_players, 2)
        diff = targets[None, :, :] - react_positions[:, None, :]  # (n, n_cells, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (n, n_cells)
        return self.params.reaction_time + dists / self.params.max_player_speed

    def _compute_vectorized(
        self,
        attacking_players: list[PlayerState],
        defending_players: list[PlayerState],
        ball_start_pos: npt.NDArray[np.float64] | None,
        grid: PitchGrid,
    ) -> npt.NDArray[np.float64]:
        """Vectorised PPCF over all grid cells simultaneously."""
        xx, yy = grid.meshgrid
        targets = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_cells, 2)
        n_cells = len(targets)

        if ball_start_pos is None or np.any(np.isnan(ball_start_pos)):
            ball_times = np.zeros(n_cells, dtype=float)
        else:
            ball_times = (
                np.linalg.norm(targets - ball_start_pos, axis=1)
                / self.params.average_ball_speed
            )  # (n_cells,)

        tti_att = self._ttis_all_cells(attacking_players, targets)  # (n_att, n_cells)
        tti_def = self._ttis_all_cells(defending_players, targets)  # (n_def, n_cells)

        tau_min_att = tti_att.min(axis=0) if len(attacking_players) else np.full(n_cells, np.inf)
        tau_min_def = tti_def.min(axis=0) if len(defending_players) else np.full(n_cells, np.inf)

        gap_att = tau_min_att - np.maximum(ball_times, tau_min_def)
        gap_def = tau_min_def - np.maximum(ball_times, tau_min_att)

        def_wins = gap_att >= self.params.time_to_control_def  # defender fully controls
        att_wins = gap_def >= self.params.time_to_control_att  # attacker fully controls

        ppcf_att = np.where(att_wins, 1.0, 0.0).astype(float)

        integrate_mask = ~def_wins & ~att_wins
        if integrate_mask.any():
            idx = np.where(integrate_mask)[0]
            ppcf_att[idx] = self._integrate_cells(
                tti_att[:, idx], tti_def[:, idx], ball_times[idx]
            )

        return ppcf_att.reshape(grid.shape)

    def _integrate_cells(
        self,
        tti_att: npt.NDArray[np.float64],
        tti_def: npt.NDArray[np.float64],
        ball_times: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Euler integration of PPCF for a batch of cells simultaneously."""
        n_cells = tti_att.shape[1]
        t0 = float(ball_times.min()) - self.params.int_dt
        n_steps = int(self.params.max_int_time / self.params.int_dt)
        sigma = self.params.tti_sigma
        lam_att = self.params.lambda_att
        lam_def = self.params.lambda_def
        dt = self.params.int_dt
        tol = self.params.model_converge_tol

        ppcf_att = np.zeros(n_cells, dtype=float)
        ppcf_def = np.zeros(n_cells, dtype=float)
        coeff = -np.pi / np.sqrt(3.0) / sigma

        for step in range(n_steps):
            T = t0 + step * dt
            free = np.maximum(0.0, 1.0 - ppcf_att - ppcf_def)
            if float(free.max()) < tol:
                break
            p_att = (1.0 / (1.0 + np.exp(coeff * (T - tti_att)))).sum(axis=0)
            p_def = (1.0 / (1.0 + np.exp(coeff * (T - tti_def)))).sum(axis=0)
            ppcf_att += free * p_att * lam_att * dt
            ppcf_def += free * p_def * lam_def * dt

        return np.clip(ppcf_att, 0.0, 1.0)

    def _time_to_intercept(
        self,
        player: PlayerState,
        target_position: npt.NDArray[np.float64],
    ) -> float:
        reaction_position = (
            player.position + player.velocity * self.params.reaction_time
        )
        remaining_distance = float(np.linalg.norm(target_position - reaction_position))
        return self.params.reaction_time + remaining_distance / self.params.max_player_speed  # noqa: E501

    def _probability_intercept_ball(
        self,
        T: float,
        time_to_intercept: float,
    ) -> float:
        exponent = (
            -np.pi / np.sqrt(3.0) / self.params.tti_sigma * (T - time_to_intercept)
        )
        return float(1.0 / (1.0 + np.exp(exponent)))

    def _pitch_control_at_target(
        self,
        target_position: npt.NDArray[np.float64],
        attacking_players: list[PlayerState],
        defending_players: list[PlayerState],
        ball_start_pos: npt.NDArray[np.float64] | None,
    ) -> tuple[float, float]:
        if ball_start_pos is None or np.any(np.isnan(ball_start_pos)):
            ball_travel_time = 0.0
        else:
            ball_travel_time = float(
                np.linalg.norm(target_position - ball_start_pos)
                / self.params.average_ball_speed
            )

        att_ttis = np.array(
            [self._time_to_intercept(p, target_position) for p in attacking_players],
            dtype=float,
        )
        def_ttis = np.array(
            [self._time_to_intercept(p, target_position) for p in defending_players],
            dtype=float,
        )

        tau_min_att = np.min(att_ttis)
        tau_min_def = np.min(def_ttis)

        gap_att = tau_min_att - max(ball_travel_time, tau_min_def)
        gap_def = tau_min_def - max(ball_travel_time, tau_min_att)
        if gap_att >= self.params.time_to_control_def:
            return 0.0, 1.0
        if gap_def >= self.params.time_to_control_att:
            return 1.0, 0.0

        valid_att = [
            (player, tti)
            for player, tti in zip(attacking_players, att_ttis, strict=False)
            if tti - tau_min_att < self.params.time_to_control_att
        ]
        valid_def = [
            (player, tti)
            for player, tti in zip(defending_players, def_ttis, strict=False)
            if tti - tau_min_def < self.params.time_to_control_def
        ]

        dT_array = np.arange(
            ball_travel_time - self.params.int_dt,
            ball_travel_time + self.params.max_int_time,
            self.params.int_dt,
        )

        ppcf_att = np.zeros_like(dT_array)
        ppcf_def = np.zeros_like(dT_array)

        player_att_contrib = np.zeros(len(valid_att), dtype=float)
        player_def_contrib = np.zeros(len(valid_def), dtype=float)

        ptot = 0.0
        i = 1

        while 1.0 - ptot > self.params.model_converge_tol and i < dT_array.size:
            T = dT_array[i]

            for k, (_, tti) in enumerate(valid_att):
                dppcf_dt = (
                    (1.0 - ppcf_att[i - 1] - ppcf_def[i - 1])
                    * self._probability_intercept_ball(T, tti)
                    * self.params.lambda_att
                )
                if dppcf_dt < 0.0:
                    raise ValueError("Invalid attacking player probability increment")
                player_att_contrib[k] += dppcf_dt * self.params.int_dt
                ppcf_att[i] += player_att_contrib[k]

            for k, (_, tti) in enumerate(valid_def):
                dppcf_dt = (
                    (1.0 - ppcf_att[i - 1] - ppcf_def[i - 1])
                    * self._probability_intercept_ball(T, tti)
                    * self.params.lambda_def
                )
                if dppcf_dt < 0.0:
                    raise ValueError("Invalid defending player probability increment")
                player_def_contrib[k] += dppcf_dt * self.params.int_dt
                ppcf_def[i] += player_def_contrib[k]

            ptot = ppcf_att[i] + ppcf_def[i]
            i += 1

        return float(ppcf_att[i - 1]), float(ppcf_def[i - 1])

    def _filter_offside_attackers(
        self,
        attacking_players: list[PlayerState],
        defending_players: list[PlayerState],
        ball_position: npt.NDArray[np.float64],
        tol: float = 0.2,
    ) -> list[PlayerState]:
        if len(defending_players) < 2:
            return attacking_players

        defending_x = np.array([p.x for p in defending_players], dtype=float)
        ball_x = float(ball_position[0])

        if np.mean(defending_x) > 0:
            second_deepest_defender_x = np.sort(defending_x)[1]
            offside_line = max(second_deepest_defender_x, ball_x, 0.0) + tol
            return [p for p in attacking_players if p.x <= offside_line]
        else:
            second_deepest_defender_x = np.sort(defending_x)[-2]
            offside_line = min(second_deepest_defender_x, ball_x, 0.0) - tol
            return [p for p in attacking_players if p.x >= offside_line]

