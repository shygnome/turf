"""obpv â€” Off-Ball Positioning Value for pass evaluation.

Public API
----------
GameState, PlayerState      build frame snapshots from tracking data
PitchGrid                   configure the computation grid
PPCFParameters              tune the PPCF pitch-control model
OBPVAnalyzer                compute OBPV for a single frame
FrameAnalysis               result of OBPVAnalyzer.analyze()
compute_pass_obpv           OBPV gain = end snapshot âˆ’ start snapshot

Transition models (optional, pass to OBPVAnalyzer directly)
-----------------------------------------------------------
TransitionGaussModel        Gaussian kernel, no data files (default)
CsvTransitionModel          single pre-computed 64Ã—100 transition CSV
KernelTransitionModel       18-zone KDE, requires Area*.csv files
"""

from obpv._grid import PitchGrid
from obpv._pitch_control import PPCFParameters
from obpv._state import GameState, PlayerState
from obpv._transition import (
    CsvTransitionModel,
    DummyTransitionModel,
    KernelTransitionModel,
    TransitionGaussModel,
)
from obpv.analyzer import FrameAnalysis, OBPVAnalyzer
from obpv.pass_obpv import compute_pass_obpv

__all__ = [
    "GameState",
    "PlayerState",
    "PitchGrid",
    "PPCFParameters",
    "OBPVAnalyzer",
    "FrameAnalysis",
    "compute_pass_obpv",
    "TransitionGaussModel",
    "CsvTransitionModel",
    "KernelTransitionModel",
    "DummyTransitionModel",
]

