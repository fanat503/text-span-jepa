# Copyright 2026 Text-Span JEPA Authors
# Licensed under the Apache License, Version 2.0

from .cgn import ContextualGatingNetwork
from .cmc import CrossMaskConsistency
from .collapse import (
    CollapseDiagnostics,
    CovarianceRegularization,
    TargetCentering,
    VarianceRegularization,
)
from .decoder import TiedTokenDecoder
from .encoder import TextSpanJEPLEncoder
from .gac import GradientAllocatedCapacity
from .jepa import TextSpanJEPA, TextSpanJEPAConfig
from .predictor import TextSpanJEPApredictor
from .puc import PredictionUncertaintyCalibration
from .rdc import RepresentationDriftCompensation
from .spc import SpectralPredictiveCoding
from .sta import SpectralTransportAlignment
from .wsd import WorkspaceSyncDrift
from .wsr import WorkspaceSharpnessRegularization

__all__ = [
    "CollapseDiagnostics",
    "ContextualGatingNetwork",
    "CovarianceRegularization",
    "CrossMaskConsistency",
    "GradientAllocatedCapacity",
    "PredictionUncertaintyCalibration",
    "RepresentationDriftCompensation",
    "SpectralPredictiveCoding",
    "SpectralTransportAlignment",
    "TargetCentering",
    "TiedTokenDecoder",
    "TextSpanJEPA",
    "TextSpanJEPAConfig",
    "TextSpanJEPLEncoder",
    "TextSpanJEPApredictor",
    "VarianceRegularization",
    "WorkspaceSharpnessRegularization",
    "WorkspaceSyncDrift",
]
