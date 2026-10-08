# Copyright 2026 Slyatski Ilya
# Licensed under the Apache License, Version 2.0
#
# Interpretability infrastructure for JEPA vs baseline comparison
# v0.9.0: + Interpretability Index, robustness, ground truth validation
#
# Module structure (20 modules):
#   sae.py                  — SparseAutoencoder (TopK, dead feature resampling, SAETrainer)
#   structural_probe.py     — StructuralProbe (Hewitt & Manning 2019)
#   causal_intervention.py  — direction_ablation, feature_steering, activation_patching
#   disentanglement.py      — DCI, MIG, SAP, Modularity scores
#   compare.py              — RepresentationComparator, full_comparison_report
#   probing_complexity.py   — ProbingComplexityCurve, LinguisticProbeTasks
#   polysemanticity.py      — PolysemanticityIndex, SuperpositionIndex, FeatureDeduplicationScore
#   causal_scrubbing.py     — CausalScrubber, FeatureHypothesis, InterventionPredictabilityScorer
#   representation_geometry.py — RepresentationGeometry, GeometryDegradationTest
#   feature_composition.py  — FeatureCompositionScore, FeatureInterferenceScore
#   statistical_tests.py    — BootstrapCI, PairedPermutationTest, MultipleComparisonCorrection,
#                              EffectSize, BayesianComparison, MetricComparisonReport
#   information_theory.py   — MINEEstimator, InfoNCEEstimator, ConditionalMIEstimator,
#                              RepresentationCompression, InformationPlane
#   layer_analysis.py       — LayerwiseProbe, LayerwiseCKA, LayerwiseGeometry, LayerRoutingAnalysis
#   stability.py            — TrainingStability, LossStability, EarlyStoppingAdvantage,
#                              CheckpointConsistency
#   probe_generalization.py — ProbeGeneralizationTest, ProbeSelectivityTest,
#                              StructuralProbeGeneralization

from .ablation import (
    ABLATION_CONFIGS,
    AblatedModel,
    AblationConfig,
    AblationStudy,
)
from .causal_intervention import (
    CausalIntervention,
    activation_patching,
    direction_ablation,
    feature_steering,
    intervention_predictability_score,
)
from .causal_scrubbing import (
    CausalScrubber,
    FeatureHypothesis,
    InterventionPredictabilityScorer,
)
from .compare import RepresentationComparator, extract_linguistic_features
from .disentanglement import (
    DCIMetrics,
    MIGScore,
    ModularityScore,
    SAPScore,
    compute_all_disentanglement_metrics,
)
from .feature_composition import (
    FeatureCompositionScore,
    FeatureInterferenceScore,
)
from .ground_truth import (
    GroundTruthValidation,
    SyntheticStructuredModel,
)
from .information_theory import (
    ConditionalMIEstimator,
    InfoNCEEstimator,
    InformationPlane,
    MINEEstimator,
    RepresentationCompression,
)
from .interpretability_index import InterpretabilityIndex
from .layer_analysis import (
    LayerRoutingAnalysis,
    LayerwiseCKA,
    LayerwiseGeometry,
    LayerwiseProbe,
)
from .polysemanticity import (
    FeatureDeduplicationScore,
    PolysemanticityIndex,
    SuperpositionIndex,
)
from .probe_generalization import (
    ProbeGeneralizationTest,
    ProbeSelectivityTest,
    StructuralProbeGeneralization,
)
from .probing_complexity import LinguisticProbeTasks, ProbingComplexityCurve
from .representation_geometry import (
    GeometryDegradationTest,
    RepresentationGeometry,
)
from .robustness import (
    RepresentationRobustness,
    RobustnessBattery,
    span_corruption,
    token_dropout,
    token_permutation,
    token_substitution,
)
from .sae import SAETrainer, SparseAutoencoder
from .scaling import (
    ComputeOptimalScale,
    InterpretabilityEfficiency,
    ScalingAnalysis,
)
from .stability import (
    CheckpointConsistency,
    EarlyStoppingAdvantage,
    LossStability,
    TrainingStability,
)
from .statistical_tests import (
    BayesianComparison,
    BootstrapCI,
    EffectSize,
    MetricComparisonReport,
    MultipleComparisonCorrection,
    PairedPermutationTest,
)
from .structural_probe import StructuralProbe
from .visualization import (
    ablation_comparison_chart,
    bar_chart_with_errors,
    convergence_plot,
    information_plane,
    layer_heatmap,
    probing_complexity_curve,
    radar_chart,
    robustness_curve,
    scaling_law_plot,
)

__all__ = [
    "ABLATION_CONFIGS",
    "AblatedModel",
    # Ablation
    "AblationConfig",
    "AblationStudy",
    "BayesianComparison",
    # Statistical tests
    "BootstrapCI",
    "CausalIntervention",
    # Causal scrubbing
    "CausalScrubber",
    "CheckpointConsistency",
    "ComputeOptimalScale",
    "ConditionalMIEstimator",
    # Disentanglement
    "DCIMetrics",
    "EarlyStoppingAdvantage",
    "EffectSize",
    # Composition
    "FeatureCompositionScore",
    "FeatureDeduplicationScore",
    "FeatureHypothesis",
    "FeatureInterferenceScore",
    "GeometryDegradationTest",
    "GroundTruthValidation",
    "InfoNCEEstimator",
    "InformationPlane",
    "InterpretabilityEfficiency",
    # Interpretability Index
    "InterpretabilityIndex",
    "InterventionPredictabilityScorer",
    "LayerRoutingAnalysis",
    "LayerwiseCKA",
    "LayerwiseGeometry",
    # Layer analysis
    "LayerwiseProbe",
    "LinguisticProbeTasks",
    "LossStability",
    "MIGScore",
    # Information theory
    "MINEEstimator",
    "MetricComparisonReport",
    "ModularityScore",
    "MultipleComparisonCorrection",
    "PairedPermutationTest",
    # Polysemanticity
    "PolysemanticityIndex",
    # Probe generalization
    "ProbeGeneralizationTest",
    "ProbeSelectivityTest",
    # Probing complexity
    "ProbingComplexityCurve",
    # Comparator
    "RepresentationComparator",
    "RepresentationCompression",
    # Geometry
    "RepresentationGeometry",
    # Robustness
    "RepresentationRobustness",
    "RobustnessBattery",
    "SAETrainer",
    "SAPScore",
    # Scaling
    "ScalingAnalysis",
    # SAE
    "SparseAutoencoder",
    # Structural
    "StructuralProbe",
    "StructuralProbeGeneralization",
    "SuperpositionIndex",
    # Ground truth
    "SyntheticStructuredModel",
    # Stability
    "TrainingStability",
    "ablation_comparison_chart",
    "activation_patching",
    "bar_chart_with_errors",
    "compute_all_disentanglement_metrics",
    "convergence_plot",
    # Causal intervention
    "direction_ablation",
    "extract_linguistic_features",
    "feature_steering",
    "information_plane",
    "intervention_predictability_score",
    "layer_heatmap",
    "probing_complexity_curve",
    # Visualization
    "radar_chart",
    "robustness_curve",
    "scaling_law_plot",
    "span_corruption",
    "token_dropout",
    "token_permutation",
    "token_substitution",
]
