"""Apex-748M / 776M Production Classifier Package.

Zero-token, non-autoregressive decision & classifier model with sandboxed AST math engine
and Reinforcement Learning with Verifiable Rewards (RLVR).
"""

from apex_classifier_750m.config import Apex748MConfig
from apex_classifier_750m.ast_sandbox import SandboxedMathEvaluator, SecurityError
from apex_classifier_750m.model import Apex748MClassifier
from apex_classifier_750m.dataset import SyntheticDecisionDataset

__all__ = [
    "Apex748MConfig",
    "SandboxedMathEvaluator",
    "SecurityError",
    "Apex748MClassifier",
    "SyntheticDecisionDataset",
]
