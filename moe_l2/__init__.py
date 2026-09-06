"""moe-l2: MoE inference L2 hot-cache scheduler."""

from .predictor import (
    DOMAINS,
    domain_to_expert_ids,
    enable_semantic,
    get_backbone_experts,
    get_layer_specificity,
    get_preload_set,
    is_semantic_available,
    load_mapping,
    predict,
    predict_hybrid,
)

__version__ = "0.12.1"
__all__ = [
    "DOMAINS",
    "load_mapping",
    "predict",
    "predict_hybrid",
    "domain_to_expert_ids",
    "get_preload_set",
    "get_backbone_experts",
    "get_layer_specificity",
    "enable_semantic",
    "is_semantic_available",
]
