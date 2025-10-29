#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Expert registry defining zone-specific expert configurations and checkpoint mapping.
Zones: hippocampus (memory), amygdala (emotion), hypothalamus (homeostasis), thalamus (routing), language (LLM experts)
"""

from typing import Dict, List, Optional, Tuple
import os

ZoneName = str

# Default expert configurations per zone (types list defines order)
DEFAULT_ZONE_CONFIG: Dict[ZoneName, List[str]] = {
    # Memory-oriented
    "hippocampus": ["mlp", "conv1d", "rational", "code", "self_improve", "mlp"],
    # Emotion-oriented
    "amygdala": ["rational", "mlp", "self_improve"],
    # Homeostatic/regulatory
    "hypothalamus": ["mlp", "conv1d", "self_improve"],
    # Routing/attention-like experts
    "thalamus": ["rational", "mlp", "mlp", "mlp"],
    # Language-specific (placeholder for LLM)
    "language": ["mlp", "code", "mlp", "code", "rational", "self_improve"],
}

# Optional: map each zone+index to a checkpoint path
# Example structure: {("hippocampus", 0): "checkpoints/hippo/exp0.msgpack", ...}
CheckpointMap = Dict[Tuple[ZoneName, int], str]


def get_zone_expert_types(zone: ZoneName, override: Optional[List[str]] = None) -> List[str]:
    """Return expert type list for a given zone, with optional override."""
    if override is not None and len(override) > 0:
        return list(override)
    return list(DEFAULT_ZONE_CONFIG.get(zone, DEFAULT_ZONE_CONFIG["hippocampus"]))


def build_core_kwargs_for_zone(zone: ZoneName, hidden_dim: int, override_types: Optional[List[str]] = None,
                               freeze_experts: bool = False) -> dict:
    """Return kwargs to construct EnhancedSpikingRetrievalCore from a zone registry entry."""
    types = get_zone_expert_types(zone, override_types)
    return {
        "hidden_dim": hidden_dim,
        "num_experts": len(types),
        "expert_dim": 64,
        "phasor_harmonics": 32,
        "expert_types": tuple(types),
        "freeze_experts": freeze_experts,
    }


def expert_ckpt_path(zone: ZoneName, idx: int, root: str = "checkpoints") -> str:
    """Default checkpoint path for a given zone's expert index."""
    return os.path.join(root, zone, f"expert_{idx}.msgpack")
