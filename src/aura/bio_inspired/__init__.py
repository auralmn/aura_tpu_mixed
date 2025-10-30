from .enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore
from .spiking_attention import SpikingAttentionJAX
from .phasor_bank import PhasorBankJAX
from .merit_board import MeritBoard
from .personality_jax import PersonalityModulator
from .personality_engine import PersonalityEngineJAX
from .thalamic_router import ThalamicGradientBroadcasterJAX
from .experts import MLPExpert, Conv1DExpert, RationalExpert, CodeExpert, SelfImproveExpert

# Consciousness-aware extensions
from .consciousness_aware import (
    ProsodyExtractorJAX,
    PlutchikEmotionEncoderJAX,
    ConsciousnessAwareRetrievalCore,
)
