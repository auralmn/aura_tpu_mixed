#!/usr/bin/env python3
"""
AURA Neuromorphic LLM Configuration
Advanced configuration for neuromorphic language models with consciousness and neurogenesis
"""

import torch
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class NeuromorphicLLMConfig:
    """Configuration class for AURA Neuromorphic LLM"""
    
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    
    # Neuromorphic parameters
    spike_threshold: float = 0.5
    spike_decay: float = 0.9
    membrane_potential_init: float = 0.0
    refractory_period: int = 2
    
    # Consciousness parameters
    enable_consciousness: bool = True
    consciousness_threshold: float = 0.7
    consciousness_dim: int = 256
    meta_cognitive_layers: int = 3
    awareness_update_rate: float = 0.1
    
    # Neurogenesis parameters
    enable_neurogenesis: bool = True
    neurogenesis_threshold: float = 0.3
    max_experts_per_category: int = 16
    min_experts_per_category: int = 2
    expert_creation_rate: float = 0.05
    expert_pruning_rate: float = 0.01
    
    # Memory and learning
    memory_capacity: int = 10000
    episodic_memory_size: int = 5000
    semantic_memory_size: int = 5000
    working_memory_size: int = 512
    memory_consolidation_rate: float = 0.1
    
    # Knowledge ingestion
    knowledge_buffer_size: int = 50
    ingestion_batch_size: int = 4
    knowledge_integration_rate: float = 0.05
    real_time_learning: bool = True
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 1000
    
    # Device and optimization
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Attention parameters
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Generation parameters
    max_generation_length: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        # Ensure head dimension is valid
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")
        
        # Auto-detect device
        if self.device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        
        # Adjust consciousness dimension if needed
        if self.consciousness_dim > self.hidden_size:
            self.consciousness_dim = self.hidden_size // 2
        
        # Ensure memory sizes are reasonable
        if self.episodic_memory_size + self.semantic_memory_size > self.memory_capacity:
            self.episodic_memory_size = self.memory_capacity // 2
            self.semantic_memory_size = self.memory_capacity // 2
    
    @property
    def head_dim(self) -> int:
        """Dimension of each attention head"""
        return self.hidden_size // self.num_attention_heads
    
    @property
    def total_memory_size(self) -> int:
        """Total memory capacity"""
        return self.episodic_memory_size + self.semantic_memory_size + self.working_memory_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NeuromorphicLLMConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file"""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'NeuromorphicLLMConfig':
        """Load configuration from file"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def get_model_size_estimate(self) -> Dict[str, int]:
        """Estimate model size in parameters"""
        # Embedding parameters
        embedding_params = self.vocab_size * self.hidden_size * 2  # token + position embeddings
        
        # Transformer layer parameters
        attention_params = 4 * self.hidden_size * self.hidden_size  # Q, K, V, O projections
        mlp_params = 2 * self.hidden_size * self.intermediate_size  # up and down projections
        layer_norm_params = 2 * self.hidden_size  # 2 layer norms per layer
        layer_params = (attention_params + mlp_params + layer_norm_params) * self.num_layers
        
        # Output head parameters
        output_params = self.hidden_size * self.vocab_size
        
        # Consciousness parameters
        consciousness_params = 0
        if self.enable_consciousness:
            consciousness_params = (
                self.hidden_size * self.consciousness_dim +
                self.consciousness_dim * self.consciousness_dim * (self.meta_cognitive_layers - 1) +
                self.consciousness_dim * self.hidden_size
            )
        
        # Neurogenesis parameters (estimated)
        neurogenesis_params = 0
        if self.enable_neurogenesis:
            avg_experts = (self.max_experts_per_category + self.min_experts_per_category) // 2
            neurogenesis_params = avg_experts * 8 * self.hidden_size * self.intermediate_size  # 8 categories
        
        total_params = (
            embedding_params + 
            layer_params + 
            output_params + 
            consciousness_params + 
            neurogenesis_params
        )
        
        return {
            "embedding_params": embedding_params,
            "transformer_params": layer_params,
            "output_params": output_params,
            "consciousness_params": consciousness_params,
            "neurogenesis_params": neurogenesis_params,
            "total_params": total_params,
            "total_params_millions": total_params / 1_000_000
        }
    
    def __str__(self) -> str:
        """String representation of config"""
        size_info = self.get_model_size_estimate()
        return f"""NeuromorphicLLMConfig(
    Model: {size_info['total_params_millions']:.1f}M parameters
    Architecture: {self.num_layers} layers, {self.num_attention_heads} heads, {self.hidden_size} hidden
    Neuromorphic: spike_threshold={self.spike_threshold}
    Consciousness: {'enabled' if self.enable_consciousness else 'disabled'} (threshold={self.consciousness_threshold})
    Neurogenesis: {'enabled' if self.enable_neurogenesis else 'disabled'} (threshold={self.neurogenesis_threshold})
    Memory: {self.memory_capacity} total capacity
    Device: {self.device}
)"""


# Predefined configurations for different use cases
class NeuromorphicLLMConfigs:
    """Predefined configurations for different AURA model sizes and use cases"""
    
    @staticmethod
    def small() -> NeuromorphicLLMConfig:
        """Small AURA model for testing and development"""
        return NeuromorphicLLMConfig(
            vocab_size=16000,
            hidden_size=512,
            num_layers=6,
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=1024,
            memory_capacity=5000,
            knowledge_buffer_size=25
        )
    
    @staticmethod
    def medium() -> NeuromorphicLLMConfig:
        """Medium AURA model for general use"""
        return NeuromorphicLLMConfig(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            memory_capacity=10000,
            knowledge_buffer_size=50
        )
    
    @staticmethod
    def large() -> NeuromorphicLLMConfig:
        """Large AURA model for advanced applications"""
        return NeuromorphicLLMConfig(
            vocab_size=50000,
            hidden_size=1024,
            num_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            max_position_embeddings=4096,
            memory_capacity=20000,
            knowledge_buffer_size=100,
            consciousness_dim=512,
            meta_cognitive_layers=5
        )
    
    @staticmethod
    def research() -> NeuromorphicLLMConfig:
        """Research configuration with all features enabled"""
        return NeuromorphicLLMConfig(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            enable_consciousness=True,
            enable_neurogenesis=True,
            consciousness_threshold=0.6,
            neurogenesis_threshold=0.25,
            memory_capacity=15000,
            real_time_learning=True,
            knowledge_buffer_size=75
        )
    
    @staticmethod
    def production() -> NeuromorphicLLMConfig:
        """Production configuration optimized for deployment"""
        return NeuromorphicLLMConfig(
            vocab_size=32000,
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=2048,
            enable_consciousness=True,
            enable_neurogenesis=False,  # Disabled for stability
            consciousness_threshold=0.7,
            memory_capacity=8000,
            mixed_precision=True,
            gradient_checkpointing=True,
            knowledge_buffer_size=30
        )


def create_neuromorphic_llm_config(
    model_size: str = "medium",
    custom_overrides: Optional[Dict[str, Any]] = None
) -> NeuromorphicLLMConfig:
    """
    Create a neuromorphic LLM configuration
    
    Args:
        model_size: One of 'small', 'medium', 'large', 'research', 'production'
        custom_overrides: Dictionary of custom parameter overrides
    
    Returns:
        NeuromorphicLLMConfig instance
    """
    # Get base configuration
    if model_size == "small":
        config = NeuromorphicLLMConfigs.small()
    elif model_size == "medium":
        config = NeuromorphicLLMConfigs.medium()
    elif model_size == "large":
        config = NeuromorphicLLMConfigs.large()
    elif model_size == "research":
        config = NeuromorphicLLMConfigs.research()
    elif model_size == "production":
        config = NeuromorphicLLMConfigs.production()
    else:
        raise ValueError(f"Unknown model size: {model_size}")
    
    # Apply custom overrides
    if custom_overrides:
        for key, value in custom_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    return config
