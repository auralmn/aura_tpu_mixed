#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
TPU Optimization utilities for AURA
High-impact performance optimizations for TPU training and inference
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, Tuple, Optional
import math
import optax

class DynamicBatchSizer:
    """Dynamically adjust batch size based on sequence length and memory constraints."""
    
    def __init__(self, 
                 base_batch_size: int = 128,
                 max_sequence_length: int = 2048,
                 memory_limit_gb: float = 8.0):
        self.base_batch_size = base_batch_size
        self.max_sequence_length = max_sequence_length
        self.memory_limit_gb = memory_limit_gb
        self.memory_profile = {}
    
    def compute_optimal_batch_size(self, sequence_length: int, model_size: int) -> int:
        """Compute optimal batch size based on sequence length and model parameters."""
        # Memory usage scales roughly as: batch_size * seq_len^2 * model_size
        memory_factor = (sequence_length / self.max_sequence_length) ** 1.5
        model_factor = model_size / 1e6  # Assume 1M parameter baseline
        
        # Adjust batch size inversely to memory requirements
        optimal_batch = int(self.base_batch_size / (memory_factor * model_factor))
        
        # Ensure batch size is power of 2 for TPU efficiency
        optimal_batch = 2 ** int(math.log2(max(1, optimal_batch)))
        
        return min(optimal_batch, self.base_batch_size)
    
    def update_memory_profile(self, sequence_length: int, batch_size: int, 
                            memory_used: float, execution_time: float):
        """Update memory profile for future predictions."""
        key = (sequence_length, batch_size)
        self.memory_profile[key] = {
            'memory_gb': memory_used,
            'time_seconds': execution_time,
            'efficiency': batch_size / execution_time
        }


class ExpertSharding:
    """Shard experts across TPU cores for better memory utilization."""
    
    def __init__(self, num_experts: int, num_cores: int = 8):
        self.num_experts = num_experts
        self.num_cores = num_cores
        self.shard_map = self._compute_shard_mapping()
    
    def _compute_shard_mapping(self) -> Dict[int, int]:
        """Compute optimal expert-to-core mapping."""
        experts_per_core = math.ceil(self.num_experts / self.num_cores)
        shard_map = {}
        
        for expert_idx in range(self.num_experts):
            core_idx = expert_idx // experts_per_core  
            shard_map[expert_idx] = min(core_idx, self.num_cores - 1)
        
        return shard_map
    
    def get_expert_core(self, expert_idx: int) -> int:
        """Get the TPU core assignment for a specific expert."""
        return self.shard_map.get(expert_idx, 0)
    
    def create_sharded_computation(self, experts, inputs):
        """Create sharded computation across TPU cores."""
        def sharded_expert_fn(expert_idx, x):
            core_id = self.get_expert_core(expert_idx)
            with jax.sharding.NamedSharding(f'core_{core_id}'):
                return experts[expert_idx](x)
        
        return jax.vmap(sharded_expert_fn, in_axes=(0, None))(
            jnp.arange(len(experts)), inputs
        )


class GradientCheckpointing:
    """Memory-efficient gradient computation through checkpointing."""
    
    @staticmethod
    def checkpoint_fn(fn):
        """Wrapper to enable gradient checkpointing for a function."""
        return jax.checkpoint(fn)
    
    @staticmethod  
    def create_checkpointed_layer(layer_fn, checkpoint_every_n: int = 2):
        """Create a checkpointed version of a layer."""
        def checkpointed_layer(*args, **kwargs):
            return GradientCheckpointing.checkpoint_fn(layer_fn)(*args, **kwargs)
        return checkpointed_layer


class MixedPrecisionOptimizer:
    """BF16 mixed precision optimization for TPU."""
    
    def __init__(self, loss_scale: float = 2**15):
        self.loss_scale = loss_scale
        self.dtype = jnp.bfloat16
    
    def convert_to_bf16(self, params):
        """Convert parameters to BF16 for forward pass."""
        return jax.tree_map(lambda x: x.astype(self.dtype), params)
    
    def scale_gradients(self, grads):
        """Scale gradients to prevent underflow."""
        return jax.tree_map(lambda g: g * self.loss_scale, grads)
    
    def unscale_gradients(self, grads):
        """Unscale gradients before parameter updates.""" 
        return jax.tree_map(lambda g: g / self.loss_scale, grads)
    
    def create_mixed_precision_train_step(self, model_fn, optimizer):
        """Create a mixed precision training step."""
        
        def train_step(state, batch):
            def loss_fn(params):
                # Forward pass in BF16
                bf16_params = self.convert_to_bf16(params)
                logits = model_fn(bf16_params, batch['inputs'])
                # Loss computation in FP32
                loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                    logits.astype(jnp.float32), batch['targets']
                ))
                return loss * self.loss_scale
            
            # Compute gradients
            scaled_loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = self.unscale_gradients(grads)
            loss = scaled_loss / self.loss_scale
            
            # Update state
            state = state.apply_gradients(grads=grads)
            return state, {'loss': loss}
        
        return train_step


class TPUMemoryProfiler:
    """Profile TPU memory usage and optimize allocation."""
    
    def __init__(self):
        self.memory_snapshots = []
        self.peak_memory = 0.0
    
    def take_snapshot(self, step: int, description: str = ""):
        """Take a memory usage snapshot."""
        # Get current memory usage (this would integrate with TPU profiling APIs)
        try:
            devices = jax.devices()
            memory_info = []
            for device in devices:
                # Placeholder - would use actual TPU memory APIs
                memory_info.append({
                    'device': str(device),
                    'allocated_gb': 0.0,  # Would be actual memory usage
                    'peak_gb': 0.0
                })
            
            snapshot = {
                'step': step,
                'description': description,
                'devices': memory_info
            }
            self.memory_snapshots.append(snapshot)
            
        except Exception as e:
            print(f"Memory profiling failed: {e}")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report."""
        if not self.memory_snapshots:
            return {'error': 'No snapshots available'}
        
        return {
            'total_snapshots': len(self.memory_snapshots),
            'peak_memory_gb': self.peak_memory,
            'snapshots': self.memory_snapshots[-10:]  # Last 10 snapshots
        }


class PipelineParallelism:
    """Implement pipeline parallelism for large models."""
    
    def __init__(self, num_stages: int = 4, num_microbatches: int = 8):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches
    
    def create_pipeline_stages(self, model_layers):
        """Split model into pipeline stages."""
        layers_per_stage = len(model_layers) // self.num_stages
        stages = []
        
        for stage_idx in range(self.num_stages):
            start_idx = stage_idx * layers_per_stage
            end_idx = start_idx + layers_per_stage
            if stage_idx == self.num_stages - 1:
                end_idx = len(model_layers)  # Include remaining layers in last stage
            
            stage_layers = model_layers[start_idx:end_idx]
            stages.append(stage_layers)
        
        return stages
    
    def execute_pipeline(self, stages, inputs):
        """Execute pipelined computation."""
        # Split input into microbatches
        microbatch_size = inputs.shape[0] // self.num_microbatches
        microbatches = [
            inputs[i:i+microbatch_size] 
            for i in range(0, inputs.shape[0], microbatch_size)
        ]
        
        # Pipeline execution (simplified version)
        stage_outputs = []
        for microbatch in microbatches:
            x = microbatch
            for stage in stages:
                for layer in stage:
                    x = layer(x)
            stage_outputs.append(x)
        
        return jnp.concatenate(stage_outputs, axis=0)


class OptimizedTPUConfig:
    """Centralized TPU optimization configuration."""
    
    def __init__(self, 
                 model_size: str = "medium",
                 sequence_length: int = 512,
                 available_memory_gb: float = 32.0):
        
        self.model_size = model_size
        self.sequence_length = sequence_length
        self.available_memory_gb = available_memory_gb
        
        # Initialize optimizers
        self.batch_sizer = DynamicBatchSizer()
        self.expert_sharding = None  # Will be set when experts are known
        self.mixed_precision = MixedPrecisionOptimizer()
        self.profiler = TPUMemoryProfiler()
        
        # Configure based on model size
        self.config = self._get_optimized_config()
    
    def _get_optimized_config(self) -> Dict[str, Any]:
        """Get optimized configuration based on model size."""
        configs = {
            "small": {
                "batch_size": 256,
                "experts_per_core": 2,
                "checkpoint_every_n": 4,
                "pipeline_stages": 2
            },
            "medium": {
                "batch_size": 128,
                "experts_per_core": 4,
                "checkpoint_every_n": 2,
                "pipeline_stages": 4
            },
            "large": {
                "batch_size": 64,
                "experts_per_core": 8,
                "checkpoint_every_n": 1,
                "pipeline_stages": 8
            }
        }
        
        base_config = configs.get(self.model_size, configs["medium"])
        
        # Adjust for sequence length
        if self.sequence_length > 1024:
            base_config["batch_size"] //= 2
            base_config["checkpoint_every_n"] = max(1, base_config["checkpoint_every_n"] // 2)
        
        return base_config
    
    def setup_expert_sharding(self, num_experts: int, num_cores: int = 8):
        """Setup expert sharding configuration."""
        self.expert_sharding = ExpertSharding(num_experts, num_cores)
        self.config["expert_sharding"] = {
            "num_experts": num_experts,
            "num_cores": num_cores,
            "experts_per_core": num_experts // num_cores
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get complete training configuration."""
        training_config = {
            "batch_size": self.batch_sizer.compute_optimal_batch_size(
                self.sequence_length, 
                {"small": 1e6, "medium": 10e6, "large": 100e6}[self.model_size]
            ),
            "mixed_precision": True,
            "gradient_checkpointing": True,
            "checkpoint_every_n": self.config["checkpoint_every_n"],
            "dtype": "bfloat16",
            "optimization_flags": {
                "xla_force_host_platform_device_count": 8,
                "xla_tpu_enable_async_collective_fusion": True,
                "xla_tpu_enable_megascale_barrier": True
            }
        }
        
        return training_config


# Utility functions for easy integration
def create_optimized_training_setup(model_size: str = "medium", 
                                  sequence_length: int = 512,
                                  num_experts: int = 16) -> OptimizedTPUConfig:
    """Create optimized TPU training setup with all optimizations enabled."""
    
    config = OptimizedTPUConfig(model_size, sequence_length)
    config.setup_expert_sharding(num_experts)
    
    print(f"🚀 TPU Optimization Setup Complete:")
    print(f"   Model Size: {model_size}")
    print(f"   Sequence Length: {sequence_length}")
    print(f"   Experts: {num_experts}")
    print(f"   Optimal Batch Size: {config.get_training_config()['batch_size']}")
    print(f"   Mixed Precision: Enabled (BF16)")
    print(f"   Gradient Checkpointing: Enabled")
    print(f"   Expert Sharding: {num_experts} experts across 8 cores")
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_optimized_training_setup("large", 1024, 32)
    training_config = config.get_training_config()
    print("\nTraining Configuration:")
    for key, value in training_config.items():
        print(f"  {key}: {value}")
