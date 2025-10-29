# SPDX-License-Identifier: Apache-2.0
"""
AURA Neuromorphic Transformer Block
Combines spiking attention with feed-forward processing
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from aura.neural.spiking_attention import SpikingMultiHeadAttention
from aura.neural.neuromorphic_tokenizer import SpikingLayerNorm
from aura.neural.interfaces import AURAModule

# Import your existing SRFFN
try:
    from aura.neural.srffn import SRFFN
except ImportError:
    # Fallback implementation if SRFFN is not available
    class SRFFN(nn.Module):
        def __init__(self, module_id: str, config: Dict[str, Any]):
            super().__init__()
            self.module_id = module_id
            self.config = config
            input_size = config.get('input_size', 512)
            hidden_size = config.get('hidden_size', 2048)
            output_size = config.get('output_size', 512)
            
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear2 = nn.Linear(hidden_size, output_size)
            self.activation = nn.ReLU()
            
        def initialize(self) -> bool:
            return True
            
        def process(self, input_data: Any) -> Any:
            if isinstance(input_data, torch.Tensor):
                x = self.linear1(input_data)
                x = self.activation(x)
                x = self.linear2(x)
                return x
            return input_data

class NeuromorphicTransformerBlock(AURAModule):
    """
    Single transformer block with spiking attention and neuromorphic processing
    Integrates with your existing AURA components
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        self.hidden_size = config.get('hidden_size', 512)
        
        # Spiking attention mechanism
        self.attention = SpikingMultiHeadAttention(f'{module_id}_attention', config)
        
        # Layer normalization for spikes
        self.attention_layer_norm = SpikingLayerNorm(self.hidden_size)
        self.ff_layer_norm = SpikingLayerNorm(self.hidden_size)
        
        # Feed-forward network - use your existing SRFFN
        ff_config = {
            'input_size': self.hidden_size,
            'hidden_size': config.get('intermediate_size', self.hidden_size * 4),
            'output_size': self.hidden_size,
            'spike_threshold': config.get('spike_threshold', 0.5)
        }
        self.feed_forward = SRFFN(f'{module_id}_srffn', ff_config)
        
        # Residual connection scaling for spikes
        self.residual_alpha = nn.Parameter(torch.tensor(0.5))
        
    def initialize(self) -> bool:
        """Initialize transformer block"""
        try:
            success = True
            success &= self.attention.initialize()
            success &= self.feed_forward.initialize()
            
            self.state = 'active'
            self.logger.info(f"NeuromorphicTransformerBlock initialized")
            return success
        except Exception as e:
            self.logger.error(f"TransformerBlock initialization failed: {e}")
            return False
    
    def forward(self, spike_input: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through neuromorphic transformer block
        
        Args:
            spike_input: (batch, seq_len, hidden_size) spike sequences
            attention_mask: (batch, seq_len) attention mask
            
        Returns:
            Dict with processed spikes and intermediate results
        """
        # 1. Self-attention with residual connection
        attention_result = self.attention(spike_input, attention_mask)
        attention_output = attention_result['attention_output']
        
        # Residual connection with learnable scaling
        attention_residual = self.residual_alpha * attention_output + (1 - self.residual_alpha) * spike_input
        attention_normalized = self.attention_layer_norm(attention_residual)
        
        # 2. Feed-forward with residual connection  
        ff_output = self.feed_forward.process(attention_normalized)
        if isinstance(ff_output, dict):
            ff_spikes = ff_output.get('spikes', ff_output.get('output', attention_normalized))
        else:
            ff_spikes = ff_output
            
        # Residual connection
        ff_residual = self.residual_alpha * ff_spikes + (1 - self.residual_alpha) * attention_normalized
        output_spikes = self.ff_layer_norm(ff_residual)
        
        return {
            'hidden_states': output_spikes,
            'attention_output': attention_output,
            'attention_weights': attention_result.get('attention_weights'),
            'spike_rate': torch.mean(output_spikes).item(),
            'attention_stats': attention_result
        }
    
    def process(self, input_data: Any) -> Any:
        """AURAModule interface"""
        if isinstance(input_data, torch.Tensor):
            return self.forward(input_data)
        elif isinstance(input_data, dict):
            spike_input = input_data.get('spike_sequences', input_data.get('input'))
            attention_mask = input_data.get('attention_mask')
            return self.forward(spike_input, attention_mask)
        else:
            raise ValueError("Input must be tensor or dict with spike sequences")
    
    def get_state(self) -> Dict[str, Any]:
        """Get block state for serialization"""
        return {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'attention_state': self.attention.get_state(),
            'ff_state': self.feed_forward.get_state() if hasattr(self.feed_forward, 'get_state') else {}
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set block state from serialization"""
        try:
            self.load_state_dict(state['model_state_dict'])
            success = self.attention.set_state(state['attention_state'])
            if 'ff_state' in state and hasattr(self.feed_forward, 'set_state'):
                success &= self.feed_forward.set_state(state['ff_state'])
            return success
        except Exception as e:
            self.logger.error(f"Failed to set transformer block state: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate transformer block functionality"""
        try:
            test_input = torch.randint(0, 2, (2, 32, self.hidden_size)).float()
            test_mask = torch.ones(2, 32)
            
            result = self.forward(test_input, test_mask)
            
            if 'hidden_states' not in result:
                return False, "Missing hidden_states in result"
                
            output = result['hidden_states']
            if output.shape != test_input.shape:
                return False, f"Output shape mismatch: {output.shape} vs {test_input.shape}"
            
            return True, "Transformer block validation successful"
        except Exception as e:
            return False, f"Transformer block validation error: {str(e)}"

def create_transformer_block(config: Dict[str, Any]) -> NeuromorphicTransformerBlock:
    """Factory function to create transformer block"""
    return NeuromorphicTransformerBlock('neuromorphic_block', config)

def test_transformer_block():
    """Test the complete transformer block"""
    config = {
        'hidden_size': 256,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
        'spike_threshold': 0.5,
        'attention_temperature': 1.0
    }
    
    block = create_transformer_block(config)
    assert block.initialize() == True
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    spike_input = torch.randint(0, 2, (batch_size, seq_len, config['hidden_size'])).float()
    attention_mask = torch.ones(batch_size, seq_len)
    
    result = block(spike_input, attention_mask)
    
    print(f"Input shape: {spike_input.shape}")
    print(f"Output shape: {result['hidden_states'].shape}")
    print(f"Spike rate: {result['spike_rate']:.3f}")
    print(f"Attention stats: {result['attention_stats']['spike_rate']:.3f}")
    
    # Test validation
    is_valid, message = block.validate()
    print(f"Validation: {is_valid}, {message}")
    
    return block

if __name__ == "__main__":
    test_transformer_block()
