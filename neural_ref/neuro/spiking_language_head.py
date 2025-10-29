# SPDX-License-Identifier: Apache-2.0
"""
AURA Spiking Language Head
Neural language modeling head for neuromorphic text generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from aura.neural.interfaces import AURAModule
from aura.neural.spiking_attention import SpikingLinear

class SpikingVocabularyProjection(nn.Module):
    """
    Projects spike sequences to vocabulary logits with neuromorphic processing
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, spike_threshold: float = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.spike_threshold = spike_threshold
        
        # Projection layers
        self.pre_projection = SpikingLinear(hidden_size, hidden_size, spike_threshold)
        self.vocab_projection = nn.Linear(hidden_size, vocab_size)  # Final layer is non-spiking
        
        # Adaptive temperature for different tokens
        self.token_temperature = nn.Parameter(torch.ones(vocab_size))
        
        # Initialize vocabulary projection
        nn.init.xavier_uniform_(self.vocab_projection.weight, gain=1.0)
        nn.init.zeros_(self.vocab_projection.bias)
    
    def forward(self, spike_input: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Project spikes to vocabulary logits
        
        Args:
            spike_input: (batch, seq_len, hidden_size) spike sequences
            temperature: Generation temperature
            
        Returns:
            logits: (batch, seq_len, vocab_size) vocabulary logits
        """
        # Pre-process spikes
        processed_spikes = self.pre_projection(spike_input)
        
        # Project to vocabulary (no spiking for final layer)
        logits = self.vocab_projection(processed_spikes)
        
        # Apply adaptive temperature per token
        adaptive_temp = self.token_temperature.unsqueeze(0).unsqueeze(0) * temperature
        adjusted_logits = logits / adaptive_temp
        
        return adjusted_logits

class NeuroSampler(nn.Module):
    """
    Neuromorphic sampling strategies for text generation
    Implements various sampling methods with spike-based probability computation
    """
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Learnable sampling bias (some tokens naturally more likely)
        self.sampling_bias = nn.Parameter(torch.zeros(vocab_size))
        
        # Dynamic top-k/top-p thresholds
        self.dynamic_top_k = nn.Parameter(torch.tensor(50.0))
        self.dynamic_top_p = nn.Parameter(torch.tensor(0.9))
    
    def spike_based_sampling(self, logits: torch.Tensor, method: str = 'top_p', 
                           temperature: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Sample next tokens using neuromorphic principles
        
        Args:
            logits: (batch, vocab_size) or (batch, seq_len, vocab_size)
            method: 'greedy', 'top_k', 'top_p', 'nucleus', 'typical'
            temperature: Sampling temperature
            
        Returns:
            sampled_tokens: (batch,) or (batch, seq_len) next tokens
        """
        if logits.dim() == 3:
            # Take last position for generation
            logits = logits[:, -1, :]
        
        batch_size = logits.size(0)
        
        # Apply sampling bias
        biased_logits = logits + self.sampling_bias.unsqueeze(0)
        
        # Temperature scaling
        scaled_logits = biased_logits / temperature
        
        if method == 'greedy':
            return torch.argmax(scaled_logits, dim=-1)
        
        elif method == 'top_k':
            top_k = kwargs.get('top_k', int(self.dynamic_top_k.item()))
            top_k = min(top_k, self.vocab_size)
            
            # Get top-k tokens
            top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
            
            # Sample from top-k distribution
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            
            # Map back to original vocabulary
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            
        elif method == 'top_p' or method == 'nucleus':
            top_p = kwargs.get('top_p', self.dynamic_top_p.item())
            
            # Sort logits in descending order
            sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
            
            # Compute cumulative probabilities
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            
            # Set logits to -inf for removed tokens
            sorted_logits[sorted_indices_to_remove] = float('-inf')
            
            # Sample from filtered distribution
            probs = F.softmax(sorted_logits, dim=-1)
            sampled_indices = torch.multinomial(probs, 1).squeeze(-1)
            
            # Map back to original vocabulary
            sampled_tokens = torch.gather(sorted_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            
        elif method == 'typical':
            # Typical sampling (entropy-based)
            typical_p = kwargs.get('typical_p', 0.95)
            
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Compute surprisal (negative log probability)
            surprisal = -torch.log(probs + 1e-8)
            
            # Compute entropy
            entropy = torch.sum(probs * surprisal, dim=-1, keepdim=True)
            
            # Compute absolute difference from entropy
            abs_diff = torch.abs(surprisal - entropy)
            
            # Sort by absolute difference
            sorted_diffs, sorted_indices = torch.sort(abs_diff, dim=-1)
            sorted_probs = torch.gather(probs, -1, sorted_indices)
            
            # Cumulative probability until typical_p
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff_mask = cumulative_probs < typical_p
            
            # Keep at least one token
            cutoff_mask[:, 0] = True
            
            # Create filtered probability distribution
            filtered_probs = torch.zeros_like(probs)
            filtered_probs.scatter_(-1, sorted_indices, sorted_probs * cutoff_mask.float())
            filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            sampled_tokens = torch.multinomial(filtered_probs, 1).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        return sampled_tokens

class SpikingLanguageHead(AURAModule):
    """
    Complete language modeling head for neuromorphic text generation
    Handles next-token prediction, sampling, and generation strategies
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # Configuration
        self.hidden_size = config.get('hidden_size', 512)
        self.vocab_size = config.get('vocab_size', 50000)
        self.spike_threshold = config.get('spike_threshold', 0.5)
        self.max_length = config.get('max_length', 1024)    
        
        # Generation parameters
        self.default_temperature = config.get('temperature', 1.0)
        self.default_top_k = config.get('top_k', 50)
        self.default_top_p = config.get('top_p', 0.9)
        
        # Core components
        embedding_dim = config.get('embedding_dim', 256)
        
        # Pre-projection layer
        self.pre_projection = SpikingLinear(embedding_dim, embedding_dim)
        
        # Vocabulary projection layer
        self.vocab_projection = SpikingLinear(embedding_dim, embedding_dim)
        self.projection = nn.Linear(embedding_dim, config.get('vocab_size', 50000))
        
        # Initialize weights
        self.projection.weight.data.normal_(mean=0.0, std=0.02)
        
        self.sampler = NeuroSampler(self.vocab_size)
        
        # Generation statistics
        self.register_buffer('total_tokens_generated', torch.tensor(0))
        self.register_buffer('avg_generation_entropy', torch.tensor(0.0))
        self.register_buffer('avg_confidence', torch.tensor(0.0))
        
        # Special tokens (configurable)
        self.pad_token_id = config.get('pad_token_id', 0)
        self.eos_token_id = config.get('eos_token_id', 2)
        self.bos_token_id = config.get('bos_token_id', 1)
    
    def initialize(self) -> bool:
        """Initialize language modeling head"""
        try:
            self.state = 'active'
            self.logger.info(f"SpikingLanguageHead initialized: vocab_size={self.vocab_size}")
            return True
        except Exception as e:
            self.logger.error(f"Language head initialization failed: {e}")
            return False
    
    def forward(self, spike_input: torch.Tensor, labels: Optional[torch.Tensor] = None,
                temperature: float = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for language modeling
        
        Args:
            spike_input: (batch, seq_len, hidden_size) spike sequences
            labels: (batch, seq_len) target token ids for training
            temperature: Generation temperature
            
        Returns:
            Dict containing logits, loss, and generation info
        """
        if temperature is None:
            temperature = self.default_temperature
        
        # Apply pre-projection spiking layer
        processed_spikes = self.pre_projection(spike_input)
        
        # Apply vocabulary projection
        logits = self.projection(processed_spikes)

        # In a neuromorphic system, you might also return spike counts, timing, etc.
        return {
            'logits': logits,
            'spike_output': processed_spikes
        }
    
    def generate(self, input_ids: torch.Tensor, spike_sequences: torch.Tensor, 
                 max_length: Optional[int] = None, temperature: float = None,
                 sampling_method: str = 'top_p', consciousness_engine=None, **kwargs) -> Dict[str, Any]:
        """
        Generate text using neuromorphic sampling
        
        Args:
            input_ids: (batch, seq_len) input token ids
            spike_sequences: (batch, seq_len, hidden_size) current spike sequences
            max_length: Maximum generation length
            temperature: Sampling temperature
            sampling_method: 'greedy', 'top_k', 'top_p', 'nucleus', 'typical'
            consciousness_engine: Optional self-awareness engine for conscious generation
            
        Returns:
            Dict with generated tokens and generation info
        """
        if max_length is None:
            max_length = self.max_length
        if temperature is None:
            temperature = self.default_temperature
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize generation
        generated_ids = input_ids.clone()
        current_spikes = spike_sequences.clone()
        
        generation_log = {
            'entropy_history': [],
            'confidence_history': [],
            'consciousness_decisions': [] if consciousness_engine else None
        }
        
        # Generate tokens one by one
        for step in range(max_length - seq_len):
            # Get logits from current spike state
            output = self.forward(current_spikes, temperature=temperature)
            current_logits = output['logits'][:, -1, :]  # Last position
            
            # Consciousness-guided generation (if available)
            if consciousness_engine is not None:
                # Process through consciousness
                consciousness_state = current_spikes.mean(dim=1)  # Average spikes
                awareness_result = consciousness_engine.process_experience(
                    consciousness_state,
                    context={
                        'task': 'text_generation',
                        'step': step,
                        'entropy': output['entropy'].item(),
                        'confidence': output['confidence'].item()
                    }
                )
                
                # Adjust generation based on awareness level
                awareness_level = awareness_result['awareness_level'].value
                if awareness_level >= 3:  # Metacognitive level
                    # More creative/diverse generation
                    adjusted_temperature = temperature * 1.2
                    adjusted_method = 'top_p'
                    top_p = 0.95
                elif awareness_level >= 2:  # Reflective level
                    # Balanced generation
                    adjusted_temperature = temperature
                    adjusted_method = sampling_method
                    top_p = kwargs.get('top_p', self.default_top_p)
                else:  # Reactive level
                    # More conservative generation
                    adjusted_temperature = temperature * 0.8
                    adjusted_method = 'top_k'
                    top_k = min(kwargs.get('top_k', self.default_top_k), 20)
                
                generation_log['consciousness_decisions'].append({
                    'awareness_level': awareness_level,
                    'temperature': adjusted_temperature,
                    'method': adjusted_method,
                    'confidence': awareness_result['introspection']['confidence'].item()
                })
            else:
                adjusted_temperature = temperature
                adjusted_method = sampling_method
            
            # Sample next token
            if adjusted_method == 'top_k':
                next_token = self.sampler.spike_based_sampling(
                    current_logits, method='top_k', temperature=adjusted_temperature,
                    top_k=kwargs.get('top_k', self.default_top_k)
                )
            elif adjusted_method in ['top_p', 'nucleus']:
                next_token = self.sampler.spike_based_sampling(
                    current_logits, method='top_p', temperature=adjusted_temperature,
                    top_p=kwargs.get('top_p', self.default_top_p)
                )
            elif adjusted_method == 'typical':
                next_token = self.sampler.spike_based_sampling(
                    current_logits, method='typical', temperature=adjusted_temperature,
                    typical_p=kwargs.get('typical_p', 0.95)
                )
            else:  # greedy
                next_token = self.sampler.spike_based_sampling(
                    current_logits, method='greedy'
                )
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(-1)], dim=-1)
            
            # Update statistics
            self.total_tokens_generated += batch_size
            generation_log['entropy_history'].append(output['entropy'].item())
            generation_log['confidence_history'].append(output['confidence'].item())
            
            # Check for end of sequence
            if torch.all(next_token == self.eos_token_id):
                break
            
            # Update spike sequences for next iteration (would need tokenizer)
            # For now, we'll extend with zeros - in practice, you'd re-tokenize
            next_spike = torch.zeros(batch_size, 1, current_spikes.size(-1), device=device)
            current_spikes = torch.cat([current_spikes, next_spike], dim=1)
        
        return {
            'generated_ids': generated_ids,
            'input_length': seq_len,
            'generated_length': generated_ids.size(1) - seq_len,
            'generation_log': generation_log,
            'final_spike_sequences': current_spikes
        }
    
    def beam_search(self, input_ids: torch.Tensor, spike_sequences: torch.Tensor,
                    num_beams: int = 5, max_length: Optional[int] = None,
                    temperature: float = None) -> Dict[str, Any]:
        """
        Beam search generation for neuromorphic LLM
        
        Args:
            input_ids: (batch, seq_len) input tokens
            spike_sequences: (batch, seq_len, hidden_size) spike sequences
            num_beams: Number of beams
            max_length: Maximum generation length
            temperature: Generation temperature
            
        Returns:
            Dict with beam search results
        """
        if max_length is None:
            max_length = self.max_length
        if temperature is None:
            temperature = self.default_temperature
            
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize beams
        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        beam_ids = input_ids.unsqueeze(1).repeat(1, num_beams, 1)  # (batch, num_beams, seq_len)
        beam_spikes = spike_sequences.unsqueeze(1).repeat(1, num_beams, 1, 1)  # (batch, num_beams, seq_len, hidden_size)
        
        # Reshape for processing
        beam_ids_flat = beam_ids.view(batch_size * num_beams, seq_len)
        beam_spikes_flat = beam_spikes.view(batch_size * num_beams, seq_len, spike_sequences.size(-1))
        
        for step in range(max_length - seq_len):
            # Get logits for all beams
            output = self.forward(beam_spikes_flat, temperature=temperature)
            logits = output['logits'][:, -1, :]  # (batch*num_beams, vocab_size)
            
            # Compute log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Reshape back to beams
            log_probs = log_probs.view(batch_size, num_beams, -1)  # (batch, num_beams, vocab_size)
            
            # Add to beam scores
            next_scores = beam_scores.unsqueeze(-1) + log_probs  # (batch, num_beams, vocab_size)
            
            # Flatten and get top k
            next_scores_flat = next_scores.view(batch_size, -1)  # (batch, num_beams * vocab_size)
            top_scores, top_indices = torch.topk(next_scores_flat, num_beams, dim=-1)
            
            # Convert indices back to beam and token indices
            beam_indices = top_indices // self.vocab_size
            token_indices = top_indices % self.vocab_size
            
            # Update beams
            new_beam_ids = []
            new_beam_spikes = []
            
            for b in range(batch_size):
                batch_beam_ids = []
                batch_beam_spikes = []
                
                for beam_idx in range(num_beams):
                    # Get source beam and token
                    source_beam = beam_indices[b, beam_idx]
                    next_token = token_indices[b, beam_idx]
                    
                    # Copy from source beam and add token
                    source_ids = beam_ids[b, source_beam]
                    new_ids = torch.cat([source_ids, next_token.unsqueeze(0)])
                    batch_beam_ids.append(new_ids)
                    
                    # Update spikes (simplified - would need proper tokenizer integration)
                    source_spikes = beam_spikes[b, source_beam]
                    next_spike = torch.zeros(1, source_spikes.size(-1), device=device)
                    new_spikes = torch.cat([source_spikes, next_spike])
                    batch_beam_spikes.append(new_spikes)
                
                new_beam_ids.append(torch.stack(batch_beam_ids))
                new_beam_spikes.append(torch.stack(batch_beam_spikes))
            
            beam_ids = torch.stack(new_beam_ids)
            beam_spikes = torch.stack(new_beam_spikes)
            beam_scores = top_scores
            
            # Flatten for next iteration
            beam_ids_flat = beam_ids.view(batch_size * num_beams, beam_ids.size(-1))
            beam_spikes_flat = beam_spikes.view(batch_size * num_beams, beam_spikes.size(-2), beam_spikes.size(-1))
            
            # Check for EOS tokens
            if torch.all(token_indices == self.eos_token_id):
                break
        
        # Return best beam for each batch
        best_beam_idx = torch.argmax(beam_scores, dim=-1)
        best_sequences = torch.gather(beam_ids, 1, best_beam_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, beam_ids.size(-1))).squeeze(1)
        
        return {
            'generated_ids': best_sequences,
            'beam_scores': beam_scores,
            'all_beams': beam_ids,
            'input_length': seq_len,
            'generated_length': best_sequences.size(1) - seq_len
        }
    
    def process(self, input_data: Any) -> Any:
        """AURAModule interface"""
        if isinstance(input_data, dict):
            spike_input = input_data.get('spike_sequences', input_data.get('input'))
            labels = input_data.get('labels')
            temperature = input_data.get('temperature')
            return self.forward(spike_input, labels, temperature)
        else:
            return self.forward(input_data)
    
    def get_language_statistics(self) -> Dict[str, float]:
        """Get comprehensive language modeling statistics"""
        return {
            'total_tokens_generated': float(self.total_tokens_generated),
            'avg_generation_entropy': float(self.avg_generation_entropy),
            'avg_confidence': float(self.avg_confidence),
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'default_temperature': self.default_temperature,
            'spike_threshold': self.spike_threshold
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get language head state for serialization"""
        return {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'statistics': self.get_language_statistics()
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set language head state from serialization"""
        try:
            self.load_state_dict(state['model_state_dict'])
            return True
        except Exception as e:
            self.logger.error(f"Failed to set language head state: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate language head functionality"""
        try:
            # Test forward pass
            test_spikes = torch.randn(2, 32, self.hidden_size)
            test_labels = torch.randint(0, self.vocab_size, (2, 32))
            
            output = self.forward(test_spikes, test_labels)
            
            if 'logits' not in output:
                return False, "Missing logits in output"
            
            logits = output['logits']
            expected_shape = (2, 32, self.vocab_size)
            if logits.shape != expected_shape:
                return False, f"Wrong logits shape: {logits.shape} vs {expected_shape}"
            
            if 'loss' not in output:
                return False, "Missing loss in output"
            
            # Test generation
            input_ids = torch.randint(0, self.vocab_size, (1, 10))
            test_spikes_gen = torch.randn(1, 10, self.hidden_size)
            
            gen_result = self.generate(input_ids, test_spikes_gen, max_length=15)
            
            if 'generated_ids' not in gen_result:
                return False, "Missing generated_ids in generation result"
            
            return True, "Language head validation successful"
        except Exception as e:
            return False, f"Language head validation error: {str(e)}"

# Utility functions
def create_language_head(config: Dict[str, Any]) -> SpikingLanguageHead:
    """Factory function to create language head"""
    return SpikingLanguageHead('spiking_language_head', config)

def test_language_head():
    """Test the spiking language head"""
    config = {
        'hidden_size': 256,
        'vocab_size': 10000,
        'spike_threshold': 0.5,
        'max_length': 128,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 0.9,
        'pad_token_id': 0,
        'eos_token_id': 2,
        'bos_token_id': 1,
        'embedding_dim': 256 # Added embedding_dim for testing
    }
    
    language_head = create_language_head(config)
    assert language_head.initialize() == True
    
    # Test training forward pass
    batch_size, seq_len = 4, 32
    spike_input = torch.randn(batch_size, seq_len, config['hidden_size'])
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    output = language_head(spike_input, labels)
    
    print(f"Training Results:")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Perplexity: {output['perplexity'].item():.2f}")
    print(f"Entropy: {output['entropy'].item():.3f}")
    print(f"Confidence: {output['confidence'].item():.3f}")
    
    # Test generation
    input_ids = torch.randint(1, 100, (2, 10))  # Start tokens
    input_spikes = torch.randn(2, 10, config['hidden_size'])
    
    print(f"\nGeneration Test:")
    print(f"Input IDs: {input_ids[0].tolist()}")
    
    # Test different sampling methods
    for method in ['greedy', 'top_k', 'top_p']:
        gen_result = language_head.generate(
            input_ids, input_spikes, max_length=20, 
            sampling_method=method, temperature=1.0
        )
        print(f"Generated ({method}): {gen_result['generated_ids'][0].tolist()}")
        print(f"Generation length: {gen_result['generated_length']}")
    
    # Test beam search
    beam_result = language_head.beam_search(input_ids, input_spikes, num_beams=3, max_length=15)
    print(f"Beam search result: {beam_result['generated_ids'][0].tolist()}")
    
    # Test validation
    is_valid, message = language_head.validate()
    print(f"\nValidation: {is_valid}, {message}")
    
    # Show statistics
    stats = language_head.get_language_statistics()
    print(f"\nStatistics: {stats}")
    
    return language_head

if __name__ == "__main__":
    test_language_head()
