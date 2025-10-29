#!/usr/bin/env python3
"""
AURA Neuromorphic Text Generation System
Implements spiking language generation with knowledge retrieval from AURA memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Try to import Norse for advanced neuromorphic features
try:
    import norse.torch as snn
    NORSE_AVAILABLE = True
except ImportError:
    NORSE_AVAILABLE = False
    print("Norse not available, using custom neuromorphic implementations")

# Import AURA systems
from aura.memory.hierarchical_memory import HierarchicalMemory, MemoryItem, MemoryType
from aura.neural.interfaces import AURAModule
from aura.neural.embedding import BinarySpikeEmbedding

logger = logging.getLogger(__name__)


class SpikingLanguageCore(nn.Module):
    """
    Neuromorphic language core using spiking RNN cells
    Processes context states with biologically-plausible dynamics
    """
    
    def __init__(self, hidden_dim: int, dt: float = 1e-3, use_norse: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        self.use_norse = use_norse and NORSE_AVAILABLE
        
        if self.use_norse:
            # Use Norse LIF cell for true neuromorphic processing
            self.rnn_cell = snn.LIFCell(snn.LIFParameters(
                tau_mem_inv=torch.tensor(1.0),
                tau_syn_inv=torch.tensor(1.0),
                v_leak=torch.tensor(0.0),
                v_th=torch.tensor(1.0),
                v_reset=torch.tensor(0.0),
            ))
        else:
            # Custom spiking implementation
            self.adaptive_threshold = nn.Parameter(torch.full((hidden_dim,), 0.5))
            self.membrane_potential = None
            self.refractory_period = 2
            self.refractory_counter = None
        
        # State tracking
        self.register_buffer('rnn_state', None)
        self.register_buffer('firing_rates', None)
        
    def reset_state(self, batch_size: int, device: torch.device):
        """Reset RNN state for new sequence"""
        if self.use_norse:
            # rnn cell state
            try:
                st = self.rnn_cell.initial_state(batch_size)
            except TypeError:
                try:
                    st = self.rnn_cell.initial_state(batch_size, self.hidden_dim)
                except TypeError:
                    st = self.rnn_cell.initial_state(batch_size)
            try:
                self.rnn_state = type(st)(*(t.to(device) for t in st))
            except Exception:
                self.rnn_state = st
        else:
            self.membrane_potential = torch.zeros(batch_size, self.hidden_dim, device=device)
            self.refractory_counter = torch.zeros(batch_size, self.hidden_dim, device=device)
            self.firing_rates = torch.zeros(batch_size, self.hidden_dim, device=device)
    
    def forward(self, input_state: torch.Tensor, prev_rnn_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input state through spiking RNN
        
        Args:
            input_state: [batch, hidden_dim] - driving current/context
            prev_rnn_state: Previous RNN state (optional)
            
        Returns:
            rate_out: [batch, hidden_dim] - firing rate representation
            next_rnn_state: Updated RNN state
        """
        if self.use_norse:
            return self._norse_forward(input_state, prev_rnn_state)
        else:
            return self._custom_forward(input_state, prev_rnn_state)
    
    def process(self, context_spikes: Optional[torch.Tensor], 
                prompt_spikes: torch.Tensor) -> torch.Tensor:
        """Alias for forward to comply with AURAModule interface."""
        return self.forward(context_spikes, prompt_spikes)

    def _norse_forward(self, input_state: torch.Tensor, prev_rnn_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using Norse LIF neurons"""
        # Coerce input to tensor, float32, on a stable device
        if not isinstance(input_state, torch.Tensor):
            input_state = torch.tensor(input_state)
        input_state = input_state.to(dtype=torch.float32, device=self.rnn_state.v.device if (self.use_norse and self.rnn_state is not None and hasattr(self.rnn_state, 'v')) else input_state.device)

        if prev_rnn_state is not None:
            # Move prev state to input device if needed
            try:
                self.rnn_state = type(prev_rnn_state)(*(t.to(input_state.device) for t in prev_rnn_state))
            except Exception:
                self.rnn_state = prev_rnn_state
                
        # Initialize state if missing
        if self.rnn_state is None:
            try:
                st = self.rnn_cell.initial_state(input_state.shape[0])
            except TypeError:
                st = self.rnn_cell.initial_state(input_state.shape[0])
            try:
                self.rnn_state = type(st)(*(t.to(input_state.device) for t in st))
            except Exception:
                self.rnn_state = st
        
        # Process through LIF cell
        spk_out, self.rnn_state = self.rnn_cell(input_state.float(), self.rnn_state)
        
        # Convert spikes to continuous firing rate representation
        rate = spk_out.float()
        
        return rate, self.rnn_state
    
    def _custom_forward(self, input_state: torch.Tensor, prev_rnn_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Custom forward pass with surrogate gradients"""
        batch_size = input_state.size(0)
        device = input_state.device
        
        # Initialize state if needed
        if self.membrane_potential is None:
            self.reset_state(batch_size, device)
        
        # Update membrane potential
        self.membrane_potential = 0.9 * self.membrane_potential + input_state
        
        # Apply refractory period
        if self.refractory_counter is not None:
            self.membrane_potential = self.membrane_potential * (self.refractory_counter == 0).float()
            self.refractory_counter = torch.clamp(self.refractory_counter - 1, min=0)
        
        # Generate spikes
        spike_probs = torch.sigmoid((self.membrane_potential - self.adaptive_threshold) * 5.0)
        
        if self.training:
            spikes = torch.bernoulli(spike_probs)
            # Surrogate gradient
            surrogate_grad = torch.clamp(1 - torch.abs(self.membrane_potential - self.adaptive_threshold), 0, 1) * 0.1
            spikes = spikes + surrogate_grad - surrogate_grad.detach()
            spikes = torch.clamp(spikes, 0, 1)
        else:
            spikes = (spike_probs > 0.5).float()
        
        # Update refractory counter for spiking neurons
        self.refractory_counter = torch.where(spikes > 0.5, 
                                            torch.full_like(self.refractory_counter, self.refractory_period),
                                            self.refractory_counter)
        
        # Reset membrane potential for spiking neurons
        self.membrane_potential = torch.where(spikes > 0.5, 
                                            torch.zeros_like(self.membrane_potential),
                                            self.membrane_potential)
        
        # Update firing rates (exponential moving average)
        self.firing_rates = 0.9 * self.firing_rates + 0.1 * spikes
        
        return self.firing_rates, self.membrane_potential


class TokenDecoder(nn.Module):
    """
    Linear readout layer for mapping firing rates to vocabulary logits
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Linear mapping from firing rates to vocabulary
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights
        nn.init.normal_(self.fc.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, rate_vector: torch.Tensor) -> torch.Tensor:
        """
        Map firing rate vector to token probabilities
        
        Args:
            rate_vector: [batch, hidden_dim] - firing rate representation
            
        Returns:
            probs: [batch, vocab_size] - token probabilities
        """
        logits = self.fc(rate_vector)  # [batch, vocab_size]
        probs = F.softmax(logits, dim=-1)  # [batch, vocab_size]
        return probs


class KnowledgeRetrievalNetwork(nn.Module):
    """
    Spiking MoE expert ensemble for retrieving context from AURA memory
    """
    
    def __init__(self, query_dim: int, hidden_dim: int, num_experts: int = 8, 
                 use_spiking: bool = True, use_norse: bool = True):
        super().__init__()
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.use_spiking = use_spiking
        self.use_norse = use_norse
        
        # Expert networks for different memory types
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(num_experts)
        ])
        
        # Gating network for expert selection
        self.gating_net = nn.Linear(query_dim, num_experts)
        
        # Initialize weights
        nn.init.normal_(self.gating_net.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.gating_net.bias)
    
    def _create_expert(self) -> nn.Module:
        """Create a single expert network"""
        if self.use_spiking:
            return SpikingExpert(self.query_dim, self.hidden_dim, use_norse=self.use_norse)
        else:
            return nn.Sequential(
                nn.Linear(self.query_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )
    
    def forward(self, memory_query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve context from memory using spiking experts
        
        Args:
            memory_query: [batch, query_dim] - query embedding
            
        Returns:
            h_t: [batch, hidden_dim] - context vector
        """
        # Normalize query into [0,1] range for Poisson encoding
        query_spikes = torch.bernoulli(torch.clamp(memory_query, 0, 1))
        
        # Compute gating scores
        gate_logits = self.gating_net(query_spikes)  # [batch, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # [batch, num_experts]
        
        # Gather weighted expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(query_spikes)  # [batch, hidden_dim]
            expert_outputs.append(expert_out)
        
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [batch, hidden_dim, num_experts]
        
        # Weighted sum by routing probabilities
        h_t = (stacked_outputs * gate_probs.unsqueeze(1)).sum(dim=-1)  # [batch, hidden_dim]
        
        return h_t


class SpikingExpert(nn.Module):
    """
    Spiking expert network for knowledge retrieval
    """
    
    def __init__(self, input_dim: int, output_dim: int, use_norse: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_norse = use_norse and NORSE_AVAILABLE
        
        # Dense weight layers
        self.w1 = nn.Linear(input_dim, output_dim, bias=False)
        self.w2 = nn.Linear(output_dim, output_dim, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        
        if self.use_norse:
            # LIF neurons
            self.neuron1 = snn.LIFCell(snn.LIFParameters(
                tau_mem_inv=torch.tensor(1.0),
                tau_syn_inv=torch.tensor(1.0),
                v_leak=torch.tensor(0.0),
                v_th=torch.tensor(1.0),
                v_reset=torch.tensor(0.0),
            ))
            self.neuron2 = snn.LIFCell(snn.LIFParameters(
                tau_mem_inv=torch.tensor(1.0),
                tau_syn_inv=torch.tensor(1.0),
                v_leak=torch.tensor(0.0),
                v_th=torch.tensor(1.0),
                v_reset=torch.tensor(0.0),
            ))
            self.register_buffer('state1', None)
            self.register_buffer('state2', None)
        else:
            # Custom spiking implementation
            self.adaptive_threshold1 = nn.Parameter(torch.full((output_dim,), 0.5))
            self.adaptive_threshold2 = nn.Parameter(torch.full((output_dim,), 0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spiking expert"""
        if self.use_norse:
            return self._norse_forward(x)
        else:
            return self._custom_forward(x)
    
    def _norse_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using Norse LIF neurons"""
        # Coerce input to float tensor on device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(dtype=torch.float32)
        batch_size = x.size(0)
        device = x.device
        
        # Initialize states if needed
        if self.state1 is None:
            # output neuron states
            try:
                st1 = self.neuron1.initial_state(batch_size)
            except TypeError:
                try:
                    st1 = self.neuron1.initial_state(batch_size, self.output_dim)
                except TypeError:
                    st1 = self.neuron1.initial_state(batch_size)
            try:
                self.state1 = type(st1)(*(t.to(device) for t in st1))
            except Exception:
                self.state1 = st1
            try:
                st2 = self.neuron2.initial_state(batch_size)
            except TypeError:
                try:
                    st2 = self.neuron2.initial_state(batch_size, self.output_dim)
                except TypeError:
                    st2 = self.neuron2.initial_state(batch_size)
            try:
                self.state2 = type(st2)(*(t.to(device) for t in st2))
            except Exception:
                self.state2 = st2
        
        # Simulate over T time steps
        T = 10
        out_spikes = torch.zeros(batch_size, self.output_dim, device=device)
        
        for t in range(T):
            # Encode input as Poisson spikes
            prob = torch.clamp(x, min=0.0, max=1.0)
            input_spikes = torch.bernoulli(prob)
            
            # Layer 1
            current1 = self.w1(input_spikes)
            spk1, self.state1 = self.neuron1(current1.float(), self.state1)
            
            # Layer 2
            current2 = self.w2(spk1)
            spk2, self.state2 = self.neuron2(current2.float(), self.state2)
            
            # Accumulate output spikes
            out_spikes += spk2
        
        # Normalize by time to get firing rates
        return out_spikes / T
    
    def _custom_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Custom forward pass with surrogate gradients"""
        # Layer 1
        linear1 = self.w1(x)
        threshold1 = self.adaptive_threshold1.unsqueeze(0)
        spike_probs1 = torch.sigmoid((linear1 - threshold1) * 5.0)
        
        if self.training:
            spikes1 = torch.bernoulli(spike_probs1)
            surrogate_grad1 = torch.clamp(1 - torch.abs(linear1 - threshold1), 0, 1) * 0.1
            spikes1 = spikes1 + surrogate_grad1 - surrogate_grad1.detach()
            spikes1 = torch.clamp(spikes1, 0, 1)
        else:
            spikes1 = (spike_probs1 > 0.5).float()
        
        # Layer 2
        linear2 = self.w2(spikes1)
        threshold2 = self.adaptive_threshold2.unsqueeze(0)
        spike_probs2 = torch.sigmoid((linear2 - threshold2) * 5.0)
        
        if self.training:
            spikes2 = torch.bernoulli(spike_probs2)
            surrogate_grad2 = torch.clamp(1 - torch.abs(linear2 - threshold2), 0, 1) * 0.1
            spikes2 = spikes2 + surrogate_grad2 - surrogate_grad2.detach()
            spikes2 = torch.clamp(spikes2, 0, 1)
        else:
            spikes2 = (spike_probs2 > 0.5).float()
        
        return spikes2


class NeuromorphicTextGenerator(AURAModule):
    """
    Complete neuromorphic text generation system
    Integrates knowledge retrieval, spiking language core, and token decoding
    Uses AURA memory system and BinarySpikeEmbedding
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        
        # Configuration
        self.hidden_dim = config.get('hidden_dim', 768)
        self.vocab_size = config.get('vocab_size', 32000)
        self.query_dim = config.get('query_dim', 768)
        self.num_experts = config.get('num_experts', 8)
        self.use_norse = config.get('use_norse', True)
        self.dt = config.get('dt', 1e-3)
        self.eos_token_id = config.get('eos_token_id', 2)  # default EOS from BasicTokenizer
        self.enable_norse_debug = config.get('enable_norse_debug', False)
        
        # AURA Memory system integration
        self.memory_system = config.get('memory_system')  # Will be injected from boot sequence
        
        # Binary spike embedding for token-to-spike conversion
        embedding_config = {
            'num_embeddings': self.vocab_size,
            'embedding_dim': self.query_dim,
            'timesteps': config.get('timesteps', 10),
            'encoding_scheme': config.get('encoding_scheme', 'rate'),
            'spike_threshold': config.get('spike_threshold', 0.5),
            'enable_temporal_dynamics': True,
            'decay_rate': 0.9
        }
        self.spike_embedding = BinarySpikeEmbedding(f'{module_id}_embedding', embedding_config)
        
        # Components
        self.knowledge_retrieval = KnowledgeRetrievalNetwork(
            query_dim=self.query_dim,
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            use_spiking=True,
            use_norse=self.use_norse
        )
        
        self.language_core = SpikingLanguageCore(
            hidden_dim=self.hidden_dim,
            dt=self.dt,
            use_norse=self.use_norse
        )
        
        self.token_decoder = TokenDecoder(
            hidden_dim=self.hidden_dim,
            vocab_size=self.vocab_size
        )

        # Optional Norse debug hooks
        if self.use_norse and self.enable_norse_debug:
            try:
                self._setup_norse_debug()
            except Exception as e:
                logger.warning(f"Failed to enable Norse debug hooks: {e}")
        
        # Generation state
        self.generation_state = None
        self.generated_tokens = []
        self.current_context = None
        
    def initialize(self) -> bool:
        """Initialize the text generator"""
        try:
            # Ensure spike embedding module is moved to correct device
            try:
                self.spike_embedding.initialize()
            except Exception:
                pass
            # Default device follows spike embedding
            try:
                self.device = self.spike_embedding.device
                self.knowledge_retrieval.to(self.device)
                self.language_core.to(self.device)
                self.token_decoder.to(self.device)
            except Exception:
                pass
            self.state = 'active'
            self.logger.info(f"NeuromorphicTextGenerator initialized with {self.num_experts} experts")
            return True
        except Exception as e:
            self.logger.error(f"Text generator initialization failed: {e}")
            return False

    def _setup_norse_debug(self):
        """Attach lightweight debug wrappers around Norse LIF cells used here."""
        if not NORSE_AVAILABLE:
            return
        import logging as _logging
        _logging.getLogger('norse').setLevel(_logging.DEBUG)
        _logging.getLogger('norse.torch').setLevel(_logging.DEBUG)
        ch = _logging.StreamHandler()
        ch.setLevel(_logging.DEBUG)
        ch.setFormatter(_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for name in ['norse', 'norse.torch']:
            lg = _logging.getLogger(name)
            if not any(isinstance(h, _logging.StreamHandler) for h in lg.handlers):
                lg.addHandler(ch)

        def wrap_forward(cell, name):
            orig = cell.forward
            def fwd(inp, state):
                try:
                    logger.debug(f"{name}: inp={getattr(inp,'shape',type(inp))} state={type(state)}")
                except Exception:
                    pass
                out = orig(inp, state)
                try:
                    if isinstance(out, (tuple, list)) and len(out) >= 1:
                        logger.debug(f"{name}: out={getattr(out[0],'shape',type(out[0]))}")
                except Exception:
                    pass
                return out
            cell.forward = fwd

        # Wrap language core cell
        if getattr(self.language_core, 'rnn_cell', None) is not None:
            wrap_forward(self.language_core.rnn_cell, 'LIFCore')
        # Wrap expert cells
        for idx, ex in enumerate(self.knowledge_retrieval.experts):
            if hasattr(ex, 'neuron1'):
                wrap_forward(ex.neuron1, f'LIFExpert{idx}_1')
            if hasattr(ex, 'neuron2'):
                wrap_forward(ex.neuron2, f'LIFExpert{idx}_2')
    
    def retrieve_context(self, memory_query: torch.Tensor) -> torch.Tensor:
        """
        Retrieve context from AURA memory system using spiking experts
        
        Args:
            memory_query: [batch, query_dim] - query embedding
            
        Returns:
            h_t: [batch, hidden_dim] - context vector
        """
        # If memory system is available, use it for enhanced retrieval
        if self.memory_system is not None:
            # Convert query to numpy for memory system
            query_np = memory_query.detach().cpu().numpy()
            
            # Retrieve from memory (this would be async in practice)
            # For now, we'll use the spiking experts directly
            context_vector = self.knowledge_retrieval(memory_query)
            
            # Store current context for memory integration
            self.current_context = context_vector
            
            return context_vector
        else:
            # Fallback to direct spiking expert retrieval
            return self.knowledge_retrieval(memory_query)
    
    def process(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """AURAModule interface for processing token sequences into spike outputs."""
        # Retrieve context (if memory system is available)
        if self.memory_system:
            # For simplicity, use the mean of prompt embeddings as the memory query
            prompt_embeddings = self.spike_embedding.get_spike_rate(self.spike_embedding(prompt_tokens))
            memory_query = torch.mean(prompt_embeddings, dim=1)
            context_spikes = self.retrieve_context(memory_query)
        else:
            context_spikes = None

        # Convert prompt to spikes
        prompt_spikes = self.spike_embedding(prompt_tokens)

        # Process through the language core
        return self.language_core.process(context_spikes, prompt_spikes)

    def generate_text(self, prompt_tokens: torch.Tensor, max_length: int = 100, 
                      temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> torch.Tensor:
        """
        Generate text from a prompt, decoding token by token.
        """
        generated_tokens = prompt_tokens.clone()
        
        # Ensure prompt is on the correct device
        prompt_tokens = prompt_tokens.to(self.device)

        # Main generation loop
        with torch.no_grad():
            for _ in range(max_length):
                # Process the current sequence of tokens
                output = self.process(generated_tokens)
                output_spikes = output[0] if isinstance(output, tuple) else output
                
                # Get logits from the last timestep
                last_spikes = output_spikes[:, -1, :]
                logits = self.token_decoder(last_spikes)
                
                # Apply temperature
                logits = logits / temperature
                
                # Top-k and top-p filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    # Create a mask for the logits
                    mask = torch.full_like(logits, -float('Inf'))
                    mask.scatter_(1, top_k_indices, top_k_logits)
                    logits = mask
                
                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[indices_to_remove] = -float('Inf')

                # Sample the next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append the new token to the generated sequence
                generated_tokens = torch.cat([generated_tokens, next_token], dim=1)
                
                # Check for EOS token
                if self.spike_embedding and hasattr(self.spike_embedding, 'tokenizer') and hasattr(self.spike_embedding.tokenizer, 'eos_id') and next_token.item() == self.spike_embedding.tokenizer.eos_id():
                    break
        
        return generated_tokens
    
    def get_state(self) -> Dict[str, Any]:
        """Get generator state for serialization"""
        return {
            'config': self.config,
            'model_state_dict': self.state_dict(),
            'generation_state': self.generation_state,
            'generated_tokens': self.generated_tokens
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set generator state from serialization"""
        try:
            self.load_state_dict(state['model_state_dict'])
            self.generation_state = state.get('generation_state')
            self.generated_tokens = state.get('generated_tokens', [])
            return True
        except Exception as e:
            self.logger.error(f"Failed to set text generator state: {e}")
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate text generator functionality"""
        try:
            # Test with random prompt
            batch_size = 2
            prompt_embeddings = torch.randn(batch_size, self.query_dim)
            
            generated = self.generate_text(prompt_embeddings, max_len=10)
            
            if generated.shape != (batch_size, 10):
                return False, f"Wrong output shape: {generated.shape}"
            
            return True, "Text generator validation successful"
        except Exception as e:
            return False, f"Text generator validation error: {str(e)}"


# Utility functions
def create_text_generator(config: Dict[str, Any]) -> NeuromorphicTextGenerator:
    """Factory function to create text generator"""
    return NeuromorphicTextGenerator('neuromorphic_text_generator', config)


def test_text_generator():
    """Test the complete text generation system"""
    config = {
        'hidden_dim': 256,
        'vocab_size': 1000,
        'query_dim': 256,
        'num_experts': 4,
        'use_norse': True,
        'dt': 1e-3
    }
    
    generator = create_text_generator(config)
    assert generator.initialize() == True
    
    # Test generation
    batch_size = 2
    prompt_embeddings = torch.randn(batch_size, config['query_dim'])
    
    generated = generator.generate_text(prompt_embeddings, max_len=20)
    
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0][:10].tolist()}")
    
    # Test validation
    is_valid, message = generator.validate()
    print(f"Validation: {is_valid}, {message}")
    
    return generator


if __name__ == "__main__":
    test_text_generator()
