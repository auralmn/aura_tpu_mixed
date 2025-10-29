# SPDX-License-Identifier: Apache-2.0
"""
AURA Neural Interfaces - Complete Version with CognitiveModule
Base interfaces for all AURA neural components including cognitive modules
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum

class ModuleState(Enum):
    """Module operational states"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"  
    ACTIVE = "active"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class ModuleMetrics:
    """Standard metrics for all modules"""
    processing_time: float = 0.0
    memory_usage: float = 0.0
    energy_consumption: float = 0.0
    accuracy: float = 0.0
    throughput: float = 0.0
    error_count: int = 0

class AURAModule(nn.Module, ABC):
    """
    Base interface for all AURA system modules
    Provides common functionality for state management, metrics, and hot-swapping
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__()
        self.module_id = module_id
        self.config = config
        self.state = ModuleState.INACTIVE
        self.logger = logging.getLogger(f"AURA.{self.__class__.__name__}.{module_id}")
        self.metrics = ModuleMetrics()
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the module - must be implemented by subclasses"""
        pass
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data - must be implemented by subclasses"""
        pass
        
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get module state for hot-swapping"""
        pass
        
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set module state during hot-swapping"""
        pass
        
    @abstractmethod
    def validate(self) -> Tuple[bool, str]:
        """Validate module functionality"""
        pass
        
    def get_metrics(self) -> ModuleMetrics:
        """Get current module metrics"""
        return self.metrics
        
    def reset_metrics(self):
        """Reset module metrics"""
        self.metrics = ModuleMetrics()

class NeuronModel(AURAModule):
    """
    Base interface for neuron models in AURA
    Extends AURAModule with neuron-specific functionality
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.spike_count = 0
        self.firing_rate = 0.0
        
    @abstractmethod
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuron"""
        pass
        
    @abstractmethod
    def reset_state(self, batch_size: int = 1):
        """Reset neuron internal state"""
        pass
        
    @abstractmethod
    def get_neuron_statistics(self) -> Dict[str, float]:
        """Get neuron-specific statistics"""
        pass
        
    def update_firing_rate(self, spikes: torch.Tensor):
        """Update firing rate statistics"""
        self.spike_count += torch.sum(spikes).item()
        # Simple running average - could be more sophisticated
        self.firing_rate = 0.9 * self.firing_rate + 0.1 * torch.mean(spikes).item()

class CognitiveModule(AURAModule):
    """
    Base interface for cognitive modules in AURA
    Handles higher-level cognitive functions like self-awareness, reasoning, memory
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.cognitive_state = {}
        self.memory_systems = {}
        self.attention_mechanisms = {}
        
    @abstractmethod
    def process_experience(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process an experience or input through the cognitive module"""
        pass
        
    @abstractmethod
    def update_memory(self, experience: Dict[str, Any], memory_type: str = "episodic"):
        """Update memory systems with new experience"""
        pass
        
    @abstractmethod
    def retrieve_memory(self, query: Dict[str, Any], memory_type: str = "episodic") -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query"""
        pass
        
    @abstractmethod
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state"""
        pass
        
    @abstractmethod
    def set_cognitive_state(self, state: Dict[str, Any]) -> bool:
        """Set cognitive state"""
        pass
        
    def get_attention_focus(self) -> Optional[Dict[str, Any]]:
        """Get current attention focus"""
        return self.attention_mechanisms.get('current_focus')
        
    def set_attention_focus(self, focus: Dict[str, Any]):
        """Set attention focus"""
        self.attention_mechanisms['current_focus'] = focus
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of all memory systems"""
        summary = {}
        for memory_type, memory_system in self.memory_systems.items():
            if hasattr(memory_system, '__len__'):
                summary[f'{memory_type}_count'] = len(memory_system)
            else:
                summary[f'{memory_type}_status'] = 'active' if memory_system else 'inactive'
        return summary

class NetworkInterface(AURAModule):
    """
    Interface for network-level components (layers, connections, routing)
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.connections = {}
        self.routing_table = {}
        
    @abstractmethod
    def connect_modules(self, source_id: str, target_id: str, connection_config: Dict[str, Any]):
        """Connect two modules"""
        pass
        
    @abstractmethod
    def route_signal(self, signal: torch.Tensor, source_id: str, target_id: str) -> torch.Tensor:
        """Route signal between modules"""
        pass
        
    @abstractmethod
    def update_routing(self, routing_update: Dict[str, Any]):
        """Update routing configuration"""
        pass

class DataInterface(AURAModule):
    """
    Interface for data processing modules (datasets, embeddings, preprocessing)
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.data_statistics = {}
        self.preprocessing_pipeline = []
        
    @abstractmethod
    def load_data(self, data_path: str) -> bool:
        """Load data from path"""
        pass
        
    @abstractmethod
    def preprocess_data(self, raw_data: Any) -> Any:
        """Preprocess raw data"""
        pass
        
    @abstractmethod
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get batch of processed data"""
        pass
        
    def add_preprocessing_step(self, step_function, step_config: Dict[str, Any]):
        """Add preprocessing step to pipeline"""
        self.preprocessing_pipeline.append({
            'function': step_function,
            'config': step_config
        })
        
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data statistics"""
        return self.data_statistics

# Legacy interfaces for backward compatibility
class SpikingNeuron(NeuronModel):
    """Interface for spiking neuron models"""
    
    @abstractmethod
    def get_available_patterns(self) -> List[str]:
        """Get list of available spiking patterns"""
        pass
    
    @abstractmethod
    def get_pattern_parameters(self, pattern_name: str) -> Dict[str, float]:
        """Get parameters for specific pattern"""
        pass
    
    @abstractmethod
    def set_pattern(self, pattern_name: str) -> bool:
        """Set neuron to specific biological pattern"""
        pass

class FeedForwardNetwork(AURAModule):
    """Interface for feed-forward network components"""
    
    @abstractmethod
    def forward(self, x):
        """Forward pass through the network"""
        pass

class EmbeddingLayer(AURAModule):
    """Interface for embedding layers"""
    
    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through embedding layer"""
        pass

# Factory functions for creating modules
def create_neuron_model(neuron_type: str, module_id: str, config: Dict[str, Any]) -> NeuronModel:
    """Factory function to create neuron models"""
    if neuron_type == "AdEx":
        from aura.neural.adex_fixed import AdExNeuron
        return AdExNeuron(module_id, config)
    elif neuron_type == "SRM":
        from aura.neural.srm_fixed import SRMNeuron
        return SRMNeuron(module_id, config)
    else:
        raise ValueError(f"Unknown neuron type: {neuron_type}")

def create_cognitive_module(module_type: str, module_id: str, config: Dict[str, Any]) -> CognitiveModule:
    """Factory function to create cognitive modules"""
    if module_type == "SelfAwareness":
        from aura.neural.self_awareness import SelfAwarenessEngine
        return SelfAwarenessEngine(module_id, config)
    else:
        raise ValueError(f"Unknown cognitive module type: {module_type}")

# Utility functions for module management
def validate_module_config(config: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, str]:
    """Validate module configuration"""
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    return True, "Configuration valid"

def get_module_info(module: AURAModule) -> Dict[str, Any]:
    """Get comprehensive module information"""
    return {
        'module_id': module.module_id,
        'module_type': module.__class__.__name__,
        'state': module.state.value,
        'metrics': module.get_metrics(),
        'config': module.config
    }

# Module registry for dynamic loading
MODULE_REGISTRY = {
    'neurons': {
        'AdEx': 'aura.neural.adex_fixed.AdExNeuron',
        'SRM': 'aura.neural.srm_fixed.SRMNeuron'
    },
    'cognitive': {
        'SelfAwareness': 'aura.neural.self_awareness.SelfAwarenessEngine'
    },
    'network': {
        'SRWKV': 'aura.neural.attention.srwkv.SpikingSRWKV',
        'SRFFN': 'aura.neural.srffn.SpikingSRFFN'
    },
    'data': {
        'BinaryEmbedding': 'aura.neural.embedding.BinarySpikeEmbedding'
    }
}

def load_module_from_registry(module_category: str, module_type: str, 
                             module_id: str, config: Dict[str, Any]) -> AURAModule:
    """Load module dynamically from registry"""
    if module_category not in MODULE_REGISTRY:
        raise ValueError(f"Unknown module category: {module_category}")
    
    if module_type not in MODULE_REGISTRY[module_category]:
        raise ValueError(f"Unknown module type: {module_type} in category {module_category}")
    
    module_path = MODULE_REGISTRY[module_category][module_type]
    module_parts = module_path.split('.')
    class_name = module_parts[-1]
    module_name = '.'.join(module_parts[:-1])
    
    # Dynamic import
    import importlib
    module = importlib.import_module(module_name)
    module_class = getattr(module, class_name)
    
    return module_class(module_id, config)