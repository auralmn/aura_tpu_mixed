"""
Izhikevich Neuron Model Implementation
Supports all 23 biological spiking patterns
"""

import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any
from aura.core.base import AURAModule, NeuronModel


class IzhikevichNeuron(NeuronModel):
    """Izhikevich neuron model with 23 biological patterns"""
    
    def __init__(self, module_id: str, config: Dict[str, Any]):
        super().__init__(module_id, config)
        self.patterns = self._load_patterns()
        
        # Handle both direct parameters and pattern-based configuration
        if 'parameters' in config:
            self.parameters = config['parameters']
            self.current_pattern = None
        elif 'pattern' in config:
            self.current_pattern = config['pattern']
            self.parameters = self.patterns[self.current_pattern]
        else:
            # Default to regular spiking
            self.current_pattern = 'regular_spiking'
            self.parameters = self.patterns[self.current_pattern]
        
        # Neuron state variables
        self.v = -65.0  # Membrane potential
        self.u = 0.0    # Recovery variable
        
        # Performance tracking
        self.spike_count = 0
        self.simulation_time_total = 0.0
        
        # Initialize performance metrics dict
        self.performance_metrics = {
            'total_spikes': 0,
            'total_simulation_time': 0.0,
            'average_firing_rate': 0.0
        }
    
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Forward pass through neuron (for NeuronModel interface)"""
        # Convert input tensor to current value
        current = torch.mean(input_spikes).item()
        
        # Simulate for one timestep (dt=0.1ms by default)
        dt = 0.1
        spikes, _ = self.simulate(current, dt, dt)
        
        # Return spike tensor
        spike_output = torch.tensor(spikes, dtype=torch.float32)
        return spike_output.unsqueeze(0) if spike_output.dim() == 1 else spike_output
    
    def reset_state(self, batch_size: int = 1):
        """Reset neuron internal state (for NeuronModel interface)"""
        self.v = self.parameters['c']
        self.u = self.parameters['b'] * self.v
        self.spike_count = 0
    
    def get_neuron_statistics(self) -> Dict[str, float]:
        """Get neuron-specific statistics (for NeuronModel interface)"""
        return {
            'membrane_potential': float(self.v),
            'recovery_variable': float(self.u),
            'spike_count': float(self.spike_count),
            'firing_rate': float(self.firing_rate) if hasattr(self, 'firing_rate') else 0.0,
            'simulation_time': float(self.simulation_time_total)
        }
    
    def _load_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load 23 Izhikevich patterns from configuration"""
        patterns_file = Path("config/neural_patterns.yaml")
        try:
            with open(patterns_file, 'r') as f:
                data = yaml.safe_load(f)
            return data['patterns']
        except FileNotFoundError:
            # Fallback patterns if config file not found
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, float]]:
        """Default patterns if config file is not available"""
        return {
            'regular_spiking': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
            'intrinsically_bursting': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
            'chattering': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
            'fast_spiking': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
            'tonic_spiking': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
            'phasic_spiking': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 6},
            'tonic_bursting': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
            'phasic_bursting': {'a': 0.02, 'b': 0.25, 'c': -55, 'd': 0.05},
            'mixed_mode': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
            'spike_frequency_adaptation': {'a': 0.01, 'b': 0.2, 'c': -65, 'd': 8},
            'class_1_excitable': {'a': 0.02, 'b': -0.1, 'c': -55, 'd': 6},
            'class_2_excitable': {'a': 0.2, 'b': 0.26, 'c': -65, 'd': 0},
            'spike_latency': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6},
            'subthreshold_oscillations': {'a': 0.05, 'b': 0.26, 'c': -60, 'd': 0},
            'resonator': {'a': 0.1, 'b': 0.26, 'c': -60, 'd': -1},
            'integrator': {'a': 0.02, 'b': -0.1, 'c': -55, 'd': 6},
            'rebound_spike': {'a': 0.03, 'b': 0.25, 'c': -60, 'd': 4},
            'rebound_burst': {'a': 0.03, 'b': 0.25, 'c': -52, 'd': 0},
            'threshold_variability': {'a': 0.03, 'b': 0.25, 'c': -60, 'd': 4},
            'bistability': {'a': 0.1, 'b': 0.26, 'c': -60, 'd': 0},
            'dap': {'a': 1, 'b': 0.2, 'c': -60, 'd': -21},
            'accommodation': {'a': 0.02, 'b': 1, 'c': -55, 'd': 4},
            'inhibition_induced_spiking': {'a': -0.02, 'b': -1, 'c': -60, 'd': 8},
            'inhibition_induced_bursting': {'a': -0.026, 'b': -1, 'c': -45, 'd': -2},
            'low_threshold_spiking': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2}
        }
    
    def initialize(self) -> bool:
        """Initialize the neuron module"""
        try:
            # Validate parameters
            required_params = ['a', 'b', 'c', 'd']
            for param in required_params:
                if param not in self.parameters:
                    return False
            
            # Reset state
            self.v = self.parameters['c']
            self.u = self.parameters['b'] * self.v
            self.spike_count = 0
            self.simulation_time_total = 0.0
            
            return True
        except Exception:
            return False
    
    def process(self, input_data: Any) -> Any:
        """Process input data (for AURAModule interface)"""
        if isinstance(input_data, dict):
            current = input_data.get('current', 0.0)
            time = input_data.get('time', 100.0)
            dt = input_data.get('dt', 0.1)
            return self.simulate(current, time, dt)
        return None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current module state for hot-swapping"""
        return {
            'parameters': self.parameters.copy(),
            'current_pattern': self.current_pattern,
            'v': self.v,
            'u': self.u,
            'spike_count': self.spike_count,
            'simulation_time_total': self.simulation_time_total,
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set module state during hot-swapping"""
        try:
            self.parameters = state['parameters']
            self.current_pattern = state.get('current_pattern')
            self.v = state['v']
            self.u = state['u']
            self.spike_count = state['spike_count']
            self.simulation_time_total = state['simulation_time_total']
            self.performance_metrics = state['performance_metrics']
            return True
        except Exception:
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate module functionality"""
        try:
            # Check that we have valid parameters
            required_params = ['a', 'b', 'c', 'd']
            for param in required_params:
                if param not in self.parameters:
                    return False, f"Missing parameter: {param}"
            
            # Test basic simulation
            test_spikes, test_voltage = self.simulate(10.0, 100.0, 0.1)
            if len(test_spikes) == 0 or len(test_voltage) == 0:
                return False, "Simulation failed to produce output"
            
            return True, "Validation successful"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def simulate(self, input_current: float, simulation_time: float, dt: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate neuron with current parameters"""
        n_steps = int(simulation_time / dt)
        voltage = np.zeros(n_steps)
        spikes = np.zeros(n_steps)
        
        # Get current parameters
        a = self.parameters['a']
        b = self.parameters['b'] 
        c = self.parameters['c']
        d = self.parameters['d']
        
        # Reset initial conditions
        v = float(c)
        u = b * v
        
        for i in range(n_steps):
            # Izhikevich equations
            v += dt * (0.04 * v**2 + 5 * v + 140 - u + input_current)
            u += dt * a * (b * v - u)
            
            # Check for spike
            if v >= 30:
                spikes[i] = 1
                voltage[i] = 30.0  # Record the spike in voltage trace
                v = c
                u += d
                self.spike_count += 1
            else:
                voltage[i] = v
        
        # Update state
        self.v = v
        self.u = u
        self.simulation_time_total += simulation_time
        
        # Update performance metrics
        self.performance_metrics['total_spikes'] = self.spike_count
        self.performance_metrics['total_simulation_time'] = self.simulation_time_total
        self.performance_metrics['average_firing_rate'] = (
            self.spike_count / (self.simulation_time_total / 1000.0) if self.simulation_time_total > 0 else 0
        )
        
        return spikes, voltage
    
    def get_parameters(self) -> Dict[str, float]:
        """Get neuron model parameters"""
        return self.parameters.copy()
    
    def set_parameters(self, params: Dict[str, float]) -> bool:
        """Set neuron model parameters"""
        try:
            # Validate parameters
            required_params = ['a', 'b', 'c', 'd']
            for param in required_params:
                if param not in params:
                    return False
            
            self.parameters = params.copy()
            self.current_pattern = None  # Clear pattern since we're using custom parameters
            
            # Reset state with new parameters
            self.v = self.parameters['c']
            self.u = self.parameters['b'] * self.v
            
            return True
        except Exception:
            return False
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available spiking patterns"""
        return list(self.patterns.keys())
    
    def get_pattern_parameters(self, pattern_name: str) -> Dict[str, float]:
        """Get parameters for specific pattern"""
        if pattern_name in self.patterns:
            return self.patterns[pattern_name].copy()
        else:
            raise ValueError(f"Pattern '{pattern_name}' not available")
    
    def set_pattern(self, pattern_name: str) -> bool:
        """Set neuron to specific biological pattern"""
        if pattern_name in self.patterns:
            self.current_pattern = pattern_name
            self.parameters = self.patterns[pattern_name].copy()
            
            # Reset state with new pattern
            self.v = self.parameters['c']
            self.u = self.parameters['b'] * self.v
            
            return True
        return False
    
    # AURAModule interface methods
    def initialize(self) -> bool:
        """Initialize the module. Return True if successful."""
        try:
            # Reset state with current parameters
            self.v = self.parameters['c']
            self.u = self.parameters['b'] * self.v
            self.spike_count = 0
            self.simulation_time_total = 0.0
            return True
        except Exception:
            return False
    
    def process(self, input_data: Any) -> Any:
        """Process input data according to module function."""
        if isinstance(input_data, (int, float)):
            # Single current value
            spikes, voltage = self.simulate(input_data, 100.0, 0.1)
            return {'spikes': spikes, 'voltage': voltage}
        elif isinstance(input_data, dict) and 'current' in input_data:
            # Dictionary with current and optional time parameters
            current = input_data['current']
            time = input_data.get('time', 100.0)
            dt = input_data.get('dt', 0.1)
            spikes, voltage = self.simulate(current, time, dt)
            return {'spikes': spikes, 'voltage': voltage}
        else:
            # Default processing
            spikes, voltage = self.simulate(0.0, 100.0, 0.1)
            return {'spikes': spikes, 'voltage': voltage}
    
    def get_state(self) -> Dict[str, Any]:
        """Get current module state for hot-swapping."""
        return {
            'module_id': self.module_id,
            'parameters': self.parameters.copy(),
            'current_pattern': self.current_pattern,
            'v': self.v,
            'u': self.u,
            'spike_count': self.spike_count,
            'simulation_time_total': self.simulation_time_total,
            'performance_metrics': self.performance_metrics.copy(),
            'is_active': self.is_active
        }
    
    def set_state(self, state: Dict[str, Any]) -> bool:
        """Set module state during hot-swapping."""
        try:
            self.parameters = state.get('parameters', self.parameters)
            self.current_pattern = state.get('current_pattern', self.current_pattern)
            self.v = state.get('v', self.v)
            self.u = state.get('u', self.u)
            self.spike_count = state.get('spike_count', self.spike_count)
            self.simulation_time_total = state.get('simulation_time_total', self.simulation_time_total)
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            self.is_active = state.get('is_active', self.is_active)
            return True
        except Exception:
            return False
    
    def validate(self) -> Tuple[bool, str]:
        """Validate module functionality. Return (is_valid, message)."""
        try:
            # Check required parameters
            required_params = ['a', 'b', 'c', 'd']
            for param in required_params:
                if param not in self.parameters:
                    return False, f"Missing required parameter: {param}"
            
            # Test simulation
            test_spikes, test_voltage = self.simulate(0.0, 10.0, 0.1)
            if len(test_spikes) != len(test_voltage):
                return False, "Simulation output length mismatch"
            
            return True, "Module validation successful"
        except Exception as e:
            return False, f"Validation failed: {str(e)}"
