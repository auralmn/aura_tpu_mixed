"""
Configuration loader for AURA_GENESIS system
Handles loading and validation of YAML configuration files
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuraConfig:
    """AURA system configuration loaded from YAML"""
    
    # System information
    system_name: str = "AURA_GENESIS"
    version: str = "2.0.0"
    description: str = "Advanced Neural Network System"
    
    # Boot configuration
    validate_dependencies: bool = True
    initialize_weights: bool = True
    enable_health_checks: bool = True
    timeout_seconds: int = 60
    retry_attempts: int = 3
    safe_mode: bool = True
    
    # Network architecture
    neuron_count: int = 1000
    features: int = 384
    input_channels: int = 384
    output_channels: int = 384
    enable_span: bool = False
    span_neurons_per_region: int = 10
    domains: list = field(default_factory=list)
    realms: list = field(default_factory=list)
    domain_labels_path: str = "/Volumes/Others2/AURA_GENESIS/svc_domain_labels.json"
    offline: bool = True
    nlms_clamp: tuple = (0.0, 1.0)
    nlms_l2: float = 1e-4
    features_mode: str = "sbert"
    features_alpha: float = 0.7
    weights_dir: str = "svc_nlms_weights"
    startnew: bool = False
    
    # Brain region configuration
    thalamus_neuron_count: int = 100
    thalamus_input_channels: int = 384
    thalamus_output_channels: int = 384
    hippocampus_neuron_count: int = 100
    hippocampus_features: int = 384
    hippocampus_input_dim: int = 384
    amygdala_neuron_count: int = 30
    amygdala_features: int = 384
    amygdala_input_dim: int = 384
    thalamic_router_neuron_count: int = 60
    thalamic_router_features: int = 384
    thalamic_router_input_dim: int = 384
    cns_input_dim: int = 384
    
    # Paths
    weights_dir_path: str = "/Volumes/Others2/AURA_GENESIS/weights"
    models_dir_path: str = "/Volumes/Others2/AURA_GENESIS/models"
    svc_data_path: Optional[str] = None
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Models
    model_files: Dict[str, str] = field(default_factory=lambda: {
        'emotion_classifier': 'clf_emotion.pt',
        'intent_classifier': 'clf_intent.pt',
        'tone_classifier': 'clf_tone.pt',
        'svc_domain_classifier': 'svc_domain_classifier_enhanced.pt',
        'svc_realm_classifier': 'svc_realm_classifier_enhanced.pt',
        'svc_difficulty_regressor': 'svc_difficulty_regressor_enhanced.pt'
    })
    
    # SVC Analysis
    enable_svc_analysis: bool = True
    linguistic_features_enabled: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    memory_threshold_mb: int = 1024
    cpu_threshold_percent: float = 80.0
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "aura_system.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Device
    device_type: str = "mps"
    fallback_to_cpu: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Security
    enable_encryption: bool = False
    api_key_required: bool = False
    rate_limiting: bool = False
    max_requests_per_minute: int = 100
    
    # Development
    debug_mode: bool = False
    verbose_logging: bool = False
    profile_performance: bool = False
    save_debug_info: bool = False


def load_aura_config(config_path: Optional[str] = None) -> AuraConfig:
    """Convenience function to load AURA configuration with better error handling"""
    try:
        loader = ConfigLoader(config_path)
        config = loader.load_config()
        
        # Try to validate, but provide more specific error information
        if not loader.validate_config(config):
            print("Warning: Configuration validation failed, attempting to fix common issues...")
            
            # Create a default configuration if validation fails
            config = create_default_aura_config()
            
            # Try validation again with default config
            if not loader.validate_config(config):
                print("Warning: Using minimal safe configuration due to validation issues")
                config = create_minimal_safe_config()
        
        return config
        
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Creating default configuration...")
        return create_default_aura_config()
        
    except Exception as e:
        print(f"Configuration loading error: {e}")
        print("Falling back to minimal safe configuration...")
        return create_minimal_safe_config()


def create_default_aura_config() -> AuraConfig:
    """Create a default AURA configuration"""
    return AuraConfig(
        system_name="AURA_GENESIS",
        version="1.0.0",
        log_level="INFO",
        offline=True,
        
        # Network configuration
        neuron_count=1000,
        features=256,
        enable_span=True,
        span_neurons_per_region=100,
        
        # SVC configuration
        domains=["general", "technical", "creative"],
        realms=["education", "research", "application"],
        features_mode="enhanced",
        features_alpha=0.5,
        
        # Training configuration
        weights_dir_path="./weights",
        models_dir_path="./models",
        performance_monitoring=True,
        
        # System monitoring
        health_check_interval=60,
        metrics_collection_interval=30,
        
        # SVC Analysis configuration
        enable_svc_analysis=True,
        svc_data_path="./data/svc",
        linguistic_features_enabled=True,
        
        # Device configuration
        device_type="cpu",
        fallback_to_cpu=True,
        model_files={},
        nlms_clamp=2.0,
        nlms_l2=0.0,
        input_channels=256,
        output_channels=256,
        startnew=True,
        
        # Brain region configuration
        thalamus_neuron_count=100,
        thalamus_input_channels=256,
        thalamus_output_channels=128,
        hippocampus_neuron_count=200,
        hippocampus_features=256,
        hippocampus_input_dim=256,
        amygdala_neuron_count=150,
        amygdala_features=256,
        amygdala_input_dim=256,
        thalamic_router_neuron_count=50,
        thalamic_router_features=128,
        thalamic_router_input_dim=128,
        cns_input_dim=256
    )


def create_minimal_safe_config() -> AuraConfig:
    """Create a minimal safe configuration that should always work"""
    return AuraConfig(
        system_name="AURA_GENESIS_SAFE",
        version="1.0.0",
        log_level="WARNING",
        offline=True,
        
        # Minimal network configuration
        neuron_count=100,
        features=64,
        enable_span=False,
        span_neurons_per_region=10,
        
        # Minimal SVC configuration
        domains=["general"],
        realms=["basic"],
        features_mode="basic",
        features_alpha=0.3,
        
        # Minimal paths (current directory)
        weights_dir_path="./",
        models_dir_path="./",
        performance_monitoring=False,
        
        # Minimal monitoring
        health_check_interval=300,
        metrics_collection_interval=300,
        
        # Disabled SVC Analysis for safety
        enable_svc_analysis=False,
        svc_data_path="./",
        linguistic_features_enabled=False,
        
        # Safe device configuration
        device_type="cpu",
        fallback_to_cpu=True,
        model_files={},
        nlms_clamp=1.0,
        nlms_l2=0.0,
        input_channels=64,
        output_channels=64,
        startnew=True,
        
        # Minimal brain regions
        thalamus_neuron_count=20,
        thalamus_input_channels=64,
        thalamus_output_channels=32,
        hippocampus_neuron_count=50,
        hippocampus_features=64,
        hippocampus_input_dim=64,
        amygdala_neuron_count=30,
        amygdala_features=64,
        amygdala_input_dim=64,
        thalamic_router_neuron_count=10,
        thalamic_router_features=32,
        thalamic_router_input_dim=32,
        cns_input_dim=64
    )


class ConfigLoader:
    """Enhanced ConfigLoader with better validation and error handling"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self.get_default_config_path()
    
    def get_default_config_path(self) -> str:
        """Get the default configuration file path"""
        # Look for config file in several common locations
        possible_paths = [
            "./config.yaml",
            "./aura_config.yaml", 
            "./configs/aura_config.yaml",
            "./aura/configs/config.yaml",
            "~/.aura/config.yaml"
        ]
        
        import os
        for path in possible_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.exists(expanded_path):
                return expanded_path
        
        # Return first option as default (will create if needed)
        return possible_paths[0]
    
    def load_config(self) -> AuraConfig:
        """Load configuration with enhanced error handling"""
        import os
        
        if not os.path.exists(self.config_path):
            print(f"Configuration file not found at {self.config_path}")
            print("Creating default configuration file...")
            self.create_default_config_file()
        
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert dict to AuraConfig object
            return self.dict_to_aura_config(config_data)
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def create_default_config_file(self):
        """Create a default configuration YAML file"""
        import yaml
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
        
        default_config = {
            'system_name': 'AURA_GENESIS',
            'version': '1.0.0',
            'log_level': 'INFO',
            'offline': True,
            'neuron_count': 1000,
            'features': 256,
            'enable_span': True,
            'span_neurons_per_region': 100,
            'domains': ['general', 'technical', 'creative'],
            'realms': ['education', 'research', 'application'],
            'features_mode': 'enhanced',
            'features_alpha': 0.5,
            'weights_dir_path': './weights',
            'models_dir_path': './models',
            'performance_monitoring': True,
            'health_check_interval': 60,
            'metrics_collection_interval': 30,
            'enable_svc_analysis': True,
            'svc_data_path': './data/svc',
            'linguistic_features_enabled': True,
            'device_type': 'cpu',
            'fallback_to_cpu': True,
            'model_files': {},
            'nlms_clamp': 2.0,
            'nlms_l2': 0.0,
            'input_channels': 256,
            'output_channels': 256,
            'startnew': True,
            'thalamus_neuron_count': 100,
            'thalamus_input_channels': 256,
            'thalamus_output_channels': 128,
            'hippocampus_neuron_count': 200,
            'hippocampus_features': 256,
            'hippocampus_input_dim': 256,
            'amygdala_neuron_count': 150,
            'amygdala_features': 256,
            'amygdala_input_dim': 256,
            'thalamic_router_neuron_count': 50,
            'thalamic_router_features': 128,
            'thalamic_router_input_dim': 128,
            'cns_input_dim': 256
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"Created default configuration file: {self.config_path}")
    
    def dict_to_aura_config(self, config_data: dict) -> AuraConfig:
        """Convert dictionary to AuraConfig object"""
        # This is a simplified version - you'll need to implement the full AuraConfig class
        # For now, create a simple object with the required attributes
        config = type('AuraConfig', (), {})()
        
        for key, value in config_data.items():
            setattr(config, key, value)
        
        return config
    
    def validate_config(self, config: AuraConfig) -> bool:
        """Enhanced configuration validation"""
        try:
            # Check required attributes
            required_attrs = [
                'system_name', 'version', 'neuron_count', 'features',
                'domains', 'realms', 'weights_dir_path', 'models_dir_path'
            ]
            
            for attr in required_attrs:
                if not hasattr(config, attr):
                    print(f"Missing required configuration: {attr}")
                    return False
                    
                value = getattr(config, attr)
                if value is None:
                    print(f"Configuration {attr} cannot be None")
                    return False
            
            # Validate specific types and ranges
            if not isinstance(config.neuron_count, int) or config.neuron_count <= 0:
                print("neuron_count must be a positive integer")
                return False
            
            if not isinstance(config.features, int) or config.features <= 0:
                print("features must be a positive integer")
                return False
            
            if not isinstance(config.domains, list) or len(config.domains) == 0:
                print("domains must be a non-empty list")
                return False
            
            if not isinstance(config.realms, list) or len(config.realms) == 0:
                print("realms must be a non-empty list")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration validation error: {e}")
            return False