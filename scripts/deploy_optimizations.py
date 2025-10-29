#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Optimization Deployment Script for AURA
Deploy all high-impact optimizations and advanced features
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np

# Import optimization modules
from aura.optimization.tpu_optimizer import create_optimized_training_setup, OptimizedTPUConfig
from aura.optimization.neuroplasticity import NeuroplasticityEngine, PlasticityConfig, PlasticExpertCore
from aura.optimization.causal_reasoning import CausalReasoningEngine, CausalConsciousnessModule
from aura.optimization.evolutionary_experts import ExpertEvolutionEngine, accuracy_fitness_function
from aura.optimization.meta_learning import MetaLearningEngine, MetaLearningConfig

# Import AURA core components
from aura.bio_inspired.enhanced_spiking_retrieval import EnhancedSpikingRetrievalCore
from aura.consciousness.aura_consciousness_system import AURAConsciousnessSystem


class OptimizationDeployment:
    """Main class for deploying AURA optimizations."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.deployment_log = []
        
        # Initialize optimization components
        self.tpu_config = None
        self.neuroplasticity_engine = None
        self.causal_engine = None
        self.evolution_engine = None
        self.meta_learning_engine = None
        
        print("🚀 AURA Optimization Deployment System Initialized")
    
    def _load_config(self, config_path: str) -> dict:
        """Load deployment configuration."""
        default_config = {
            "model_size": "medium",
            "sequence_length": 512,
            "num_experts": 16,
            "enable_tpu_optimization": True,
            "enable_neuroplasticity": True,
            "enable_causal_reasoning": True,
            "enable_evolutionary_experts": True,
            "enable_meta_learning": True,
            "tpu": {
                "available_memory_gb": 32.0,
                "target_batch_size": 128
            },
            "neuroplasticity": {
                "hebbian_rate": 0.01,
                "decay_rate": 0.001,
                "homeostatic_target": 0.1
            },
            "evolution": {
                "population_size": 20,
                "generations": 10,
                "mutation_rate": 0.1
            },
            "meta_learning": {
                "inner_learning_rate": 0.01,
                "outer_learning_rate": 0.001,
                "support_shots": 5
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def deploy_tpu_optimization(self):
        """Deploy TPU performance optimizations."""
        if not self.config["enable_tpu_optimization"]:
            return
        
        print("\n📈 Deploying TPU Optimizations...")
        
        self.tpu_config = create_optimized_training_setup(
            model_size=self.config["model_size"],
            sequence_length=self.config["sequence_length"],
            num_experts=self.config["num_experts"]
        )
        
        training_config = self.tpu_config.get_training_config()
        
        self.deployment_log.append({
            "component": "TPU Optimization",
            "status": "deployed",
            "config": training_config
        })
        
        print("✅ TPU optimizations deployed successfully")
        return self.tpu_config
    
    def deploy_neuroplasticity(self):
        """Deploy neuroplasticity engine."""
        if not self.config["enable_neuroplasticity"]:
            return
        
        print("\n🧠 Deploying Neuroplasticity Engine...")
        
        plasticity_config = PlasticityConfig(
            hebbian_rate=self.config["neuroplasticity"]["hebbian_rate"],
            decay_rate=self.config["neuroplasticity"]["decay_rate"],
            homeostatic_target=self.config["neuroplasticity"]["homeostatic_target"]
        )
        
        self.neuroplasticity_engine = NeuroplasticityEngine(plasticity_config)
        
        # Test with sample expert activities
        key = jax.random.key(0)
        expert_activities = {
            f'expert_{i}': jax.random.normal(key, (8, 64))
            for i in range(min(4, self.config["num_experts"]))
        }
        expert_rewards = {f'expert_{i}': np.random.uniform(0.5, 1.0) for i in range(4)}
        
        # Initialize connections
        connections = self.neuroplasticity_engine.update_expert_connections(
            expert_activities, expert_rewards
        )
        
        self.deployment_log.append({
            "component": "Neuroplasticity Engine",
            "status": "deployed",
            "connections_initialized": len(connections),
            "config": plasticity_config.__dict__
        })
        
        print(f"✅ Neuroplasticity engine deployed with {len(connections)} connections")
        return self.neuroplasticity_engine
    
    def deploy_causal_reasoning(self):
        """Deploy causal reasoning engine."""
        if not self.config["enable_causal_reasoning"]:
            return
        
        print("\n🔬 Deploying Causal Reasoning Engine...")
        
        self.causal_engine = CausalReasoningEngine()
        
        # Initialize with sample causal relationships
        key = jax.random.key(0)
        sample_data = {
            'input_complexity': jax.random.normal(key, (100,)),
            'expert_utilization': jax.random.normal(key, (100,)),
            'performance': jax.random.normal(key, (100,)),
        }
        
        # Learn causal structure
        causal_dag = self.causal_engine.learn_causal_structure(sample_data)
        insights = self.causal_engine.get_causal_insights()
        
        self.deployment_log.append({
            "component": "Causal Reasoning Engine",
            "status": "deployed",
            "causal_insights": insights
        })
        
        print(f"✅ Causal reasoning deployed with {insights['total_relationships']} relationships")
        return self.causal_engine
    
    def deploy_evolutionary_experts(self):
        """Deploy evolutionary expert system."""
        if not self.config["enable_evolutionary_experts"]:
            return
        
        print("\n🧬 Deploying Evolutionary Expert System...")
        
        evolution_config = self.config["evolution"]
        self.evolution_engine = ExpertEvolutionEngine(
            population_size=evolution_config["population_size"],
            mutation_rate=evolution_config["mutation_rate"]
        )
        
        # Initialize population
        population = self.evolution_engine.initialize_population(
            input_dim=64, output_dim=10
        )
        
        # Run a few evolution steps with sample data
        key = jax.random.key(0)
        sample_input = jax.random.normal(key, (50, 64))
        sample_target = jax.random.randint(key, (50,), 0, 10)
        
        print(f"   Running {min(3, evolution_config['generations'])} evolution steps...")
        for gen in range(min(3, evolution_config["generations"])):
            population = self.evolution_engine.evolve_generation(
                accuracy_fitness_function, sample_input, sample_target
            )
            
            best_fitness = max(genome.fitness for genome in population)
            print(f"   Generation {gen + 1}: Best fitness = {best_fitness:.3f}")
        
        summary = self.evolution_engine.get_evolution_summary()
        
        self.deployment_log.append({
            "component": "Evolutionary Experts",
            "status": "deployed",
            "evolution_summary": summary
        })
        
        print(f"✅ Evolutionary experts deployed after {summary['generations']} generations")
        return self.evolution_engine
    
    def deploy_meta_learning(self):
        """Deploy meta-learning system."""
        if not self.config["enable_meta_learning"]:
            return
        
        print("\n🎯 Deploying Meta-Learning System...")
        
        meta_config = MetaLearningConfig(
            inner_learning_rate=self.config["meta_learning"]["inner_learning_rate"],
            outer_learning_rate=self.config["meta_learning"]["outer_learning_rate"],
            support_shots=self.config["meta_learning"]["support_shots"]
        )
        
        self.meta_learning_engine = MetaLearningEngine(meta_config)
        
        # Create sample meta-expert
        meta_expert = self.meta_learning_engine.create_meta_expert(
            input_dim=32, output_dim=5, expert_type="maml"
        )
        
        # Test few-shot capability
        key = jax.random.key(0)
        sample_data_x = jax.random.normal(key, (200, 32))
        sample_data_y = jax.random.randint(key, (200,), 0, 5)
        
        support_x, support_y, query_x, query_y = self.meta_learning_engine.generate_few_shot_task(
            sample_data_x, sample_data_y, n_classes=3,
            support_shots=meta_config.support_shots, query_shots=10
        )
        
        # Initialize and test
        training_state = self.meta_learning_engine.create_meta_training_state(
            meta_expert, (32,)
        )
        
        metrics = self.meta_learning_engine.few_shot_evaluate(
            meta_expert, training_state.params,
            support_x, support_y, query_x, query_y
        )
        
        self.deployment_log.append({
            "component": "Meta-Learning System",
            "status": "deployed",
            "few_shot_metrics": metrics,
            "config": meta_config.__dict__
        })
        
        print(f"✅ Meta-learning deployed with {meta_config.support_shots}-shot capability")
        return self.meta_learning_engine
    
    def create_integrated_system(self):
        """Create integrated AURA system with all optimizations."""
        print("\n🔧 Creating Integrated Optimized AURA System...")
        
        # Enhanced retrieval core with plasticity
        if self.neuroplasticity_engine:
            plasticity_config = PlasticityConfig()
            enhanced_core = PlasticExpertCore(
                hidden_dim=256,
                num_experts=self.config["num_experts"],
                plasticity_config=plasticity_config
            )
        else:
            enhanced_core = EnhancedSpikingRetrievalCore(
                hidden_dim=256,
                num_experts=self.config["num_experts"]
            )
        
        # Consciousness system with causal reasoning
        consciousness_system = AURAConsciousnessSystem()
        
        if self.causal_engine:
            causal_consciousness = CausalConsciousnessModule(hidden_dim=256)
        else:
            causal_consciousness = None
        
        integrated_system = {
            "enhanced_core": enhanced_core,
            "consciousness_system": consciousness_system,
            "causal_consciousness": causal_consciousness,
            "tpu_config": self.tpu_config,
            "neuroplasticity_engine": self.neuroplasticity_engine,
            "causal_engine": self.causal_engine,
            "evolution_engine": self.evolution_engine,
            "meta_learning_engine": self.meta_learning_engine
        }
        
        print("✅ Integrated system created successfully")
        return integrated_system
    
    def run_performance_benchmarks(self, integrated_system):
        """Run performance benchmarks on the integrated system."""
        print("\n📊 Running Performance Benchmarks...")
        
        benchmarks = {}
        
        # Memory usage benchmark
        try:
            devices = jax.devices()
            benchmarks["available_devices"] = len(devices)
            benchmarks["device_types"] = [str(device.device_kind) for device in devices]
        except Exception as e:
            benchmarks["device_error"] = str(e)
        
        # TPU configuration benchmark
        if self.tpu_config:
            training_config = self.tpu_config.get_training_config()
            benchmarks["optimized_batch_size"] = training_config["batch_size"]
            benchmarks["mixed_precision"] = training_config["mixed_precision"]
        
        # Neuroplasticity benchmark
        if self.neuroplasticity_engine:
            plasticity_state = self.neuroplasticity_engine.get_plasticity_state()
            benchmarks["plasticity_connections"] = plasticity_state["connection_count"]
            benchmarks["plasticity_events"] = plasticity_state["plasticity_events"]
        
        # Causal reasoning benchmark
        if self.causal_engine:
            causal_insights = self.causal_engine.get_causal_insights()
            benchmarks["causal_variables"] = causal_insights["total_variables"]
            benchmarks["causal_relationships"] = causal_insights["total_relationships"]
        
        # Evolution benchmark
        if self.evolution_engine:
            evolution_summary = self.evolution_engine.get_evolution_summary()
            benchmarks["evolution_generations"] = evolution_summary["generations"]
            benchmarks["best_fitness"] = evolution_summary["best_fitness"]
        
        print("✅ Performance benchmarks completed")
        return benchmarks
    
    def save_deployment_report(self, integrated_system, benchmarks):
        """Save comprehensive deployment report."""
        report = {
            "deployment_timestamp": str(np.datetime64('now')),
            "configuration": self.config,
            "deployment_log": self.deployment_log,
            "benchmarks": benchmarks,
            "system_components": {
                "enhanced_core": integrated_system["enhanced_core"] is not None,
                "consciousness_system": integrated_system["consciousness_system"] is not None,
                "causal_consciousness": integrated_system["causal_consciousness"] is not None,
                "tpu_config": integrated_system["tpu_config"] is not None,
                "neuroplasticity_engine": integrated_system["neuroplasticity_engine"] is not None,
                "causal_engine": integrated_system["causal_engine"] is not None,
                "evolution_engine": integrated_system["evolution_engine"] is not None,
                "meta_learning_engine": integrated_system["meta_learning_engine"] is not None
            }
        }
        
        report_path = "optimization_deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📋 Deployment report saved to {report_path}")
        return report
    
    def deploy_all(self):
        """Deploy all optimizations in sequence."""
        print("🚀 Starting Full AURA Optimization Deployment")
        print("=" * 60)
        
        # Deploy individual components
        self.deploy_tpu_optimization()
        self.deploy_neuroplasticity()
        self.deploy_causal_reasoning()
        self.deploy_evolutionary_experts()
        self.deploy_meta_learning()
        
        # Create integrated system
        integrated_system = self.create_integrated_system()
        
        # Run benchmarks
        benchmarks = self.run_performance_benchmarks(integrated_system)
        
        # Save report
        report = self.save_deployment_report(integrated_system, benchmarks)
        
        print("\n" + "=" * 60)
        print("🎉 AURA OPTIMIZATION DEPLOYMENT COMPLETE!")
        print("=" * 60)
        
        # Summary
        deployed_components = sum(1 for log in self.deployment_log if log["status"] == "deployed")
        print(f"✅ Successfully deployed {deployed_components} optimization components")
        
        if benchmarks.get("optimized_batch_size"):
            print(f"📈 TPU batch size optimized to: {benchmarks['optimized_batch_size']}")
        
        if benchmarks.get("plasticity_connections"):
            print(f"🧠 Neuroplasticity initialized with {benchmarks['plasticity_connections']} connections")
        
        if benchmarks.get("causal_relationships"):
            print(f"🔬 Causal reasoning with {benchmarks['causal_relationships']} relationships")
        
        if benchmarks.get("evolution_generations"):
            print(f"🧬 Evolutionary system ran {benchmarks['evolution_generations']} generations")
        
        print(f"\n📋 Full report available in: optimization_deployment_report.json")
        
        return integrated_system, report


def main():
    parser = argparse.ArgumentParser(description="Deploy AURA optimizations")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], 
                       default="medium", help="Model size preset")
    parser.add_argument("--sequence-length", type=int, default=512, 
                       help="Maximum sequence length")
    parser.add_argument("--num-experts", type=int, default=16, 
                       help="Number of experts")
    parser.add_argument("--disable-tpu", action="store_true", 
                       help="Disable TPU optimizations")
    parser.add_argument("--disable-neuroplasticity", action="store_true",
                       help="Disable neuroplasticity")
    parser.add_argument("--disable-causal", action="store_true",
                       help="Disable causal reasoning")
    parser.add_argument("--disable-evolution", action="store_true",
                       help="Disable evolutionary experts")
    parser.add_argument("--disable-meta-learning", action="store_true",
                       help="Disable meta-learning")
    
    args = parser.parse_args()
    
    # Create deployment instance
    deployment = OptimizationDeployment(args.config)
    
    # Override config with command line arguments
    deployment.config.update({
        "model_size": args.model_size,
        "sequence_length": args.sequence_length,
        "num_experts": args.num_experts,
        "enable_tpu_optimization": not args.disable_tpu,
        "enable_neuroplasticity": not args.disable_neuroplasticity,
        "enable_causal_reasoning": not args.disable_causal,
        "enable_evolutionary_experts": not args.disable_evolution,
        "enable_meta_learning": not args.disable_meta_learning
    })
    
    # Deploy all optimizations
    integrated_system, report = deployment.deploy_all()
    
    return integrated_system, report


if __name__ == "__main__":
    integrated_system, report = main()
