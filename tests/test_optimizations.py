#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive test suite for AURA optimization modules.
Tests all high-impact optimizations before deployment.
"""

import unittest
import os
import sys
import tempfile
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

# Import optimization modules
from aura.optimization.tpu_optimizer import (
    DynamicBatchSizer, ExpertSharding, MixedPrecisionOptimizer,
    GradientCheckpointing, OptimizedTPUConfig, create_optimized_training_setup
)
from aura.optimization.neuroplasticity import (
    HebbianLearning, HomeostaticRegulation, NeuroplasticityEngine,
    PlasticityConfig, PlasticExpertCore
)
from aura.optimization.causal_reasoning import (
    CausalDAG, DoCalculus, CounterfactualReasoning, CausalReasoningEngine,
    CausalRelation, CausalRelationType, Intervention
)
from aura.optimization.evolutionary_experts import (
    ExpertEvolutionEngine, ExpertGenome, GeneticOperators,
    accuracy_fitness_function
)
from aura.optimization.meta_learning import (
    MAMLNetwork, MetaExpert, MetaLearningEngine, MetaLearningConfig
)


class TestTPUOptimization(unittest.TestCase):
    """Test TPU optimization components."""
    
    def test_dynamic_batch_sizer(self):
        """Test dynamic batch size computation."""
        batch_sizer = DynamicBatchSizer(base_batch_size=128)
        
        # Test with short sequences
        batch_size_short = batch_sizer.compute_optimal_batch_size(
            sequence_length=256, model_size=1e6
        )
        self.assertGreater(batch_size_short, 0)
        self.assertLessEqual(batch_size_short, 128)
        
        # Test with long sequences
        batch_size_long = batch_sizer.compute_optimal_batch_size(
            sequence_length=2048, model_size=1e6
        )
        self.assertGreater(batch_size_long, 0)
        self.assertLess(batch_size_long, batch_size_short)  # Should be smaller
        
        # Test that batch size is power of 2
        self.assertEqual(batch_size_short & (batch_size_short - 1), 0)
    
    def test_expert_sharding(self):
        """Test expert sharding across cores."""
        num_experts = 16
        num_cores = 8
        sharding = ExpertSharding(num_experts, num_cores)
        
        # Test shard map creation
        self.assertEqual(len(sharding.shard_map), num_experts)
        
        # Test that experts are distributed
        core_assignments = set(sharding.shard_map.values())
        self.assertGreater(len(core_assignments), 1)  # Multiple cores used
        
        # Test expert-to-core mapping
        for expert_idx in range(num_experts):
            core_idx = sharding.get_expert_core(expert_idx)
            self.assertGreaterEqual(core_idx, 0)
            self.assertLess(core_idx, num_cores)
    
    def test_mixed_precision_optimizer(self):
        """Test mixed precision optimization."""
        optimizer = MixedPrecisionOptimizer(loss_scale=2**15)
        
        # Test dtype
        self.assertEqual(optimizer.dtype, jnp.bfloat16)
        
        # Test parameter conversion
        params = {'layer1': jnp.ones((4, 4), dtype=jnp.float32)}
        bf16_params = optimizer.convert_to_bf16(params)
        
        self.assertEqual(bf16_params['layer1'].dtype, jnp.bfloat16)
        
        # Test gradient scaling
        grads = {'layer1': jnp.ones((4, 4))}
        scaled_grads = optimizer.scale_gradients(grads)
        self.assertAlmostEqual(
            float(jnp.mean(scaled_grads['layer1'])),
            optimizer.loss_scale,
            places=1
        )
        
        # Test gradient unscaling
        unscaled_grads = optimizer.unscale_gradients(scaled_grads)
        np.testing.assert_array_almost_equal(
            unscaled_grads['layer1'], grads['layer1'], decimal=5
        )
    
    def test_optimized_tpu_config(self):
        """Test optimized TPU configuration."""
        config = create_optimized_training_setup("medium", 512, 16)
        
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.expert_sharding)
        
        training_config = config.get_training_config()
        self.assertIn('batch_size', training_config)
        self.assertIn('mixed_precision', training_config)
        self.assertTrue(training_config['mixed_precision'])


class TestNeuroplasticity(unittest.TestCase):
    """Test neuroplasticity components."""
    
    def setUp(self):
        self.config = PlasticityConfig(
            hebbian_rate=0.01,
            decay_rate=0.001,
            homeostatic_target=0.1
        )
    
    def test_hebbian_learning(self):
        """Test Hebbian learning mechanism."""
        hebbian = HebbianLearning(self.config)
        
        key = jax.random.key(0)
        pre_activity = jax.random.normal(key, (8, 32))
        post_activity = jax.random.normal(key, (8, 64))
        
        # Update connections
        weights = hebbian.update_connections(
            pre_activity, post_activity, "test_connection"
        )
        
        self.assertEqual(weights.shape, (32, 64))
        self.assertTrue(jnp.all(weights >= 0))  # Non-negative weights
        self.assertTrue(jnp.all(weights <= 2.0))  # Within bounds
        
        # Test repeated updates
        weights2 = hebbian.update_connections(
            pre_activity, post_activity, "test_connection"
        )
        self.assertFalse(jnp.allclose(weights, weights2))  # Should change
    
    def test_homeostatic_regulation(self):
        """Test homeostatic regulation."""
        regulation = HomeostaticRegulation(self.config)
        
        key = jax.random.key(0)
        activity = jax.random.normal(key, (8, 64)) * 10.0  # High activity
        
        # Regulate activity
        regulated = regulation.regulate_activity(activity, "test_neurons")
        
        # Activity should be scaled
        self.assertNotEqual(float(jnp.mean(activity)), float(jnp.mean(regulated)))
    
    def test_neuroplasticity_engine(self):
        """Test full neuroplasticity engine."""
        engine = NeuroplasticityEngine(self.config)
        
        key = jax.random.key(0)
        expert_activities = {
            'expert_0': jax.random.normal(key, (8, 64)),
            'expert_1': jax.random.normal(key, (8, 64)),
            'expert_2': jax.random.normal(key, (8, 64))
        }
        expert_rewards = {'expert_0': 0.8, 'expert_1': 0.6, 'expert_2': 0.9}
        
        # Update connections
        connections = engine.update_expert_connections(
            expert_activities, expert_rewards
        )
        
        self.assertGreater(len(connections), 0)
        
        # Check plasticity state
        state = engine.get_plasticity_state()
        self.assertGreater(state['connection_count'], 0)
        self.assertGreater(state['plasticity_events'], 0)
    
    def test_plastic_expert_core(self):
        """Test plastic expert core integration."""
        key = jax.random.key(0)
        
        core = PlasticExpertCore(
            hidden_dim=64,
            num_experts=4,
            plasticity_config=self.config
        )
        
        # Initialize
        test_input = jax.random.normal(key, (4, 32))
        params = core.init(key, test_input)
        
        # Forward pass without plasticity update
        output, info = core.apply(
            params, test_input, 
            expert_rewards=None, 
            update_plasticity=False
        )
        
        self.assertEqual(output.shape, (4, 64))
        self.assertIsInstance(info, dict)


class TestCausalReasoning(unittest.TestCase):
    """Test causal reasoning components."""
    
    def test_causal_dag(self):
        """Test causal DAG construction."""
        dag = CausalDAG()
        
        # Add causal relations
        relation1 = CausalRelation(
            cause="X", effect="Y",
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, confidence=0.9,
            evidence=["test"]
        )
        dag.add_causal_relation(relation1)
        
        # Test graph structure
        self.assertIn("X", dag.graph.nodes())
        self.assertIn("Y", dag.graph.nodes())
        self.assertTrue(dag.graph.has_edge("X", "Y"))
        
        # Test parent/child queries
        parents = dag.get_parents("Y")
        self.assertIn("X", parents)
        
        children = dag.get_children("X")
        self.assertIn("Y", children)
    
    def test_do_calculus(self):
        """Test do-calculus interventions."""
        dag = CausalDAG()
        relation = CausalRelation(
            cause="treatment", effect="outcome",
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.7, confidence=0.85,
            evidence=[]
        )
        dag.add_causal_relation(relation)
        
        do_calc = DoCalculus(dag)
        
        # Create test data
        key = jax.random.key(0)
        data_dist = {
            'treatment': jax.random.normal(key, (100,)),
            'outcome': jax.random.normal(key, (100,))
        }
        
        # Apply intervention
        intervention = Intervention('treatment', 1.0, 'do')
        intervened_dist = do_calc.apply_intervention(intervention, data_dist)
        
        self.assertIn('treatment', intervened_dist)
        self.assertIn('outcome', intervened_dist)
    
    def test_counterfactual_reasoning(self):
        """Test counterfactual generation."""
        dag = CausalDAG()
        relation = CausalRelation(
            cause="X", effect="Y",
            relation_type=CausalRelationType.DIRECT_CAUSE,
            strength=0.8, confidence=0.9,
            evidence=[]
        )
        dag.add_causal_relation(relation)
        
        cf_reasoning = CounterfactualReasoning(dag)
        
        # Add structural equation
        cf_reasoning.add_structural_equation(
            'Y',
            lambda parent_data: parent_data.get('X', jnp.zeros(1)) * 2.0
        )
        
        # Generate counterfactual
        key = jax.random.key(0)
        factual_data = {
            'X': jax.random.normal(key, (10,)),
            'Y': jax.random.normal(key, (10,))
        }
        
        intervention = Intervention('X', 5.0)
        cf_data = cf_reasoning.generate_counterfactual(factual_data, intervention)
        
        self.assertIn('X', cf_data)
        self.assertIn('Y', cf_data)
        self.assertEqual(float(jnp.mean(cf_data['X'])), 5.0)
    
    def test_causal_reasoning_engine(self):
        """Test full causal reasoning engine."""
        engine = CausalReasoningEngine()
        
        # Create sample data
        key = jax.random.key(0)
        data = {
            'var1': jax.random.normal(key, (100,)),
            'var2': jax.random.normal(key, (100,)),
            'var3': jax.random.normal(key, (100,))
        }
        
        # Learn causal structure
        dag = engine.learn_causal_structure(data, method="correlation_based")
        
        self.assertIsNotNone(dag)
        self.assertGreater(len(dag.graph.nodes()), 0)
        
        # Get insights
        insights = engine.get_causal_insights()
        self.assertIn('total_variables', insights)
        self.assertIn('total_relationships', insights)


class TestEvolutionaryExperts(unittest.TestCase):
    """Test evolutionary expert system."""
    
    def test_genetic_operators(self):
        """Test genetic operators."""
        from aura.optimization.evolutionary_experts import ExpertGene, ExpertGeneType
        
        # Test mutation
        gene = ExpertGene(ExpertGeneType.HIDDEN_SIZE, 128, mutation_rate=1.0)
        mutated = GeneticOperators.mutate_gene(gene, mutation_strength=1.0)
        
        self.assertEqual(mutated.gene_type, gene.gene_type)
        # Value might change due to mutation
    
    def test_evolution_engine_initialization(self):
        """Test evolution engine initialization."""
        engine = ExpertEvolutionEngine(population_size=10)
        
        population = engine.initialize_population(input_dim=32, output_dim=10)
        
        self.assertEqual(len(population), 10)
        
        for genome in population:
            self.assertIsInstance(genome, ExpertGenome)
            self.assertGreater(len(genome.genes), 0)
    
    def test_fitness_evaluation(self):
        """Test fitness evaluation."""
        engine = ExpertEvolutionEngine(population_size=5)
        population = engine.initialize_population(input_dim=32, output_dim=3)
        
        # Create test data
        key = jax.random.key(0)
        input_data = jax.random.normal(key, (20, 32))
        target_data = jax.random.randint(key, (20,), 0, 3)
        
        # Evaluate one genome
        genome = population[0]
        fitness = engine.evaluate_fitness(
            genome, accuracy_fitness_function, input_data, target_data
        )
        
        self.assertIsInstance(fitness, float)
        self.assertGreaterEqual(fitness, -1000.0)  # Not catastrophically bad
    
    def test_evolution_step(self):
        """Test one evolution step."""
        engine = ExpertEvolutionEngine(population_size=5)
        engine.initialize_population(input_dim=32, output_dim=3)
        
        # Create test data
        key = jax.random.key(0)
        input_data = jax.random.normal(key, (20, 32))
        target_data = jax.random.randint(key, (20,), 0, 3)
        
        # Evolve one generation
        new_population = engine.evolve_generation(
            accuracy_fitness_function, input_data, target_data
        )
        
        self.assertEqual(len(new_population), 5)
        self.assertEqual(engine.generation, 1)
        
        # Check that best genome is tracked
        self.assertIsNotNone(engine.best_genome)


class TestMetaLearning(unittest.TestCase):
    """Test meta-learning components."""
    
    def test_maml_network(self):
        """Test MAML network."""
        key = jax.random.key(0)
        
        network = MAMLNetwork(
            hidden_dims=[64, 32],
            output_dim=5
        )
        
        # Initialize and test
        test_input = jax.random.normal(key, (4, 16))
        params = network.init(key, test_input)
        output = network.apply(params, test_input)
        
        self.assertEqual(output.shape, (4, 5))
    
    def test_meta_expert(self):
        """Test meta-expert."""
        key = jax.random.key(0)
        
        config = MetaLearningConfig()
        meta_expert = MetaExpert(
            base_hidden_dim=64,
            output_dim=3,
            meta_config=config
        )
        
        # Initialize
        test_input = jax.random.normal(key, (4, 32))
        params = meta_expert.init(key, test_input)
        
        # Forward pass
        output = meta_expert.apply(params, test_input)
        self.assertEqual(output.shape, (4, 3))
        
        # Test fast adaptation
        support_x = jax.random.normal(key, (10, 32))
        support_y = jax.random.normal(key, (10, 3))
        
        adapted_params = meta_expert.fast_adapt(
            params, support_x, support_y,
            learning_rate=0.01, num_steps=3
        )
        
        self.assertIsNotNone(adapted_params)
    
    def test_meta_learning_engine(self):
        """Test meta-learning engine."""
        config = MetaLearningConfig(
            support_shots=5,
            query_shots=10
        )
        engine = MetaLearningEngine(config)
        
        # Create meta-expert
        meta_expert = engine.create_meta_expert(
            input_dim=16, output_dim=3, expert_type="maml"
        )
        
        self.assertIsNotNone(meta_expert)
        
        # Generate few-shot task
        key = jax.random.key(0)
        data_x = jax.random.normal(key, (100, 16))
        data_y = jax.random.randint(key, (100,), 0, 5)
        
        support_x, support_y, query_x, query_y = engine.generate_few_shot_task(
            data_x, data_y, n_classes=3,
            support_shots=5, query_shots=10
        )
        
        self.assertEqual(support_x.shape[0], 15)  # 3 classes * 5 shots
        self.assertEqual(query_x.shape[0], 30)    # 3 classes * 10 shots


class TestIntegration(unittest.TestCase):
    """Integration tests for optimization deployment."""
    
    def test_deployment_config(self):
        """Test deployment configuration."""
        from aura.optimization.tpu_optimizer import OptimizedTPUConfig
        
        config = OptimizedTPUConfig("medium", 512, 32.0)
        config.setup_expert_sharding(16, 8)
        
        training_config = config.get_training_config()
        
        self.assertIsNotNone(training_config)
        self.assertIn('batch_size', training_config)
        self.assertIn('mixed_precision', training_config)
    
    def test_combined_optimizations(self):
        """Test combining multiple optimizations."""
        # TPU config
        tpu_config = create_optimized_training_setup("small", 256, 8)
        
        # Neuroplasticity
        plasticity_config = PlasticityConfig()
        neuroplasticity = NeuroplasticityEngine(plasticity_config)
        
        # Causal reasoning
        causal_engine = CausalReasoningEngine()
        
        # All components should work together
        self.assertIsNotNone(tpu_config)
        self.assertIsNotNone(neuroplasticity)
        self.assertIsNotNone(causal_engine)


class TestDeploymentScript(unittest.TestCase):
    """Test deployment script functionality."""
    
    def test_config_loading(self):
        """Test configuration file loading."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "model_size": "large",
                "num_experts": 32,
                "enable_tpu_optimization": True
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            # Test that config can be loaded
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            
            self.assertEqual(loaded_config['model_size'], 'large')
            self.assertEqual(loaded_config['num_experts'], 32)
            
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    # Set JAX to use CPU for testing
    jax.config.update('jax_platform_name', 'cpu')
    
    # Run tests with verbose output
    unittest.main(verbosity=2)
