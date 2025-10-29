#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Evolutionary Expert Architecture for AURA
Implements genetic programming for automatic expert architecture search and evolution
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
import random
from enum import Enum
import json


class ExpertGeneType(Enum):
    """Types of genes that can be evolved in expert architectures."""
    LAYER_TYPE = "layer_type"
    ACTIVATION = "activation"
    HIDDEN_SIZE = "hidden_size"
    DROPOUT_RATE = "dropout_rate"
    NORMALIZATION = "normalization"
    SKIP_CONNECTION = "skip_connection"
    ATTENTION_HEADS = "attention_heads"
    KERNEL_SIZE = "kernel_size"


@dataclass
class ExpertGene:
    """Individual gene in an expert's genetic code."""
    gene_type: ExpertGeneType
    value: Any
    mutation_rate: float = 0.1
    crossover_weight: float = 0.5


@dataclass
class ExpertGenome:
    """Complete genetic code for an expert architecture."""
    genes: Dict[str, ExpertGene]
    fitness: float = 0.0
    generation: int = 0
    parent_ids: List[str] = None
    
    def __post_init__(self):
        if self.parent_ids is None:
            self.parent_ids = []


class GeneticOperators:
    """Genetic operators for evolution: mutation, crossover, selection."""
    
    @staticmethod
    def mutate_gene(gene: ExpertGene, mutation_strength: float = 1.0) -> ExpertGene:
        """Mutate a single gene based on its type."""
        if random.random() > gene.mutation_rate * mutation_strength:
            return gene  # No mutation
        
        new_gene = ExpertGene(
            gene_type=gene.gene_type,
            value=gene.value,
            mutation_rate=gene.mutation_rate,
            crossover_weight=gene.crossover_weight
        )
        
        if gene.gene_type == ExpertGeneType.LAYER_TYPE:
            layer_types = ["dense", "conv1d", "attention", "rational", "glu"]
            new_gene.value = random.choice(layer_types)
            
        elif gene.gene_type == ExpertGeneType.ACTIVATION:
            activations = ["relu", "gelu", "swish", "tanh", "leaky_relu"]
            new_gene.value = random.choice(activations)
            
        elif gene.gene_type == ExpertGeneType.HIDDEN_SIZE:
            # Mutate hidden size within reasonable bounds
            current_size = gene.value
            mutation_factor = random.uniform(0.5, 2.0)
            new_size = int(current_size * mutation_factor)
            new_gene.value = max(16, min(1024, new_size))
            
        elif gene.gene_type == ExpertGeneType.DROPOUT_RATE:
            # Mutate dropout rate
            current_rate = gene.value
            delta = random.uniform(-0.2, 0.2)
            new_gene.value = max(0.0, min(0.8, current_rate + delta))
            
        elif gene.gene_type == ExpertGeneType.NORMALIZATION:
            norm_types = ["layer_norm", "batch_norm", "group_norm", "none"]
            new_gene.value = random.choice(norm_types)
            
        elif gene.gene_type == ExpertGeneType.SKIP_CONNECTION:
            new_gene.value = not gene.value  # Toggle boolean
            
        elif gene.gene_type == ExpertGeneType.ATTENTION_HEADS:
            current_heads = gene.value
            new_heads = random.choice([1, 2, 4, 8, 16])
            new_gene.value = new_heads
            
        elif gene.gene_type == ExpertGeneType.KERNEL_SIZE:
            kernel_sizes = [1, 3, 5, 7, 9]
            new_gene.value = random.choice(kernel_sizes)
        
        return new_gene
    
    @staticmethod
    def crossover_genomes(parent1: ExpertGenome, 
                         parent2: ExpertGenome,
                         crossover_rate: float = 0.5) -> Tuple[ExpertGenome, ExpertGenome]:
        """Create two offspring through genetic crossover."""
        
        # Combine all gene keys from both parents
        all_genes = set(parent1.genes.keys()) | set(parent2.genes.keys())
        
        child1_genes = {}
        child2_genes = {}
        
        for gene_key in all_genes:
            # Get genes from parents (use default if missing)
            gene1 = parent1.genes.get(gene_key)
            gene2 = parent2.genes.get(gene_key)
            
            if gene1 and gene2:
                # Both parents have this gene - do crossover
                if random.random() < crossover_rate:
                    # Crossover
                    child1_genes[gene_key] = gene2
                    child2_genes[gene_key] = gene1
                else:
                    # No crossover
                    child1_genes[gene_key] = gene1
                    child2_genes[gene_key] = gene2
            elif gene1:
                # Only parent1 has this gene
                if random.random() < 0.5:
                    child1_genes[gene_key] = gene1
                else:
                    child2_genes[gene_key] = gene1
            elif gene2:
                # Only parent2 has this gene
                if random.random() < 0.5:
                    child1_genes[gene_key] = gene2
                else:
                    child2_genes[gene_key] = gene2
        
        child1 = ExpertGenome(
            genes=child1_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[f"gen{parent1.generation}", f"gen{parent2.generation}"]
        )
        
        child2 = ExpertGenome(
            genes=child2_genes,
            generation=max(parent1.generation, parent2.generation) + 1,
            parent_ids=[f"gen{parent1.generation}", f"gen{parent2.generation}"]
        )
        
        return child1, child2
    
    @staticmethod
    def tournament_selection(population: List[ExpertGenome], 
                           tournament_size: int = 3) -> ExpertGenome:
        """Select parent using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    @staticmethod
    def roulette_wheel_selection(population: List[ExpertGenome]) -> ExpertGenome:
        """Select parent using fitness-proportionate selection."""
        total_fitness = sum(genome.fitness for genome in population)
        if total_fitness <= 0:
            return random.choice(population)
        
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        
        for genome in population:
            current_sum += genome.fitness
            if current_sum >= selection_point:
                return genome
        
        return population[-1]  # Fallback


class EvolvableExpert(nn.Module):
    """Expert that can evolve its architecture based on genetic code."""
    
    genome: ExpertGenome
    input_dim: int
    output_dim: int
    
    def setup(self):
        """Build expert architecture from genome."""
        self.layers = []
        
        # Build layers based on genetic code
        current_dim = self.input_dim
        
        # Main processing layers
        for i in range(self._get_num_layers()):
            layer_type = self._get_gene_value(f"layer_type_{i}", "dense")
            hidden_size = self._get_gene_value(f"hidden_size_{i}", 128)
            activation = self._get_gene_value(f"activation_{i}", "gelu")
            dropout_rate = self._get_gene_value(f"dropout_rate_{i}", 0.1)
            normalization = self._get_gene_value(f"normalization_{i}", "layer_norm")
            skip_connection = self._get_gene_value(f"skip_connection_{i}", False)
            
            # Create layer based on type
            if layer_type == "dense":
                layer = nn.Dense(hidden_size)
            elif layer_type == "conv1d":
                kernel_size = self._get_gene_value(f"kernel_size_{i}", 3)
                layer = nn.Conv(hidden_size, kernel_size=[kernel_size])
            elif layer_type == "attention":
                num_heads = self._get_gene_value(f"attention_heads_{i}", 4)
                layer = nn.MultiHeadDotProductAttention(
                    num_heads=num_heads,
                    qkv_features=hidden_size
                )
            elif layer_type == "rational":
                layer = RationalActivationLayer(hidden_size)
            elif layer_type == "glu":
                layer = GLULayer(hidden_size)
            else:
                layer = nn.Dense(hidden_size)  # Default fallback
            
            layer_info = {
                'layer': layer,
                'activation': activation,
                'dropout_rate': dropout_rate,
                'normalization': normalization,
                'skip_connection': skip_connection,
                'hidden_size': hidden_size
            }
            
            self.layers.append(layer_info)
            current_dim = hidden_size
        
        # Output layer
        self.output_layer = nn.Dense(self.output_dim)
    
    def _get_num_layers(self) -> int:
        """Get number of layers from genome."""
        return self._get_gene_value("num_layers", 3)
    
    def _get_gene_value(self, gene_key: str, default_value: Any) -> Any:
        """Get gene value with fallback to default."""
        if gene_key in self.genome.genes:
            return self.genome.genes[gene_key].value
        return default_value
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through evolved architecture."""
        current = x
        
        for i, layer_info in enumerate(self.layers):
            layer = layer_info['layer']
            
            # Apply layer
            if isinstance(layer, nn.MultiHeadDotProductAttention):
                # Attention layer needs special handling
                output = layer(current, current)
            else:
                output = layer(current)
            
            # Apply normalization
            if layer_info['normalization'] == "layer_norm":
                output = nn.LayerNorm()(output)
            elif layer_info['normalization'] == "batch_norm":
                output = nn.BatchNorm(use_running_average=False)(output)
            
            # Apply activation
            activation = layer_info['activation']
            if activation == "relu":
                output = nn.relu(output)
            elif activation == "gelu":
                output = nn.gelu(output)
            elif activation == "swish":
                output = nn.swish(output)
            elif activation == "tanh":
                output = nn.tanh(output)
            elif activation == "leaky_relu":
                output = nn.leaky_relu(output)
            
            # Apply dropout
            if layer_info['dropout_rate'] > 0:
                output = nn.Dropout(layer_info['dropout_rate'])(output, deterministic=False)
            
            # Skip connection
            if layer_info['skip_connection'] and current.shape == output.shape:
                output = current + output
            
            current = output
        
        # Final output layer
        return self.output_layer(current)


class RationalActivationLayer(nn.Module):
    """Rational activation function that can be learned."""
    
    hidden_size: int
    
    def setup(self):
        self.dense = nn.Dense(self.hidden_size)
        # Learnable rational function parameters
        self.numerator_coeffs = self.param('numerator', 
            lambda rng, shape: jax.random.normal(rng, shape) * 0.1, (4,))
        self.denominator_coeffs = self.param('denominator',
            lambda rng, shape: jnp.array([1.0, 0.0, 1.0]), (3,))
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.dense(x)
        
        # Rational activation: P(x) / Q(x)
        numerator = (self.numerator_coeffs[0] + 
                    self.numerator_coeffs[1] * x + 
                    self.numerator_coeffs[2] * x**2 + 
                    self.numerator_coeffs[3] * x**3)
        
        denominator = (self.denominator_coeffs[0] + 
                      self.denominator_coeffs[1] * x + 
                      self.denominator_coeffs[2] * x**2)
        
        # Avoid division by zero
        denominator = jnp.where(jnp.abs(denominator) < 1e-8, 1e-8, denominator)
        
        return numerator / denominator


class GLULayer(nn.Module):
    """Gated Linear Unit layer."""
    
    hidden_size: int
    
    def setup(self):
        self.linear = nn.Dense(self.hidden_size)
        self.gate = nn.Dense(self.hidden_size)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x) * nn.sigmoid(self.gate(x))


class ExpertEvolutionEngine:
    """
    Main engine for evolving expert architectures using genetic algorithms.
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism_ratio: float = 0.1):
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        
        self.population: List[ExpertGenome] = []
        self.generation = 0
        self.evolution_history = []
        self.best_genome = None
        
        # Performance tracking
        self.fitness_history = []
        self.diversity_history = []
    
    def initialize_population(self, input_dim: int, output_dim: int) -> List[ExpertGenome]:
        """Initialize random population of expert genomes."""
        population = []
        
        for i in range(self.population_size):
            genome = self._create_random_genome()
            population.append(genome)
        
        self.population = population
        return population
    
    def _create_random_genome(self) -> ExpertGenome:
        """Create a random expert genome."""
        genes = {}
        
        # Number of layers
        num_layers = random.randint(2, 5)
        genes["num_layers"] = ExpertGene(ExpertGeneType.HIDDEN_SIZE, num_layers)
        
        # Genes for each layer
        for i in range(num_layers):
            # Layer type
            layer_type = random.choice(["dense", "conv1d", "attention", "rational", "glu"])
            genes[f"layer_type_{i}"] = ExpertGene(ExpertGeneType.LAYER_TYPE, layer_type)
            
            # Hidden size
            hidden_size = random.choice([64, 128, 256, 512])
            genes[f"hidden_size_{i}"] = ExpertGene(ExpertGeneType.HIDDEN_SIZE, hidden_size)
            
            # Activation
            activation = random.choice(["relu", "gelu", "swish", "tanh"])
            genes[f"activation_{i}"] = ExpertGene(ExpertGeneType.ACTIVATION, activation)
            
            # Dropout
            dropout_rate = random.uniform(0.0, 0.5)
            genes[f"dropout_rate_{i}"] = ExpertGene(ExpertGeneType.DROPOUT_RATE, dropout_rate)
            
            # Normalization
            normalization = random.choice(["layer_norm", "batch_norm", "none"])
            genes[f"normalization_{i}"] = ExpertGene(ExpertGeneType.NORMALIZATION, normalization)
            
            # Skip connection
            skip_connection = random.choice([True, False])
            genes[f"skip_connection_{i}"] = ExpertGene(ExpertGeneType.SKIP_CONNECTION, skip_connection)
            
            # Layer-specific genes
            if layer_type == "attention":
                num_heads = random.choice([2, 4, 8])
                genes[f"attention_heads_{i}"] = ExpertGene(ExpertGeneType.ATTENTION_HEADS, num_heads)
            elif layer_type == "conv1d":
                kernel_size = random.choice([3, 5, 7])
                genes[f"kernel_size_{i}"] = ExpertGene(ExpertGeneType.KERNEL_SIZE, kernel_size)
        
        return ExpertGenome(genes=genes, generation=self.generation)
    
    def evaluate_fitness(self, 
                        genome: ExpertGenome,
                        fitness_function: callable,
                        input_data: jnp.ndarray,
                        target_data: jnp.ndarray) -> float:
        """Evaluate fitness of a genome using provided fitness function."""
        try:
            # Create expert from genome
            expert = EvolvableExpert(
                genome=genome,
                input_dim=input_data.shape[-1],
                output_dim=target_data.shape[-1] if len(target_data.shape) > 1 else 1
            )
            
            # Initialize expert
            key = jax.random.key(0)
            params = expert.init(key, input_data[:1])
            
            # Evaluate fitness
            fitness = fitness_function(expert, params, input_data, target_data)
            
            # Add complexity penalty to encourage simpler architectures
            complexity_penalty = self._compute_complexity_penalty(genome)
            adjusted_fitness = fitness - complexity_penalty
            
            return float(adjusted_fitness)
            
        except Exception as e:
            # Return low fitness for invalid architectures
            print(f"Fitness evaluation failed for genome: {e}")
            return -1000.0
    
    def _compute_complexity_penalty(self, genome: ExpertGenome) -> float:
        """Compute complexity penalty to encourage simpler architectures."""
        penalty = 0.0
        
        # Penalty for too many layers
        num_layers = genome.genes.get("num_layers", ExpertGene(ExpertGeneType.HIDDEN_SIZE, 3)).value
        if num_layers > 4:
            penalty += (num_layers - 4) * 0.1
        
        # Penalty for very large hidden sizes
        for gene_key, gene in genome.genes.items():
            if "hidden_size" in gene_key and gene.value > 512:
                penalty += (gene.value - 512) / 1000.0
        
        return penalty
    
    def evolve_generation(self, 
                         fitness_function: callable,
                         input_data: jnp.ndarray,
                         target_data: jnp.ndarray) -> List[ExpertGenome]:
        """Evolve one generation of the population."""
        
        # Evaluate fitness for all individuals
        for genome in self.population:
            if genome.fitness == 0.0:  # Only evaluate if not already evaluated
                genome.fitness = self.evaluate_fitness(
                    genome, fitness_function, input_data, target_data
                )
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Track best genome
        if self.best_genome is None or self.population[0].fitness > self.best_genome.fitness:
            self.best_genome = self.population[0]
        
        # Record generation statistics
        fitnesses = [genome.fitness for genome in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitnesses),
            'average_fitness': sum(fitnesses) / len(fitnesses),
            'worst_fitness': min(fitnesses)
        })
        
        # Create next generation
        new_population = []
        
        # Elitism: Keep best individuals
        elite_count = int(self.population_size * self.elitism_ratio)
        new_population.extend(self.population[:elite_count])
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Select parents
            parent1 = GeneticOperators.tournament_selection(self.population)
            parent2 = GeneticOperators.tournament_selection(self.population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = GeneticOperators.crossover_genomes(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            for child in [child1, child2]:
                if len(new_population) < self.population_size:
                    mutated_child = self._mutate_genome(child)
                    new_population.append(mutated_child)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        return self.population
    
    def _mutate_genome(self, genome: ExpertGenome) -> ExpertGenome:
        """Mutate a genome."""
        mutated_genes = {}
        
        for gene_key, gene in genome.genes.items():
            mutated_gene = GeneticOperators.mutate_gene(gene, self.mutation_rate)
            mutated_genes[gene_key] = mutated_gene
        
        return ExpertGenome(
            genes=mutated_genes,
            generation=genome.generation,
            parent_ids=genome.parent_ids
        )
    
    def get_best_expert(self, input_dim: int, output_dim: int) -> EvolvableExpert:
        """Get the best evolved expert."""
        if self.best_genome is None:
            raise ValueError("No evolution has been performed yet")
        
        return EvolvableExpert(
            genome=self.best_genome,
            input_dim=input_dim,
            output_dim=output_dim
        )
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolution process."""
        return {
            'generations': self.generation,
            'population_size': self.population_size,
            'best_fitness': self.best_genome.fitness if self.best_genome else 0.0,
            'fitness_history': self.fitness_history,
            'best_genome_genes': len(self.best_genome.genes) if self.best_genome else 0,
            'evolution_parameters': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_ratio': self.elitism_ratio
            }
        }


# Example fitness functions
def accuracy_fitness_function(expert: EvolvableExpert, 
                            params: Dict[str, Any],
                            input_data: jnp.ndarray, 
                            target_data: jnp.ndarray) -> float:
    """Fitness function based on classification accuracy."""
    try:
        # Forward pass
        predictions = expert.apply(params, input_data)
        
        # Convert to class predictions
        if len(predictions.shape) > 1 and predictions.shape[-1] > 1:
            predicted_classes = jnp.argmax(predictions, axis=-1)
            target_classes = jnp.argmax(target_data, axis=-1) if len(target_data.shape) > 1 else target_data
        else:
            predicted_classes = (predictions > 0.5).astype(jnp.int32)
            target_classes = target_data
        
        # Compute accuracy
        accuracy = jnp.mean(predicted_classes == target_classes)
        return float(accuracy)
        
    except Exception as e:
        return 0.0


def loss_fitness_function(expert: EvolvableExpert,
                         params: Dict[str, Any], 
                         input_data: jnp.ndarray,
                         target_data: jnp.ndarray) -> float:
    """Fitness function based on negative loss (higher is better)."""
    try:
        # Forward pass
        predictions = expert.apply(params, input_data)
        
        # Compute loss (MSE for regression, cross-entropy for classification)
        if len(target_data.shape) > 1 and target_data.shape[-1] > 1:
            # Classification
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                predictions, jnp.argmax(target_data, axis=-1)
            ))
        else:
            # Regression
            loss = jnp.mean((predictions - target_data) ** 2)
        
        # Return negative loss as fitness (higher is better)
        return float(-loss)
        
    except Exception as e:
        return -1000.0


if __name__ == "__main__":
    # Example evolution run
    evolution_engine = ExpertEvolutionEngine(population_size=20)
    
    # Generate sample data
    key = jax.random.key(0)
    input_data = jax.random.normal(key, (100, 32))
    target_data = jax.random.randint(key, (100,), 0, 3)  # 3-class classification
    
    # Initialize population
    population = evolution_engine.initialize_population(input_dim=32, output_dim=3)
    
    # Evolve for several generations
    for generation in range(5):
        print(f"Evolving generation {generation + 1}...")
        population = evolution_engine.evolve_generation(
            accuracy_fitness_function, input_data, target_data
        )
        
        best_fitness = max(genome.fitness for genome in population)
        avg_fitness = sum(genome.fitness for genome in population) / len(population)
        print(f"  Best fitness: {best_fitness:.3f}, Average: {avg_fitness:.3f}")
    
    # Get best expert
    best_expert = evolution_engine.get_best_expert(input_dim=32, output_dim=3)
    summary = evolution_engine.get_evolution_summary()
    
    print("\n🧬 Evolutionary Expert System Demo:")
    print(f"   Evolved for {summary['generations']} generations")
    print(f"   Best fitness achieved: {summary['best_fitness']:.3f}")
    print(f"   Best genome complexity: {summary['best_genome_genes']} genes")
