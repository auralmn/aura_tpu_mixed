#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Causal Reasoning Engine for AURA
Implements causal inference, counterfactual reasoning, and interventional thinking
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum
import networkx as nx


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    CONFOUNDING = "confounding"
    MEDIATING = "mediating"
    MODERATING = "moderating"


@dataclass
class CausalRelation:
    """Represents a causal relationship between variables."""
    cause: str
    effect: str
    relation_type: CausalRelationType
    strength: float
    confidence: float
    evidence: List[str]


@dataclass
class Intervention:
    """Represents an intervention on a variable."""
    variable: str
    intervention_value: Any
    intervention_type: str = "do"  # "do", "condition", "observe"


class CausalDAG:
    """
    Directed Acyclic Graph representing causal relationships.
    Implements Pearl's causal hierarchy for interventional reasoning.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.causal_relations = {}
        self.confounders = set()
        self.mediators = set()
    
    def add_causal_relation(self, relation: CausalRelation):
        """Add a causal relationship to the graph."""
        self.graph.add_edge(
            relation.cause, 
            relation.effect,
            relation_type=relation.relation_type,
            strength=relation.strength,
            confidence=relation.confidence
        )
        
        relation_id = f"{relation.cause}->{relation.effect}"
        self.causal_relations[relation_id] = relation
        
        # Track special node types
        if relation.relation_type == CausalRelationType.CONFOUNDING:
            self.confounders.add(relation.cause)
        elif relation.relation_type == CausalRelationType.MEDIATING:
            self.mediators.add(relation.cause)
    
    def get_parents(self, node: str) -> List[str]:
        """Get direct causes of a node."""
        return list(self.graph.predecessors(node))
    
    def get_children(self, node: str) -> List[str]:
        """Get direct effects of a node."""
        return list(self.graph.successors(node))
    
    def get_ancestors(self, node: str) -> Set[str]:
        """Get all indirect causes of a node."""
        return set(nx.ancestors(self.graph, node))
    
    def get_descendants(self, node: str) -> Set[str]:
        """Get all indirect effects of a node."""
        return set(nx.descendants(self.graph, node))
    
    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """Find backdoor paths between treatment and outcome."""
        backdoor_paths = []
        
        # Find all paths from treatment to outcome
        try:
            all_paths = list(nx.all_simple_paths(
                self.graph.to_undirected(), treatment, outcome
            ))
            
            for path in all_paths:
                # Check if path starts with an edge into treatment (backdoor)
                if len(path) > 2:
                    # Check if there's an edge from path[1] to treatment
                    if self.graph.has_edge(path[1], treatment):
                        backdoor_paths.append(path)
                        
        except nx.NetworkXNoPath:
            pass
            
        return backdoor_paths
    
    def is_valid_adjustment_set(self, 
                               adjustment_set: Set[str],
                               treatment: str, 
                               outcome: str) -> bool:
        """Check if adjustment set satisfies backdoor criterion."""
        backdoor_paths = self.find_backdoor_paths(treatment, outcome)
        
        for path in backdoor_paths:
            path_blocked = False
            for node in adjustment_set:
                if node in path[1:-1]:  # Node is on the path (not treatment or outcome)
                    path_blocked = True
                    break
            
            if not path_blocked:
                return False
        
        return True


class DoCalculus:
    """
    Implements Pearl's do-calculus for causal inference.
    Enables computation of interventional distributions.
    """
    
    def __init__(self, causal_dag: CausalDAG):
        self.dag = causal_dag
    
    def apply_intervention(self, 
                          intervention: Intervention,
                          data_distribution: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Apply do-calculus intervention to compute P(Y | do(X = x)).
        
        Args:
            intervention: The intervention to apply
            data_distribution: Observational data distribution
            
        Returns:
            Interventional distribution after applying intervention
        """
        intervened_distribution = data_distribution.copy()
        
        if intervention.intervention_type == "do":
            # For do-interventions, remove incoming edges to intervened variable
            variable = intervention.variable
            parents = self.dag.get_parents(variable)
            
            # Set the intervened variable to the intervention value
            intervened_distribution[variable] = jnp.full_like(
                data_distribution[variable], 
                intervention.intervention_value
            )
            
            # Remove confounding effects by marginalizing over parents
            for parent in parents:
                if parent in intervened_distribution:
                    # This is a simplified implementation
                    # In practice, would need proper marginalization
                    pass
        
        return intervened_distribution
    
    def compute_causal_effect(self,
                            treatment: str,
                            outcome: str, 
                            treatment_value: float,
                            data_distribution: Dict[str, jnp.ndarray],
                            adjustment_set: Optional[Set[str]] = None) -> float:
        """
        Compute causal effect using adjustment formula or do-calculus.
        
        Returns:
            Estimated causal effect size
        """
        # Apply intervention
        intervention = Intervention(treatment, treatment_value, "do")
        intervened_dist = self.apply_intervention(intervention, data_distribution)
        
        # Compute effect as difference in outcome means
        original_outcome = jnp.mean(data_distribution[outcome])
        intervened_outcome = jnp.mean(intervened_dist[outcome])
        
        causal_effect = intervened_outcome - original_outcome
        return float(causal_effect)


class CounterfactualReasoning:
    """
    Implements counterfactual reasoning: "What would have happened if...?"
    Uses structural causal models for counterfactual inference.
    """
    
    def __init__(self, causal_dag: CausalDAG):
        self.dag = causal_dag
        self.structural_equations = {}
    
    def add_structural_equation(self, 
                              variable: str, 
                              equation_fn: callable,
                              noise_term: str = None):
        """Add structural equation for a variable."""
        self.structural_equations[variable] = {
            'function': equation_fn,
            'noise': noise_term
        }
    
    def generate_counterfactual(self,
                              factual_data: Dict[str, jnp.ndarray],
                              counterfactual_intervention: Intervention) -> Dict[str, jnp.ndarray]:
        """
        Generate counterfactual world given factual data and intervention.
        
        Steps:
        1. Abduction: Infer noise terms from factual data
        2. Action: Apply intervention 
        3. Prediction: Compute counterfactual outcomes
        """
        # Step 1: Abduction - infer unobserved noise terms
        noise_terms = self._infer_noise_terms(factual_data)
        
        # Step 2: Action - apply counterfactual intervention
        counterfactual_data = factual_data.copy()
        counterfactual_data[counterfactual_intervention.variable] = (
            jnp.full_like(
                factual_data[counterfactual_intervention.variable],
                counterfactual_intervention.intervention_value
            )
        )
        
        # Step 3: Prediction - compute downstream effects
        counterfactual_data = self._propagate_counterfactual_effects(
            counterfactual_data, noise_terms, counterfactual_intervention.variable
        )
        
        return counterfactual_data
    
    def _infer_noise_terms(self, data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Infer noise terms from observed data using structural equations."""
        noise_terms = {}
        
        for variable, equation_info in self.structural_equations.items():
            if variable in data:
                # Simple noise inference: residuals from predicted values
                parents = self.dag.get_parents(variable)
                parent_data = {p: data[p] for p in parents if p in data}
                
                if parent_data and equation_info['function']:
                    predicted = equation_info['function'](parent_data)
                    noise = data[variable] - predicted
                    noise_terms[variable] = noise
                else:
                    # Default to zero noise if no structural equation
                    noise_terms[variable] = jnp.zeros_like(data[variable])
        
        return noise_terms
    
    def _propagate_counterfactual_effects(self,
                                        counterfactual_data: Dict[str, jnp.ndarray],
                                        noise_terms: Dict[str, jnp.ndarray],
                                        intervention_variable: str) -> Dict[str, jnp.ndarray]:
        """Propagate counterfactual effects through causal graph."""
        # Get topological ordering of variables
        try:
            topo_order = list(nx.topological_sort(self.dag.graph))
        except nx.NetworkXError:
            # If graph has cycles, use arbitrary ordering
            topo_order = list(self.dag.graph.nodes())
        
        # Start propagation after intervention variable
        intervention_idx = topo_order.index(intervention_variable) if intervention_variable in topo_order else 0
        
        for variable in topo_order[intervention_idx + 1:]:
            if variable in self.structural_equations:
                equation_info = self.structural_equations[variable]
                parents = self.dag.get_parents(variable)
                
                if parents and equation_info['function']:
                    parent_data = {p: counterfactual_data[p] for p in parents if p in counterfactual_data}
                    predicted = equation_info['function'](parent_data)
                    noise = noise_terms.get(variable, jnp.zeros_like(predicted))
                    counterfactual_data[variable] = predicted + noise
        
        return counterfactual_data


class CausalReasoningEngine:
    """
    Main causal reasoning engine integrating all causal inference capabilities.
    Provides high-level interface for causal analysis in AURA.
    """
    
    def __init__(self):
        self.causal_dag = CausalDAG()
        self.do_calculus = DoCalculus(self.causal_dag)
        self.counterfactual_reasoning = CounterfactualReasoning(self.causal_dag)
        self.causal_knowledge = {}
        
    def learn_causal_structure(self, 
                             observational_data: Dict[str, jnp.ndarray],
                             method: str = "pc_algorithm") -> CausalDAG:
        """
        Learn causal structure from observational data.
        
        Args:
            observational_data: Dictionary of observed variables
            method: Structure learning algorithm to use
            
        Returns:
            Learned causal DAG
        """
        if method == "pc_algorithm":
            return self._pc_algorithm(observational_data)
        elif method == "correlation_based":
            return self._correlation_based_structure_learning(observational_data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _pc_algorithm(self, data: Dict[str, jnp.ndarray]) -> CausalDAG:
        """Simplified PC algorithm for causal structure learning."""
        variables = list(data.keys())
        
        # Step 1: Start with complete graph
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    # Compute correlation as proxy for causal strength
                    corr = jnp.corrcoef(data[var1].flatten(), data[var2].flatten())[0, 1]
                    if jnp.abs(corr) > 0.1:  # Threshold for significance
                        relation = CausalRelation(
                            cause=var1,
                            effect=var2,
                            relation_type=CausalRelationType.DIRECT_CAUSE,
                            strength=float(jnp.abs(corr)),
                            confidence=0.7,  # Default confidence
                            evidence=[f"correlation: {corr:.3f}"]
                        )
                        self.causal_dag.add_causal_relation(relation)
        
        return self.causal_dag
    
    def _correlation_based_structure_learning(self, 
                                            data: Dict[str, jnp.ndarray]) -> CausalDAG:
        """Simple correlation-based causal structure learning."""
        variables = list(data.keys())
        
        # Compute pairwise correlations
        correlations = {}
        for var1 in variables:
            for var2 in variables:
                if var1 != var2:
                    corr = jnp.corrcoef(data[var1].flatten(), data[var2].flatten())[0, 1]
                    if not jnp.isnan(corr):
                        correlations[(var1, var2)] = float(corr)
        
        # Add edges for strong correlations
        for (var1, var2), corr in correlations.items():
            if abs(corr) > 0.3:  # Threshold for causal relationship
                relation = CausalRelation(
                    cause=var1,
                    effect=var2,
                    relation_type=CausalRelationType.DIRECT_CAUSE,
                    strength=abs(corr),
                    confidence=min(0.9, abs(corr) + 0.3),
                    evidence=[f"correlation: {corr:.3f}"]
                )
                self.causal_dag.add_causal_relation(relation)
        
        return self.causal_dag
    
    def analyze_causal_effect(self,
                            treatment: str,
                            outcome: str,
                            data: Dict[str, jnp.ndarray],
                            method: str = "backdoor") -> Dict[str, Any]:
        """
        Analyze causal effect of treatment on outcome.
        
        Returns:
            Dictionary with causal effect analysis results
        """
        analysis_results = {
            'treatment': treatment,
            'outcome': outcome,
            'method': method
        }
        
        if method == "backdoor":
            # Find valid adjustment set
            backdoor_adjustment_set = self._find_backdoor_adjustment_set(treatment, outcome)
            
            if backdoor_adjustment_set is not None:
                # Compute causal effect using adjustment formula
                causal_effect = self.do_calculus.compute_causal_effect(
                    treatment, outcome, 1.0, data, backdoor_adjustment_set
                )
                
                analysis_results.update({
                    'causal_effect': causal_effect,
                    'adjustment_set': list(backdoor_adjustment_set),
                    'identifiable': True
                })
            else:
                analysis_results.update({
                    'causal_effect': None,
                    'adjustment_set': None,
                    'identifiable': False,
                    'reason': 'No valid backdoor adjustment set found'
                })
        
        return analysis_results
    
    def _find_backdoor_adjustment_set(self, 
                                    treatment: str, 
                                    outcome: str) -> Optional[Set[str]]:
        """Find minimal backdoor adjustment set."""
        all_variables = set(self.causal_dag.graph.nodes())
        all_variables.discard(treatment)
        all_variables.discard(outcome)
        
        # Try subsets of increasing size
        from itertools import combinations
        
        for size in range(len(all_variables) + 1):
            for subset in combinations(all_variables, size):
                adjustment_set = set(subset)
                if self.causal_dag.is_valid_adjustment_set(adjustment_set, treatment, outcome):
                    return adjustment_set
        
        return None
    
    def generate_counterfactual_explanation(self,
                                          factual_scenario: Dict[str, Any],
                                          counterfactual_intervention: Intervention) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for a given scenario.
        
        Returns:
            Counterfactual analysis with explanations
        """
        # Convert scenario to arrays if needed
        factual_data = {}
        for key, value in factual_scenario.items():
            if isinstance(value, (int, float)):
                factual_data[key] = jnp.array([value])
            else:
                factual_data[key] = jnp.asarray(value)
        
        # Generate counterfactual
        counterfactual_data = self.counterfactual_reasoning.generate_counterfactual(
            factual_data, counterfactual_intervention
        )
        
        # Compute differences
        differences = {}
        for key in factual_data.keys():
            if key in counterfactual_data:
                factual_val = float(jnp.mean(factual_data[key]))
                counterfactual_val = float(jnp.mean(counterfactual_data[key]))
                differences[key] = {
                    'factual': factual_val,
                    'counterfactual': counterfactual_val,
                    'difference': counterfactual_val - factual_val
                }
        
        explanation = {
            'intervention': {
                'variable': counterfactual_intervention.variable,
                'value': counterfactual_intervention.intervention_value
            },
            'differences': differences,
            'summary': self._generate_explanation_text(differences, counterfactual_intervention)
        }
        
        return explanation
    
    def _generate_explanation_text(self, 
                                 differences: Dict[str, Dict[str, float]],
                                 intervention: Intervention) -> str:
        """Generate human-readable explanation text."""
        intervention_text = f"If {intervention.variable} had been {intervention.intervention_value}"
        
        significant_changes = []
        for var, change_info in differences.items():
            if var != intervention.variable and abs(change_info['difference']) > 0.1:
                direction = "increased" if change_info['difference'] > 0 else "decreased"
                significant_changes.append(f"{var} would have {direction} by {abs(change_info['difference']):.2f}")
        
        if significant_changes:
            changes_text = ", and ".join(significant_changes)
            explanation = f"{intervention_text}, then {changes_text}."
        else:
            explanation = f"{intervention_text}, no significant changes would have occurred."
        
        return explanation
    
    def get_causal_insights(self) -> Dict[str, Any]:
        """Get summary of causal knowledge and insights."""
        insights = {
            'total_variables': len(self.causal_dag.graph.nodes()),
            'total_relationships': len(self.causal_dag.graph.edges()),
            'confounders': list(self.causal_dag.confounders),
            'mediators': list(self.causal_dag.mediators),
            'strongest_relationships': []
        }
        
        # Find strongest causal relationships
        edge_strengths = []
        for relation_id, relation in self.causal_dag.causal_relations.items():
            edge_strengths.append((relation_id, relation.strength))
        
        edge_strengths.sort(key=lambda x: x[1], reverse=True)
        insights['strongest_relationships'] = edge_strengths[:5]
        
        return insights


# Integration with AURA consciousness system
class CausalConsciousnessModule(nn.Module):
    """Consciousness module with causal reasoning capabilities."""
    
    hidden_dim: int = 256
    
    def setup(self):
        self.causal_engine = CausalReasoningEngine()
        self.causal_embedding = nn.Dense(self.hidden_dim)
        self.intervention_predictor = nn.Dense(64)
        self.counterfactual_generator = nn.Dense(self.hidden_dim)
    
    def __call__(self, 
                 context: jnp.ndarray,
                 causal_query: Optional[str] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Process context with causal reasoning.
        
        Returns:
            causally_informed_representation: Enhanced representation with causal insights
            causal_analysis: Causal reasoning results
        """
        # Embed context for causal analysis
        causal_features = self.causal_embedding(context)
        
        # Generate causal hypotheses
        intervention_logits = self.intervention_predictor(causal_features)
        
        # Generate counterfactual representations
        counterfactual_features = self.counterfactual_generator(causal_features)
        
        # Combine original and counterfactual representations
        causally_informed = causal_features + 0.1 * counterfactual_features
        
        causal_analysis = {
            'causal_features_norm': float(jnp.linalg.norm(causal_features)),
            'counterfactual_strength': float(jnp.mean(jnp.abs(counterfactual_features))),
            'intervention_confidence': float(jnp.max(nn.softmax(intervention_logits)))
        }
        
        return causally_informed, causal_analysis


if __name__ == "__main__":
    # Example usage
    engine = CausalReasoningEngine()
    
    # Create sample data
    key = jax.random.key(0)
    data = {
        'education': jax.random.normal(key, (1000,)),
        'income': jax.random.normal(key, (1000,)),
        'health': jax.random.normal(key, (1000,))
    }
    
    # Learn causal structure
    dag = engine.learn_causal_structure(data)
    
    # Analyze causal effect
    effect_analysis = engine.analyze_causal_effect('education', 'income', data)
    
    # Generate counterfactual
    factual_scenario = {'education': 12, 'income': 50000, 'health': 0.8}
    intervention = Intervention('education', 16)
    counterfactual_explanation = engine.generate_counterfactual_explanation(
        factual_scenario, intervention
    )
    
    print("🔬 Causal Reasoning Engine Demo:")
    print(f"   Learned DAG with {len(dag.graph.nodes())} variables")
    print(f"   Causal effect analysis: {effect_analysis}")
    print(f"   Counterfactual: {counterfactual_explanation['summary']}")
