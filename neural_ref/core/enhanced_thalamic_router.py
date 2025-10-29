# SPDX-License-Identifier: Apache-2.0
"""
Enhanced Thalamic Router with Gradient Broadcasting
Integrates the working AURA_GENESIS architecture with gradient broadcasting concepts
"""

import numpy as np
import time
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#from AURA_GENESIS.aura_phase2_optimizations import Phase2Config
# Import existing AURA components (simulated - you'd import from actual modules)
from . import Neuron, ActivityState, MaturationStage
from .thalamic_router import ThalamicConversationRouter
from . import MultiChannelSpikingAttention, prosody_channels_from_text

@dataclass
class ZoneCapabilities:
    """Defines capabilities and parameters for each neural zone"""
    specialization: str
    learning_rate_modifier: float = 1.0
    attention_sensitivity: float = 1.0
    temporal_integration_window: int = 5
    sparsity_target: float = 0.5
    threshold_adaptation_rate: float = 0.1
    inhibition_strength: float = 0.3


class ThalamicGradientBroadcaster:
    """
    Central thalamic gradient broadcasting system
    Routes learning signals based on attention, zone capabilities, and neural activity
    """

    def __init__(self, zone_definitions: Dict[str, ZoneCapabilities],
                 total_neurons: int, zone_map: Dict[str, Tuple[int, int]]):
        self.zone_definitions = zone_definitions
        self.total_neurons = total_neurons
        self.zone_map = zone_map  # zone_name -> (start_idx, end_idx)

        # Attention-based routing matrix
        self.attention_router = np.eye(total_neurons) * 0.1  # Base connectivity
        self._initialize_zone_connectivity()

        # Zone-specific gradient modulation
        self.zone_modulators = {
            zone: np.ones(end_idx - start_idx)
            for zone, (start_idx, end_idx) in zone_map.items()
        }

        # Adaptive parameters
        self.register_usage_stats()

        # Temporal integration buffers
        self.gradient_history = np.zeros((5, total_neurons))  # Last 5 timesteps
        self.activity_history = np.zeros((10, total_neurons))  # Last 10 timesteps

    def _initialize_zone_connectivity(self):
        """Initialize inter-zone connectivity based on capabilities"""
        for zone_a, (start_a, end_a) in self.zone_map.items():
            caps_a = self.zone_definitions[zone_a]

            for zone_b, (start_b, end_b) in self.zone_map.items():
                if zone_a == zone_b:
                    continue

                # Define connectivity strength based on zone types
                strength = self._compute_inter_zone_strength(zone_a, zone_b)

                # Create connectivity pattern
                self.attention_router[start_a:end_a, start_b:end_b] = strength

    def _compute_inter_zone_strength(self, zone_a: str, zone_b: str) -> float:
        """Compute connectivity strength between zones based on biological principles"""

        # Thalamic routing patterns (simplified)
        connectivity_rules = {
            ('thalamic_router', 'amygdala'): 0.8,  # Strong threat routing
            ('thalamic_router', 'hippocampus'): 0.7,  # Memory consolidation
            ('amygdala', 'hippocampus'): 0.6,  # Fear memory formation
            ('hippocampus', 'general_chat'): 0.4,  # Memory-informed responses
            ('amygdala', 'general_chat'): 0.3,  # Emotional coloring
        }

        # Check both directions
        strength = connectivity_rules.get((zone_a, zone_b), 0.1)
        if strength == 0.1:
            strength = connectivity_rules.get((zone_b, zone_a), 0.1)

        return strength

    def register_usage_stats(self):
        """Initialize usage tracking for zone balancing"""
        self.zone_usage = {zone: 0.0 for zone in self.zone_map.keys()}
        self.zone_performance = {zone: 0.5 for zone in self.zone_map.keys()}
        self.total_updates = 0

    def route_gradients(self, attention_signals: Dict[str, float],
                        neural_activities: np.ndarray,
                        local_gradients: np.ndarray,
                        routing_context: Dict[str, Any]) -> np.ndarray:
        """
        Route gradients through thalamic broadcasting system

        Args:
            attention_signals: Attention gains per zone from multi-channel attention
            neural_activities: Current neural activation patterns
            local_gradients: Locally computed gradients
            routing_context: Context from routing decision (confidence, etc.)

        Returns:
            Enhanced gradients with thalamic broadcasting
        """

        # Store activity history
        self.activity_history[1:] = self.activity_history[:-1]
        self.activity_history[0] = neural_activities

        # Compute zone-specific attention modulation
        zone_attention_gains = self._compute_zone_attention_gains(
            attention_signals, routing_context
        )

        # Apply temporal integration from gradient history
        temporal_gradients = self._apply_temporal_integration(local_gradients)

        # Route gradients based on zone capabilities and attention
        routed_gradients = self._route_by_zones(
            temporal_gradients, zone_attention_gains, neural_activities
        )

        # Apply inter-zone broadcasting
        broadcasted_gradients = self._apply_inter_zone_broadcasting(
            routed_gradients, neural_activities
        )

        # Update gradient history
        self.gradient_history[1:] = self.gradient_history[:-1]
        self.gradient_history[0] = broadcasted_gradients

        # Track usage for zone balancing
        self._update_usage_stats(zone_attention_gains)

        return broadcasted_gradients

    def _compute_zone_attention_gains(self, attention_signals: Dict[str, float],
                                      context: Dict[str, Any]) -> Dict[str, float]:
        """Compute attention gains for each zone based on signals and context"""

        zone_gains = {}
        routing_confidence = context.get('routing_confidence', 0.5)

        for zone_name, capabilities in self.zone_definitions.items():
            # Base attention from multi-channel system
            base_attention = attention_signals.get(zone_name, 1.0)

            # Modulate by zone sensitivity
            zone_gain = base_attention * capabilities.attention_sensitivity

            # Apply confidence-based modulation
            confidence_boost = 1.0 + (routing_confidence - 0.5) * 0.5
            zone_gain *= confidence_boost

            # Apply usage balancing (encourage underused zones)
            usage_balance = 1.0 + (0.5 - self.zone_usage[zone_name]) * 0.3
            zone_gain *= usage_balance

            zone_gains[zone_name] = float(np.clip(zone_gain, 0.1, 3.0))

        return zone_gains

    def _apply_temporal_integration(self, local_gradients: np.ndarray) -> np.ndarray:
        """Apply temporal integration to gradients based on zone characteristics"""

        integrated_gradients = local_gradients.copy()
        print(self.zone_map)

        for zone_name, idx in self.zone_map.items():

            capabilities = self.zone_definitions[zone_name]
            window = capabilities.temporal_integration_window

            if window > 1:
                # Weighted average of recent gradients
                zone_history = self.gradient_history[:window, idx[0]:idx[1]]
                weights = np.exp(-np.arange(window) * 0.5)  # Exponential decay
                weights = weights / np.sum(weights)

                temporal_component = np.sum(
                    zone_history * weights.reshape(-1, 1), axis=0
                )

                # Blend with local gradients
                blend_factor = 0.3  # 30% temporal, 70% local
                integrated_gradients[idx[0]:idx[1]] = (
                        (1 - blend_factor) * local_gradients[idx[0]:idx[1]] +
                        blend_factor * temporal_component
                )

        return integrated_gradients

    def _route_by_zones(self, gradients: np.ndarray,
                        zone_gains: Dict[str, float],
                        activities: np.ndarray) -> np.ndarray:
        """Route gradients within zones based on capabilities and attention"""

        routed_gradients = gradients.copy()

        for zone_name, (start_idx, end_idx) in self.zone_map.items():
            try:
                # Ensure indices are valid
                if start_idx >= len(gradients) or end_idx > len(gradients) or start_idx >= end_idx:
                    continue
                    
                zone_gradients = gradients[start_idx:end_idx]
                zone_activities = activities[start_idx:end_idx] if start_idx < len(activities) and end_idx <= len(activities) else np.ones_like(zone_gradients)
                zone_gain = zone_gains.get(zone_name, 1.0)
                
                if zone_name not in self.zone_definitions:
                    continue
                    
                capabilities = self.zone_definitions[zone_name]

                # Apply learning rate modulation
                lr_modifier = capabilities.learning_rate_modifier
                modulated_gradients = zone_gradients * lr_modifier * zone_gain

                # Apply activity-dependent gating with safe broadcasting
                if len(zone_activities) == len(modulated_gradients):
                    activity_gate = self._compute_activity_gate(
                        zone_activities, capabilities.sparsity_target
                    )
                    
                    # Ensure activity_gate has compatible shape
                    if activity_gate.shape == modulated_gradients.shape:
                        gated_gradients = modulated_gradients * activity_gate
                    else:
                        # Fallback if shapes don't match
                        gated_gradients = modulated_gradients
                else:
                    gated_gradients = modulated_gradients

                # Apply zone-specific modulation
                zone_modulator = self.zone_modulators.get(zone_name, 1.0)
                final_gradients = gated_gradients * zone_modulator

                routed_gradients[start_idx:end_idx] = final_gradients
                
            except (ValueError, IndexError, KeyError) as e:
                logger.warning(f"Zone routing error for {zone_name}: {e}")
                # Keep original gradients for this zone
                if start_idx < len(routed_gradients) and end_idx <= len(routed_gradients):
                    routed_gradients[start_idx:end_idx] = gradients[start_idx:end_idx]

        return routed_gradients

    def _compute_activity_gate(self, activities: np.ndarray,
                               sparsity_target: float) -> np.ndarray:
        """Compute activity-dependent gating to maintain target sparsity"""

        # Current sparsity (fraction of near-zero activities)
        current_sparsity = np.mean(np.abs(activities) < 0.1)

        # Adjustment factor to reach target sparsity
        sparsity_error = sparsity_target - current_sparsity

        if sparsity_error > 0:  # Too dense, need more sparsity
            # Suppress weakly active neurons
            gate = np.where(
                np.abs(activities) < np.percentile(np.abs(activities), 50),
                0.5,  # Suppress weak neurons
                1.2  # Boost strong neurons
            )
        else:  # Too sparse, need more activity
            # Boost all neurons
            gate = np.ones_like(activities) * 1.1

        return gate

    def _apply_inter_zone_broadcasting(self, gradients: np.ndarray,
                                       activities: np.ndarray) -> np.ndarray:
        """Apply inter-zone gradient broadcasting through attention router"""

        # Compute activity-weighted broadcasting
        activity_weights = np.abs(activities) / (np.sum(np.abs(activities)) + 1e-8)

        # Broadcast gradients through attention router
        broadcasted = np.matmul(self.attention_router.T, gradients * activity_weights)

        # Combine local and broadcasted gradients
        broadcast_strength = 0.2  # 20% broadcast, 80% local
        enhanced_gradients = (
                (1 - broadcast_strength) * gradients +
                broadcast_strength * broadcasted
        )

        return enhanced_gradients

    def _update_usage_stats(self, zone_gains: Dict[str, float]):
        """Update zone usage statistics for balancing"""

        self.total_updates += 1
        alpha = 0.95  # Smoothing factor

        for zone_name, gain in zone_gains.items():
            # Update usage moving average
            current_usage = gain / max(sum(zone_gains.values()), 1.0)
            self.zone_usage[zone_name] = (
                    alpha * self.zone_usage[zone_name] +
                    (1 - alpha) * current_usage
            )

    def get_broadcaster_stats(self) -> Dict[str, Any]:
        """Get broadcasting statistics for monitoring"""

        return {
            'zone_usage': dict(self.zone_usage),
            'zone_performance': dict(self.zone_performance),
            'total_updates': self.total_updates,
            'gradient_history_depth': self.gradient_history.shape[0],
            'activity_history_depth': self.activity_history.shape[0],
            'total_neurons': self.total_neurons,
            'num_zones': len(self.zone_map)
        }


class EnhancedThalamicRouter:
    """
    Enhanced version of ThalamicConversationRouter with gradient broadcasting
    Integrates with existing AURA_GENESIS architecture
    """

    def __init__(self, base_router_config:any, enable_gradient_broadcasting: bool = True):
        # Wrap existing router
        self.base_router = ThalamicConversationRouter()
        self.enable_broadcasting = enable_gradient_broadcasting

        if enable_gradient_broadcasting:
            # Define zone capabilities based on router's neuron groups
            self.zone_capabilities = self._define_zone_capabilities()

            # Create zone map from router's neuron groups
            self.zone_map = self._create_zone_map()

            # Initialize gradient broadcaster
            self.gradient_broadcaster = ThalamicGradientBroadcaster(
                zone_definitions=self.zone_capabilities,
                total_neurons=len(self.base_router.all_neurons),
                zone_map=self.zone_map
            )

        # Enhanced statistics
        self.enhanced_stats = {
            'gradient_broadcasts': 0,
            'zone_activations': {zone: 0 for zone in self.base_router.routing_neurons},
            'attention_modulations': 0
        }

    def _define_zone_capabilities(self) -> Dict[str, ZoneCapabilities]:
        """Define capabilities for each zone based on their specialization"""

        return {
            'general_chat': ZoneCapabilities(
                specialization='conversational',
                learning_rate_modifier=1.0,
                attention_sensitivity=0.8,
                temporal_integration_window=3,
                sparsity_target=0.4
            ),
            'historical_specialist': ZoneCapabilities(
                specialization='temporal_knowledge',
                learning_rate_modifier=0.8,
                attention_sensitivity=1.2,
                temporal_integration_window=7,
                sparsity_target=0.6
            ),
            'amygdala_specialist': ZoneCapabilities(
                specialization='emotional_processing',
                learning_rate_modifier=1.5,
                attention_sensitivity=1.8,
                temporal_integration_window=2,
                sparsity_target=0.3
            ),
            'hippocampus_specialist': ZoneCapabilities(
                specialization='memory_formation',
                learning_rate_modifier=0.6,
                attention_sensitivity=1.0,
                temporal_integration_window=10,
                sparsity_target=0.7
            ),
            'analytical_specialist': ZoneCapabilities(
                specialization='logical_processing',
                learning_rate_modifier=0.9,
                attention_sensitivity=0.9,
                temporal_integration_window=5,
                sparsity_target=0.5
            ),
            'multi_specialist': ZoneCapabilities(
                specialization='integration',
                learning_rate_modifier=1.1,
                attention_sensitivity=1.3,
                temporal_integration_window=4,
                sparsity_target=0.4
            )
        }

    def _create_zone_map(self) -> Dict[str, Tuple[int, int]]:
        """Create mapping from zone names to neuron indices"""

        zone_map = {}
        current_idx = 0

        for neurons in self.base_router.routing_neurons:
            start_idx = current_idx
            end_idx = current_idx + len(self.base_router.routing_neurons)
            zone_map[neurons.lower()] = (start_idx, end_idx)
            current_idx = end_idx

        return zone_map

    def _extract_neural_signals(self) -> np.ndarray:
        """Extract current neural activity signals from all neurons"""

        signals = []
        for neuron in self.base_router.all_neurons:
            # Get activity level from neuron's current state
            if hasattr(neuron, 'membrane_potential'):
                signals.append(neuron.membrane_potential)
            elif hasattr(neuron, 'activity'):
                activity_value = 1.0 if neuron.activity.name == 'FIRING' else 0.0
                signals.append(activity_value)
            else:
                signals.append(0.0)

        return np.array(signals)

    def _compute_zone_gradients(self, routing_outcome: Dict[str, Any]) -> np.ndarray:
        """Compute gradients for each zone based on routing outcome"""

        success_score = routing_outcome.get('user_satisfaction', 0.5)
        response_quality = routing_outcome.get('response_quality', 0.5)
        overall_success = 0.5 * (success_score + response_quality)

        # Convert to learning signal
        learning_target = 1.0 if overall_success > 0.7 else 0.0

        # Compute gradients (simplified as error signal)
        gradients = []
        for neuron in self.base_router.all_neurons:
            # Simulate gradient based on neuron's current prediction vs target
            current_pred = getattr(neuron, 'last_prediction', 0.5)
            gradient = learning_target - current_pred
            gradients.append(gradient)

        return np.array(gradients)

    async def enhanced_routing_update(self, routing_plan: Dict[str, Any],
                                      conversation_outcome: Dict[str, Any],
                                      query_features: np.ndarray,
                                      query_text: str = "") -> Dict[str, Any]:
        """Enhanced routing update with gradient broadcasting"""

        if not self.enable_broadcasting:
            # Fall back to base router behavior
            return await self.base_router.adaptive_routing_update_with_attention(
                routing_plan, conversation_outcome, query_features, query_text
            )

        # Extract current neural activities
        neural_activities = self._extract_neural_signals()

        # Compute local gradients from routing outcome
        local_gradients = self._compute_zone_gradients(conversation_outcome)

        # Get attention signals from base router's attention system
        attention_signals = {}
        if self.base_router.attn and query_text:
            attention_telemetry = self.base_router.get_attention_telemetry(query_text)

            # Map attention to zones (simplified)
            base_gain = attention_telemetry.get('mu_scalar', 1.0)
            for zone_name in self.zone_map.keys():
                attention_signals[zone_name] = base_gain
        else:
            # Default uniform attention
            for zone_name in self.zone_map.keys():
                attention_signals[zone_name] = 1.0

        # Apply thalamic gradient broadcasting
        enhanced_gradients = self.gradient_broadcaster.route_gradients(
            attention_signals=attention_signals,
            neural_activities=neural_activities,
            local_gradients=local_gradients,
            routing_context={
                'routing_confidence': routing_plan.get('confidence', 0.5),
                'primary_target': routing_plan.get('primary_specialist'),
                'query_characteristics': routing_plan.get('query_characteristics', {})
            }
        )

        # Apply enhanced gradients to neurons
        await self._apply_enhanced_gradients(enhanced_gradients, query_features)

        # Update statistics
        self.enhanced_stats['gradient_broadcasts'] += 1
        primary = routing_plan.get('primary_specialist')
        if primary:
            self.enhanced_stats['zone_activations'][primary] += 1

        # Return enhanced results
        base_result = await self.base_router.adaptive_routing_update_with_attention(
            routing_plan, conversation_outcome, query_features, query_text
        )

        base_result.update({
            'gradient_broadcasting_applied': True,
            'enhanced_gradients_norm': float(np.linalg.norm(enhanced_gradients)),
            'broadcaster_stats': self.gradient_broadcaster.get_broadcaster_stats()
        })

        return base_result

    async def _apply_enhanced_gradients(self, enhanced_gradients: np.ndarray,
                                        query_features: np.ndarray):
        """Apply enhanced gradients to neurons"""

        for i, neuron in enumerate(self.base_router.all_neurons):
            if i < len(enhanced_gradients):
                # Convert gradient to learning target (simplified)
                gradient = enhanced_gradients[i]
                learning_target = 0.5 + gradient * 0.5  # Map to [0, 1]
                learning_target = np.clip(learning_target, 0.0, 1.0)

                # Apply learning update
                await neuron.update_nlms(query_features, learning_target)

    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get enhanced statistics including broadcasting metrics"""

        base_stats = self.base_router.get_routing_statistics()
        broadcaster_stats = {}

        if self.enable_broadcasting:
            broadcaster_stats = self.gradient_broadcaster.get_broadcaster_stats()

        return {
            'base_routing_stats': base_stats,
            'enhanced_stats': self.enhanced_stats,
            'gradient_broadcasting': broadcaster_stats,
            'zone_capabilities': {
                zone: {
                    'specialization': caps.specialization,
                    'learning_rate_modifier': caps.learning_rate_modifier,
                    'attention_sensitivity': caps.attention_sensitivity
                }
                for zone, caps in self.zone_capabilities.items()
            } if self.enable_broadcasting else {}
        }

    # Delegate other methods to base router
    def __getattr__(self, name):
        """Delegate undefined methods to base router"""
        return getattr(self.base_router, name)


# Usage example and integration function
def create_enhanced_thalamic_router(base_router,
                                    enable_gradient_broadcasting: bool = True) -> EnhancedThalamicRouter:
    """Create enhanced thalamic router from existing base router"""
    return EnhancedThalamicRouter(base_router, enable_gradient_broadcasting)


# Test function to validate integration
async def test_enhanced_integration():
    """Test the enhanced thalamic router integration"""

    print("Enhanced Thalamic Router Integration Test")
    print("=" * 50)

    # This would normally be your existing ThalamicConversationRouter
    # For testing, we'll create a mock
    class MockBaseRouter:
        def __init__(self):
            self.routing_neurons = {
                'general_chat': [MockNeuron(f'general_{i}') for i in range(10)],
                'historical_specialist': [MockNeuron(f'hist_{i}') for i in range(8)],
                'amygdala_specialist': [MockNeuron(f'amyg_{i}') for i in range(6)]
            }
            self.all_neurons = [n for group in self.routing_neurons.values() for n in group]
            self.attn = None

        async def adaptive_routing_update_with_attention(self, *args):
            return {'base_update': True}

        def get_routing_statistics(self):
            return {'total_conversations': 50}

        def get_attention_telemetry(self, text):
            return {'mu_scalar': 1.2}

    class MockNeuron:
        def __init__(self, neuron_id):
            self.neuron_id = neuron_id
            self.membrane_potential = np.random.uniform(-0.5, 0.5)
            self.last_prediction = np.random.uniform(0, 1)

        async def update_nlms(self, features, target):
            self.last_prediction = 0.9 * self.last_prediction + 0.1 * target
            return self.last_prediction



    # Test enhanced routing update
    routing_plan = {
        'primary_specialist': 'general_chat',
        'confidence': 0.8,
        'query_characteristics': {'is_historical': False}
    }

    conversation_outcome = {
        'user_satisfaction': 0.9,
        'response_quality': 0.8
    }

    query_features = np.random.randn(384)
    query_text = "Hello, how are you doing today?"

    # Create enhanced router
    base_router = MockBaseRouter()
    enhanced_router = create_enhanced_thalamic_router(base_router, True)

    print(f"Gradient broadcasting enabled: {enhanced_router.enable_broadcasting}")

    result = await enhanced_router.enhanced_routing_update(
        routing_plan, conversation_outcome, query_features, query_text
    )

    print(f"Gradient broadcasting applied: {result.get('gradient_broadcasting_applied', False)}")
    print(f"Enhanced gradients norm: {result.get('enhanced_gradients_norm', 0):.4f}")

    # Get enhanced statistics
    stats = enhanced_router.get_enhanced_statistics()
    print(f"Total gradient broadcasts: {stats['enhanced_stats']['gradient_broadcasts']}")
    print(f"Zone activations: {stats['enhanced_stats']['zone_activations']}")

    if stats['gradient_broadcasting']:
        print(f"Broadcaster stats: {stats['gradient_broadcasting']}")

    print("\nIntegration test completed successfully!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_enhanced_integration())