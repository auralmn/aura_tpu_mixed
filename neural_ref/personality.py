#!/usr/bin/env python3
"""
Personality and Emotion module

Implements dynamic Big Five personality traits, emotional state, and
neural networks for appraisal, trait-emotion mapping, regulation,
and long-term adaptation. Designed to integrate with AURA systems.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import time


@dataclass
class PersonalityTraits:
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    plasticity: float = 0.1
    stability_preference: float = 0.7


@dataclass
class EmotionalState:
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    intensity: float = 0.0
    joy: float = 0.0
    sadness: float = 0.0
    anger: float = 0.0
    fear: float = 0.0
    disgust: float = 0.0
    surprise: float = 0.0
    trust: float = 0.0
    anticipation: float = 0.0


class TraitEmotionNetwork(nn.Module):
    def __init__(self, trait_dim: int = 5, emotion_dim: int = 8):
        super().__init__()
        self.trait_dim = trait_dim
        self.emotion_dim = emotion_dim
        self.openness_emotions = nn.Linear(trait_dim, emotion_dim)
        self.conscientiousness_emotions = nn.Linear(trait_dim, emotion_dim)
        self.extraversion_emotions = nn.Linear(trait_dim, emotion_dim)
        self.agreeableness_emotions = nn.Linear(trait_dim, emotion_dim)
        self.neuroticism_emotions = nn.Linear(trait_dim, emotion_dim)
        self.trait_interaction = nn.Sequential(
            nn.Linear(trait_dim, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, emotion_dim)
        )
        self.emotion_fusion = nn.Linear(emotion_dim * 6, emotion_dim)

    def forward(self, appraisal_scores: torch.Tensor, traits: torch.Tensor) -> torch.Tensor:
        openness_response = self.openness_emotions(traits) * traits[0]
        conscientiousness_response = self.conscientiousness_emotions(traits) * traits[1]
        extraversion_response = self.extraversion_emotions(traits) * traits[2]
        agreeableness_response = self.agreeableness_emotions(traits) * traits[3]
        neuroticism_response = self.neuroticism_emotions(traits) * traits[4]
        interaction_response = self.trait_interaction(traits)
        combined = torch.cat([
            openness_response,
            conscientiousness_response,
            extraversion_response,
            agreeableness_response,
            neuroticism_response,
            interaction_response
        ], dim=-1)
        emotions = torch.sigmoid(self.emotion_fusion(combined))
        emotions = emotions * appraisal_scores.mean(dim=-1, keepdim=True)
        return emotions


class AppraisalNetwork(nn.Module):
    def __init__(self, input_dim: int = 512, trait_dim: int = 5, appraisal_dim: int = 16):
        super().__init__()
        self.trait_appraisal_modulation = nn.Sequential(
            nn.Linear(trait_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, appraisal_dim)
        )
        self.stimulus_processor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, appraisal_dim)
        )

    def forward(self, stimulus: torch.Tensor, traits: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
        base_appraisal = self.stimulus_processor(stimulus)
        trait_modulation = self.trait_appraisal_modulation(traits)
        if traits[0] > 0.6:
            trait_modulation[0] *= 1.5
        if traits[1] > 0.6:
            trait_modulation[5] *= 1.3
            trait_modulation[11] *= 1.2
        if traits[2] > 0.6:
            trait_modulation[8] *= 1.4
            trait_modulation[15] *= 1.3
        if traits[3] > 0.6:
            trait_modulation[4] *= 1.3
            trait_modulation[15] *= 1.2
        if traits[4] > 0.6:
            trait_modulation[9] *= 1.5
            trait_modulation[10] *= 1.4
            trait_modulation[13] *= -0.3
        final_appraisal = torch.sigmoid(base_appraisal + trait_modulation)
        return final_appraisal


class EmotionRegulationNetwork(nn.Module):
    def __init__(self, emotion_dim: int = 8, trait_dim: int = 5):
        super().__init__()
        self.regulation_strategies = nn.ModuleDict({
            'suppression': nn.Linear(trait_dim + emotion_dim, emotion_dim),
            'reappraisal': nn.Linear(trait_dim + emotion_dim, emotion_dim),
            'expression': nn.Linear(trait_dim + emotion_dim, emotion_dim),
            'rumination': nn.Linear(trait_dim + emotion_dim, emotion_dim),
            'distraction': nn.Linear(trait_dim + emotion_dim, emotion_dim)
        })
        self.strategy_selector = nn.Sequential(
            nn.Linear(trait_dim + emotion_dim, 32), nn.ReLU(),
            nn.Linear(32, 5), nn.Softmax(dim=-1)
        )

    def forward(self, raw_emotions: torch.Tensor, traits: torch.Tensor, current_state: torch.Tensor) -> torch.Tensor:
        combined_input = torch.cat([traits, raw_emotions], dim=-1)
        strategy_weights = self.strategy_selector(combined_input)
        regulated_emotions = torch.zeros_like(raw_emotions)
        for i, strategy in enumerate(self.regulation_strategies.values()):
            strategy_input = torch.cat([traits, current_state], dim=-1)
            strategy_output = strategy(strategy_input)
            regulated_emotions += strategy_weights[i] * strategy_output
        if traits[1] > 0.6:
            regulated_emotions = regulated_emotions * 0.8
        if traits[2] < 0.4:
            regulated_emotions = regulated_emotions * 1.2
        if traits[4] > 0.6:
            regulated_emotions = raw_emotions * 0.7 + regulated_emotions * 0.3
        return torch.sigmoid(regulated_emotions)


class PersonalityAdaptationNetwork(nn.Module):
    def __init__(self, trait_dim: int = 5, experience_dim: int = 64):
        super().__init__()
        self.experience_encoder = nn.Sequential(
            nn.Linear(experience_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, trait_dim)
        )
        self.trait_update_network = nn.Sequential(
            nn.Linear(trait_dim * 2, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, trait_dim), nn.Tanh()
        )

    def forward(self, current_traits: torch.Tensor, experiences: List[torch.Tensor]) -> torch.Tensor:
        if not experiences:
            return current_traits
        experience_summary = torch.stack(experiences).mean(dim=0)
        experience_influence = self.experience_encoder(experience_summary)
        combined = torch.cat([current_traits, experience_influence], dim=-1)
        trait_updates = self.trait_update_network(combined)
        adapted_traits = current_traits + trait_updates * 0.01
        adapted_traits = torch.clamp(adapted_traits, 0.0, 1.0)
        return adapted_traits


class PersonalityModule(nn.Module):
    def __init__(self, initial_traits: Optional[PersonalityTraits] = None):
        super().__init__()
        self.traits = initial_traits or PersonalityTraits()
        self.emotional_state = EmotionalState()
        self.experience_buffer: List[torch.Tensor] = []
        self.personality_memory: List[Dict[str, Any]] = []
        self.trait_emotion_network = TraitEmotionNetwork()
        self.appraisal_network = AppraisalNetwork()
        self.emotion_regulation_network = EmotionRegulationNetwork()
        self.personality_adaptation_network = PersonalityAdaptationNetwork()
        self.emotion_decay_rate = 0.95
        self.trait_adaptation_rate = 0.001

    def forward(self, stimulus: torch.Tensor, context: Dict[str, Any], social_situation: Optional[Dict[str, Any]] = None) -> Tuple[EmotionalState, PersonalityTraits]:
        appraisal_scores = self.appraisal_network(stimulus, self.traits_to_tensor(), context)
        raw_emotions = self.trait_emotion_network(appraisal_scores, self.traits_to_tensor())
        regulated_emotions = self.emotion_regulation_network(raw_emotions, self.traits_to_tensor(), self.emotional_state_to_tensor())
        self.update_emotional_state(regulated_emotions, context)
        if len(self.experience_buffer) % 100 == 0 and len(self.experience_buffer) > 0:
            self.adapt_personality()
        if social_situation:
            self.apply_social_influence(social_situation)
        return self.emotional_state, self.traits

    # Extended behaviors
    def update_emotional_state(self, new_emotions: torch.Tensor, context: Dict[str, Any]):
        emotions_list = new_emotions.detach().cpu().numpy()
        self.emotional_state.joy *= self.emotion_decay_rate
        self.emotional_state.sadness *= self.emotion_decay_rate
        self.emotional_state.anger *= self.emotion_decay_rate
        self.emotional_state.fear *= self.emotion_decay_rate
        self.emotional_state.disgust *= self.emotion_decay_rate
        self.emotional_state.surprise *= self.emotion_decay_rate
        self.emotional_state.trust *= self.emotion_decay_rate
        self.emotional_state.anticipation *= self.emotion_decay_rate
        self.emotional_state.joy = max(self.emotional_state.joy, float(emotions_list[0]))
        self.emotional_state.sadness = max(self.emotional_state.sadness, float(emotions_list[1]))
        self.emotional_state.anger = max(self.emotional_state.anger, float(emotions_list[2]))
        self.emotional_state.fear = max(self.emotional_state.fear, float(emotions_list[3]))
        self.emotional_state.disgust = max(self.emotional_state.disgust, float(emotions_list[4]))
        self.emotional_state.surprise = max(self.emotional_state.surprise, float(emotions_list[5]))
        self.emotional_state.trust = max(self.emotional_state.trust, float(emotions_list[6]))
        self.emotional_state.anticipation = max(self.emotional_state.anticipation, float(emotions_list[7]))
        self.emotional_state.valence = (
            self.emotional_state.joy + self.emotional_state.trust + self.emotional_state.anticipation
            - self.emotional_state.sadness - self.emotional_state.anger - self.emotional_state.fear
        ) / 3.0
        self.emotional_state.arousal = (
            self.emotional_state.anger + self.emotional_state.fear + self.emotional_state.surprise
            + self.emotional_state.anticipation
        ) / 4.0
        self.emotional_state.intensity = float(np.mean([
            self.emotional_state.joy, self.emotional_state.sadness,
            self.emotional_state.anger, self.emotional_state.fear,
            self.emotional_state.disgust, self.emotional_state.surprise,
            self.emotional_state.trust, self.emotional_state.anticipation
        ]))

    def adapt_personality(self):
        if len(self.experience_buffer) < 50:
            return
        current_traits_tensor = self.traits_to_tensor()
        adapted_traits_tensor = self.personality_adaptation_network(current_traits_tensor, self.experience_buffer[-50:])
        adapted_traits = adapted_traits_tensor.detach().cpu().numpy()
        max_change = self.traits.plasticity
        self.traits.openness = float(np.clip(self.traits.openness + np.clip(adapted_traits[0] - self.traits.openness, -max_change, max_change), 0.0, 1.0))
        self.traits.conscientiousness = float(np.clip(self.traits.conscientiousness + np.clip(adapted_traits[1] - self.traits.conscientiousness, -max_change, max_change), 0.0, 1.0))
        self.traits.extraversion = float(np.clip(self.traits.extraversion + np.clip(adapted_traits[2] - self.traits.extraversion, -max_change, max_change), 0.0, 1.0))
        self.traits.agreeableness = float(np.clip(self.traits.agreeableness + np.clip(adapted_traits[3] - self.traits.agreeableness, -max_change, max_change), 0.0, 1.0))
        self.traits.neuroticism = float(np.clip(self.traits.neuroticism + np.clip(adapted_traits[4] - self.traits.neuroticism, -max_change, max_change), 0.0, 1.0))
        self.personality_memory.append({'timestamp': time.time(), 'traits': self.traits_to_dict(), 'trigger': 'experience_adaptation'})

    def apply_social_influence(self, social_situation: Dict[str, Any]):
        situation_type = social_situation.get('type', 'neutral')
        social_pressure = social_situation.get('pressure', 0.0)
        if self.traits.extraversion > 0.6 and situation_type == 'social_gathering':
            self.emotional_state.joy *= 1.3
            self.emotional_state.arousal *= 1.2
        if self.traits.agreeableness > 0.6 and social_pressure > 0.5:
            self.emotional_state.trust *= 1.2
            self.emotional_state.anger *= 0.7
            self.emotional_state.disgust *= 0.8
        if self.traits.neuroticism > 0.6 and social_situation.get('conflict', False):
            self.emotional_state.fear *= 1.4

    def traits_to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.traits.openness,
            self.traits.conscientiousness,
            self.traits.extraversion,
            self.traits.agreeableness,
            self.traits.neuroticism
        ], dtype=torch.float32)

    def emotional_state_to_tensor(self) -> torch.Tensor:
        return torch.tensor([
            self.emotional_state.joy,
            self.emotional_state.sadness,
            self.emotional_state.anger,
            self.emotional_state.fear,
            self.emotional_state.disgust,
            self.emotional_state.surprise,
            self.emotional_state.trust,
            self.emotional_state.anticipation
        ], dtype=torch.float32)

    def traits_to_dict(self) -> Dict[str, float]:
        return {
            'openness': self.traits.openness,
            'conscientiousness': self.traits.conscientiousness,
            'extraversion': self.traits.extraversion,
            'agreeableness': self.traits.agreeableness,
            'neuroticism': self.traits.neuroticism
        }

    def get_personality_summary(self) -> str:
        traits: List[str] = []
        if self.traits.openness > 0.7:
            traits.append("highly creative and curious")
        elif self.traits.openness < 0.3:
            traits.append("practical and traditional")
        if self.traits.conscientiousness > 0.7:
            traits.append("organized and disciplined")
        elif self.traits.conscientiousness < 0.3:
            traits.append("spontaneous and flexible")
        if self.traits.extraversion > 0.7:
            traits.append("outgoing and energetic")
        elif self.traits.extraversion < 0.3:
            traits.append("reserved and introspective")
        if self.traits.agreeableness > 0.7:
            traits.append("compassionate and cooperative")
        elif self.traits.agreeableness < 0.3:
            traits.append("competitive and skeptical")
        if self.traits.neuroticism > 0.7:
            traits.append("emotionally sensitive")
        elif self.traits.neuroticism < 0.3:
            traits.append("emotionally stable")
        return f"I am {', '.join(traits)}."


