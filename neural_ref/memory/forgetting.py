# aura/memory/forgetting.py
"""
Memory Forgetting and Decay System
"""

from typing import List, Dict, Any
from enum import Enum
import numpy as np
import asyncio
from datetime import datetime, timezone


class ForgettingStrategy(Enum):
    """Forgetting strategies"""
    EXPONENTIAL_DECAY = "exponential_decay"
    LINEAR_DECAY = "linear_decay"
    POWER_LAW_DECAY = "power_law_decay"


class DecayFunction(Enum):
    """Decay function types"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    POWER_LAW = "power_law"


class ForgettingMechanism:
    """Handles memory forgetting and decay"""
    
    def __init__(self, strategy: ForgettingStrategy, decay_function: DecayFunction,
                 base_decay_rate: float = 0.01, minimum_strength: float = 0.1):
        self.strategy = strategy
        self.decay_function = decay_function
        self.base_decay_rate = base_decay_rate
        self.minimum_strength = minimum_strength
    
    async def apply_decay(self, memory: 'MemoryItem') -> 'MemoryItem':
        """Apply decay to a single memory item"""
        current_time = datetime.now(timezone.utc)
        time_diff = current_time - memory.last_accessed
        hours_since_access = time_diff.total_seconds() / 3600
        
        if self.decay_function == DecayFunction.EXPONENTIAL:
            decay_factor = np.exp(-self.base_decay_rate * hours_since_access)
        elif self.decay_function == DecayFunction.LINEAR:
            decay_factor = max(0, 1 - self.base_decay_rate * hours_since_access)
        else:  # POWER_LAW
            decay_factor = np.power(1 + hours_since_access, -self.base_decay_rate)
        
        new_strength = memory.strength * decay_factor
        new_strength = max(new_strength, self.minimum_strength)
        
        # Create updated memory item
        from copy import deepcopy
        decayed_memory = deepcopy(memory)
        decayed_memory.strength = new_strength
        
        return decayed_memory
    
    async def should_forget(self, memory: 'MemoryItem') -> bool:
        """Determine if memory should be forgotten"""
        return memory.strength < self.minimum_strength
    
    async def apply_batch_decay(self, memories: List['MemoryItem']) -> Dict[str, Any]:
        """Apply decay to batch of memories"""
        results = {
            "processed_count": len(memories),
            "forgotten_count": 0,
            "decayed_memories": []
        }
        
        for memory in memories:
            decayed_memory = await self.apply_decay(memory)
            
            if await self.should_forget(decayed_memory):
                results["forgotten_count"] += 1
            else:
                results["decayed_memories"].append(decayed_memory)
        
        return results
