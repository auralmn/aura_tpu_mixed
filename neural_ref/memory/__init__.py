"""
AURA Memory System
Hierarchical memory architecture with context management
"""

from .hierarchical_memory import (
    HierarchicalMemory, MemoryLayer, MemoryType, MemoryItem, MemoryQuery
)
from .context_manager import (
    ContextManager, Context, ContextType, ContextQuery, RetrievalStrategy
)
from .consolidation import (
    MemoryConsolidator, ConsolidationRule, ConsolidationStrategy
)
from .forgetting import (
    ForgettingMechanism, ForgettingStrategy, DecayFunction
)

__all__ = [
    'HierarchicalMemory',
    'MemoryLayer',
    'MemoryType',
    'MemoryItem',
    'MemoryQuery',
    'ContextManager',
    'Context',
    'ContextType',
    'ContextQuery',
    'RetrievalStrategy',
    'MemoryConsolidator',
    'ConsolidationRule',
    'ConsolidationStrategy',
    'ForgettingMechanism',
    'ForgettingStrategy',
    'DecayFunction'
]
