# aura/memory/consolidation.py
"""
Memory Consolidation System
Handles consolidation rules and strategies
"""

from typing import List, Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import asyncio


class ConsolidationStrategy(Enum):
    """Memory consolidation strategies"""
    PROMOTE_LAYER = "promote_layer"
    MERGE_SIMILAR = "merge_similar"
    STRENGTHEN = "strengthen"
    CREATE_SUMMARY = "create_summary"


@dataclass
class ConsolidationRule:
    """Rule for memory consolidation"""
    name: str
    condition: Callable[['MemoryItem'], bool]
    action: ConsolidationStrategy
    priority: int
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class MemoryConsolidator:
    """Handles memory consolidation"""
    
    def __init__(self, rules: List[ConsolidationRule]):
        self.rules = sorted(rules, key=lambda r: r.priority)
    
    def get_rules(self) -> List[ConsolidationRule]:
        """Get consolidation rules"""
        return self.rules.copy()
    
    async def consolidate(self, memories: List['MemoryItem']) -> Dict[str, Any]:
        """Apply consolidation rules to memories"""
        results = {
            "memory_count_before": len(memories),
            "actions_taken": [],
            "consolidated_memories": [],
            "memory_count_after": 0
        }
        
        processed_memories = memories.copy()
        
        for rule in self.rules:
            rule_actions = []
            
            if rule.action == ConsolidationStrategy.PROMOTE_LAYER:
                for memory in processed_memories:
                    if rule.condition(memory):
                        rule_actions.append({
                            "rule": rule.name,
                            "action": "promote_layer",
                            "memory_id": memory.id,
                            "old_type": memory.memory_type.value,
                            "new_type": "promoted"
                        })
            
            elif rule.action == ConsolidationStrategy.MERGE_SIMILAR:
                # Find similar memories for merging
                similarity_threshold = rule.parameters.get("similarity_threshold", 0.8)
                merged_pairs = []
                
                for i, memory1 in enumerate(processed_memories):
                    for j, memory2 in enumerate(processed_memories[i+1:], i+1):
                        if (rule.condition(memory1) and rule.condition(memory2) and
                            self._calculate_similarity(memory1, memory2) >= similarity_threshold):
                            merged_pairs.append((memory1, memory2))
                
                for memory1, memory2 in merged_pairs:
                    rule_actions.append({
                        "rule": rule.name,
                        "action": "merge_similar",
                        "memory_ids": [memory1.id, memory2.id],
                        "similarity": float(self._calculate_similarity(memory1, memory2))
                    })
            
            results["actions_taken"].extend(rule_actions)
        
        results["memory_count_after"] = len(processed_memories)
        results["consolidated_memories"] = processed_memories
        
        return results
    
    def _calculate_similarity(self, memory1: 'MemoryItem', memory2: 'MemoryItem') -> float:
        """Calculate similarity between two memories"""
        if memory1.embedding is None or memory2.embedding is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(memory1.embedding, memory2.embedding)
        norm1 = np.linalg.norm(memory1.embedding)
        norm2 = np.linalg.norm(memory2.embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
