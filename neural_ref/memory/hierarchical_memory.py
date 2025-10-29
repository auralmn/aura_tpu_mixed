# aura/memory/hierarchical_memory.py
"""
Hierarchical Memory System Implementation
Provides multi-layer memory architecture with biological inspiration
"""

import asyncio
import numpy as np
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import heapq
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory layers"""
    SENSORY = "sensory"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"


@dataclass
class MemoryItem:
    """Individual memory item"""
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    memory_type: MemoryType
    strength: float
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    def access(self) -> None:
        """Record access to this memory item"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def calculate_relevance(self, query_embedding: np.ndarray) -> float:
        """Calculate relevance to query embedding"""
        if self.embedding is None or query_embedding is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(self.embedding, query_embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(query_embedding)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Weight by strength and recency
        age_hours = (datetime.now(timezone.utc) - self.last_accessed).total_seconds() / 3600
        recency_weight = np.exp(-age_hours / 24)  # Decay over 24 hours
        
        return similarity * self.strength * recency_weight


class MemoryLayer:
    """Single layer in hierarchical memory"""
    
    def __init__(self, memory_type: MemoryType, capacity: int, decay_rate: float = 0.01):
        self.memory_type = memory_type
        self.capacity = capacity
        self.decay_rate = decay_rate
        self._items: Dict[str, MemoryItem] = {}
        self._access_order: List[str] = []  # LRU tracking
        self._lock = threading.RLock()
    
    def get_items(self) -> List[MemoryItem]:
        """Get all items in this layer"""
        with self._lock:
            return list(self._items.values())
    
    def get_item(self, item_id: str) -> Optional[MemoryItem]:
        """Get specific item by ID"""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.access()
                self._update_access_order(item_id)
            return item
    
    def store_item(self, item: MemoryItem) -> bool:
        """Store item in this layer"""
        with self._lock:
            # If item already exists, update it
            if item.id in self._items:
                self._items[item.id] = item
                self._update_access_order(item.id)
                return True
            
            # Check if we need to evict items before adding new one
            if len(self._items) >= self.capacity:
                # Evict one item to make room
                self._evict_items(1)
            
            # Update memory type to match layer
            item.memory_type = self.memory_type
            
            self._items[item.id] = item
            self._update_access_order(item.id)
            
            return True
    
    def remove_item(self, item_id: str) -> Optional[MemoryItem]:
        """Remove item from layer"""
        with self._lock:
            item = self._items.pop(item_id, None)
            if item_id in self._access_order:
                self._access_order.remove(item_id)
            return item
    
    def _evict_items(self, num_items: int = 1) -> List[MemoryItem]:
        """Evict least recently used items"""
        evicted = []
        
        # Sort by LRU order and strength
        candidates = [
            (item_id, self._items[item_id])
            for item_id in self._access_order
        ]
        
        # Sort by access time and strength (older and weaker first)
        candidates.sort(key=lambda x: (x[1].last_accessed, x[1].strength))
        
        for i in range(min(num_items, len(candidates))):
            item_id, item = candidates[i]
            evicted_item = self.remove_item(item_id)
            if evicted_item:
                evicted.append(evicted_item)
        
        return evicted
    
    def _update_access_order(self, item_id: str) -> None:
        """Update LRU access order"""
        if item_id in self._access_order:
            self._access_order.remove(item_id)
        self._access_order.append(item_id)
    
    def find_similar(self, query_embedding: np.ndarray, 
                    top_k: int = 5, threshold: float = 0.5) -> List[MemoryItem]:
        """Find similar items in this layer"""
        with self._lock:
            scored_items = []
            
            for item in self._items.values():
                relevance = item.calculate_relevance(query_embedding)
                if relevance >= threshold:
                    scored_items.append((relevance, item))
            
            # Sort by relevance (descending)
            scored_items.sort(key=lambda x: x[0], reverse=True)
            
            return [item for _, item in scored_items[:top_k]]
    
    def apply_decay(self) -> Dict[str, Any]:
        """Apply decay to all items in layer"""
        with self._lock:
            decayed_count = 0
            removed_count = 0
            items_to_remove = []
            
            for item in self._items.values():
                # Calculate time since last access
                time_diff = datetime.now(timezone.utc) - item.last_accessed
                hours_since_access = time_diff.total_seconds() / 3600
                
                # Apply exponential decay
                decay_factor = np.exp(-self.decay_rate * hours_since_access)
                new_strength = item.strength * decay_factor
                
                if new_strength < 0.1:  # Minimum threshold
                    items_to_remove.append(item.id)
                else:
                    item.strength = new_strength
                    decayed_count += 1
            
            # Remove weak items
            for item_id in items_to_remove:
                self.remove_item(item_id)
                removed_count += 1
            
            return {
                "decayed_count": decayed_count,
                "removed_count": removed_count,
                "remaining_count": len(self._items)
            }


@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    embedding: Optional[np.ndarray] = None
    content_keywords: List[str] = field(default_factory=list)
    metadata_filters: Dict[str, Any] = field(default_factory=dict)
    memory_types: List[MemoryType] = field(default_factory=list)
    max_results: int = 10
    min_relevance: float = 0.3
    time_range: Optional[Tuple[datetime, datetime]] = None


class HierarchicalMemory:
    """Hierarchical memory system with multiple layers"""
    
    def __init__(self, layers: List[MemoryLayer]):
        self.layers = {layer.memory_type: layer for layer in layers}
        self._lock = threading.RLock()
    
    def get_layers(self) -> List[MemoryLayer]:
        """Get all memory layers"""
        return list(self.layers.values())
    
    def get_layer(self, memory_type: MemoryType) -> Optional[MemoryLayer]:
        """Get specific memory layer"""
        return self.layers.get(memory_type)
    
    async def store_memory(self, memory_item: MemoryItem) -> bool:
        """Store memory in appropriate layer"""
        layer = self.layers.get(memory_item.memory_type)
        if not layer:
            return False
        
        return layer.store_item(memory_item)
    
    def store_memory_sync(self, memory_item: MemoryItem) -> bool:
        """Synchronous version for testing"""
        return asyncio.run(self.store_memory(memory_item))
    
    async def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve memory by ID from any layer"""
        for layer in self.layers.values():
            item = layer.get_item(memory_id)
            if item:
                return item
        return None
    
    async def retrieve_similar(self, query_embedding: np.ndarray, 
                             top_k: int = 5, threshold: float = 0.5) -> List[MemoryItem]:
        """Retrieve similar memories across all layers"""
        all_results = []
        
        for layer in self.layers.values():
            layer_results = layer.find_similar(query_embedding, top_k * 2, threshold)
            all_results.extend(layer_results)
        
        # Sort all results by relevance
        scored_results = [
            (item.calculate_relevance(query_embedding), item)
            for item in all_results
        ]
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        return [item for _, item in scored_results[:top_k]]
    
    async def consolidate_memories(self) -> Dict[str, Any]:
        """Consolidate memories between layers"""
        results = {
            "promoted_count": 0,
            "merged_count": 0,
            "consolidated_items": []
        }
        
        # Promote strong sensory memories to short-term
        if MemoryType.SENSORY in self.layers and MemoryType.SHORT_TERM in self.layers:
            sensory_layer = self.layers[MemoryType.SENSORY]
            short_term_layer = self.layers[MemoryType.SHORT_TERM]
            
            for item in sensory_layer.get_items():
                if item.strength >= 0.7:  # Strong memories
                    # Move to short-term
                    sensory_layer.remove_item(item.id)
                    item.memory_type = MemoryType.SHORT_TERM
                    short_term_layer.store_item(item)
                    results["promoted_count"] += 1
                    results["consolidated_items"].append(item.id)
        
        # Promote important short-term memories to long-term
        if MemoryType.SHORT_TERM in self.layers and MemoryType.LONG_TERM in self.layers:
            short_term_layer = self.layers[MemoryType.SHORT_TERM]
            long_term_layer = self.layers[MemoryType.LONG_TERM]
            
            for item in short_term_layer.get_items():
                # Promote if high access count or very strong
                if item.access_count >= 3 or item.strength >= 0.9:
                    short_term_layer.remove_item(item.id)
                    item.memory_type = MemoryType.LONG_TERM
                    long_term_layer.store_item(item)
                    results["promoted_count"] += 1
                    results["consolidated_items"].append(item.id)
        
        return results
