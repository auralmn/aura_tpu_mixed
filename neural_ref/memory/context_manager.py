# aura/memory/context_manager.py
"""
Context Management System
Manages contextual information and retrieval
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import asyncio
import threading
import uuid


class ContextType(Enum):
    """Types of contexts"""
    NEURAL_SIMULATION = "neural_simulation"
    CONVERSATION = "conversation"
    LEARNING_SESSION = "learning_session"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    PROBLEM_SOLVING = "problem_solving"


class RetrievalStrategy(Enum):
    """Context retrieval strategies"""
    RELEVANCE_WEIGHTED = "relevance_weighted"
    TEMPORAL_PRIORITY = "temporal_priority"
    ACCESS_FREQUENCY = "access_frequency"
    HYBRID = "hybrid"


@dataclass
class Context:
    """Context information"""
    id: str
    type: ContextType
    title: str
    content: Dict[str, Any]
    embedding: np.ndarray
    metadata: Dict[str, Any]
    relevance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    def access(self) -> None:
        """Record access to context"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def calculate_relevance(self, query_embedding: np.ndarray, 
                          time_weight: float = 0.3) -> float:
        """Calculate relevance to query"""
        if self.embedding is None or query_embedding is None:
            return self.relevance_score
        
        # Semantic similarity
        semantic_similarity = np.dot(self.embedding, query_embedding) / (
            np.linalg.norm(self.embedding) * np.linalg.norm(query_embedding)
        )
        
        # Time relevance (more recent is more relevant)
        age_hours = (datetime.now(timezone.utc) - self.last_accessed).total_seconds() / 3600
        time_relevance = np.exp(-age_hours / 24)  # 24-hour half-life
        
        # Combined score
        combined_score = (
            (1 - time_weight) * semantic_similarity + 
            time_weight * time_relevance
        ) * self.relevance_score
        
        return float(combined_score)


@dataclass
class ContextQuery:
    """Query for context retrieval"""
    embedding: Optional[np.ndarray] = None
    context_types: List[ContextType] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    min_relevance: float = 0.3
    max_results: int = 10
    time_range: Optional[Tuple[datetime, datetime]] = None
    strategy: RetrievalStrategy = RetrievalStrategy.RELEVANCE_WEIGHTED


class ContextManager:
    """Manages contextual information"""
    
    def __init__(self, max_context_size: int = 1000, context_window: int = 100, 
                 relevance_threshold: float = 0.6):
        self.max_context_size = max_context_size
        self.context_window = context_window
        self.relevance_threshold = relevance_threshold
        self._contexts: Dict[str, Context] = {}
        self._access_order: List[str] = []
        self._type_index: Dict[ContextType, List[str]] = {}
        self._lock = threading.RLock()
    
    async def add_context(self, context: Context) -> bool:
        """Add context to manager"""
        with self._lock:
            # Check window size limit
            if len(self._contexts) >= self.context_window:
                self._evict_contexts()
            
            self._contexts[context.id] = context
            self._update_access_order(context.id)
            
            # Update type index
            if context.type not in self._type_index:
                self._type_index[context.type] = []
            self._type_index[context.type].append(context.id)
            
            return True
    
    async def get_context(self, context_id: str) -> Optional[Context]:
        """Get specific context"""
        with self._lock:
            context = self._contexts.get(context_id)
            if context:
                context.access()
                self._update_access_order(context_id)
            return context
    
    async def get_contexts(self, context_type: Optional[ContextType] = None) -> List[Context]:
        """Get all contexts, optionally filtered by type"""
        with self._lock:
            if context_type is None:
                return list(self._contexts.values())
            
            context_ids = self._type_index.get(context_type, [])
            return [self._contexts[cid] for cid in context_ids if cid in self._contexts]
    
    async def query_contexts(self, query: ContextQuery) -> List[Context]:
        """Query contexts based on criteria"""
        with self._lock:
            candidates = []
            
            # Filter by type if specified
            if query.context_types:
                for context_type in query.context_types:
                    context_ids = self._type_index.get(context_type, [])
                    candidates.extend([
                        self._contexts[cid] for cid in context_ids 
                        if cid in self._contexts
                    ])
            else:
                candidates = list(self._contexts.values())
            
            # Apply filters
            filtered_contexts = []
            for context in candidates:
                # Time range filter
                if query.time_range:
                    start_time, end_time = query.time_range
                    if not (start_time <= context.created_at <= end_time):
                        continue
                
                # Relevance filter
                if query.embedding is not None:
                    relevance = context.calculate_relevance(query.embedding)
                    if relevance < query.min_relevance:
                        continue
                    filtered_contexts.append((relevance, context))
                else:
                    if context.relevance_score >= query.min_relevance:
                        filtered_contexts.append((context.relevance_score, context))
            
            # Sort by strategy
            if query.strategy == RetrievalStrategy.RELEVANCE_WEIGHTED:
                filtered_contexts.sort(key=lambda x: x[0], reverse=True)
            elif query.strategy == RetrievalStrategy.TEMPORAL_PRIORITY:
                filtered_contexts.sort(key=lambda x: x[1].last_accessed, reverse=True)
            elif query.strategy == RetrievalStrategy.ACCESS_FREQUENCY:
                filtered_contexts.sort(key=lambda x: x[1].access_count, reverse=True)
            
            # Return top results
            results = [context for _, context in filtered_contexts[:query.max_results]]
            
            # Update access for returned contexts
            for context in results:
                context.access()
            
            return results
    
    def _evict_contexts(self, num_contexts: int = 1) -> None:
        """Evict oldest/least relevant contexts"""
        if not self._access_order:
            return
        
        # Evict least recently accessed
        for _ in range(min(num_contexts, len(self._access_order))):
            oldest_id = self._access_order[0]
            context = self._contexts.get(oldest_id)
            
            if context:
                # Remove from main storage
                del self._contexts[oldest_id]
                self._access_order.remove(oldest_id)
                
                # Remove from type index
                if context.type in self._type_index:
                    if oldest_id in self._type_index[context.type]:
                        self._type_index[context.type].remove(oldest_id)
    
    def _update_access_order(self, context_id: str) -> None:
        """Update access order for LRU"""
        if context_id in self._access_order:
            self._access_order.remove(context_id)
        self._access_order.append(context_id)
    
    async def apply_relevance_decay(self, decay_rate: float = 0.05) -> None:
        """Apply relevance decay over time"""
        with self._lock:
            current_time = datetime.now(timezone.utc)
            
            for context in self._contexts.values():
                # Calculate hours since last access
                hours_since_access = (current_time - context.last_accessed).total_seconds() / 3600
                
                # Apply exponential decay
                decay_factor = np.exp(-decay_rate * hours_since_access)
                context.relevance_score *= decay_factor
                
                # Minimum relevance threshold
                context.relevance_score = max(context.relevance_score, 0.01)
