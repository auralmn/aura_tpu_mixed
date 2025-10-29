#!/usr/bin/env python3
"""
Conversational Learning Engine for AURA

Enables real-time learning from chat interactions to improve responses
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from collections import deque
import json
import time

class ConversationalLearner(nn.Module):
    """Learns from chat interactions in real-time"""
    
    def __init__(self, config: Dict[str, Any],
                 embedding_generator: Optional[Any] = None,
                 memory_system: Optional[Any] = None,
                 consciousness_system: Optional[Any] = None,
                 text_generator: Optional[Any] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.get('hidden_size', 768)
        self.learning_rate = config.get('learning_rate', 1e-4)
        
        # Optional integrations
        self.embedding_generator = embedding_generator
        self.memory_system = memory_system
        self.consciousness_system = consciousness_system
        self.text_generator = text_generator
        
        # Response quality predictor
        self.quality_predictor = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 256),  # user + aura embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Response improvement network
        self.response_improver = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, self.hidden_size)
        )
        
        # Conversation memory
        self.conversation_buffer = deque(maxlen=1000)
        self.feedback_buffer = deque(maxlen=500)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def learn_from_interaction(self, user_input: str, aura_response: str, 
                             user_feedback: Optional[float] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """Learn from a single chat interaction"""
        # Store interaction
        interaction = {
            'user_input': user_input,
            'aura_response': aura_response,
            'feedback': user_feedback,
            'timestamp': time.time()
        }
        self.conversation_buffer.append(interaction)
        
        # Persist to memory and consciousness if available
        try:
            if self.embedding_generator is not None and self.memory_system is not None:
                from datetime import datetime, timezone
                from aura.memory.hierarchical_memory import MemoryItem, MemoryType
                
                text = f"User: {user_input}\nAURA: {aura_response}"
                embedding = self.embedding_generator.generate_embedding(text)
                mem = MemoryItem(
                    id=f"conv_{int(time.time()*1000)}",
                    content=text,
                    embedding=embedding,
                    metadata=metadata or {'source': 'conversational_learning'},
                    memory_type=MemoryType.SHORT_TERM,
                    strength=0.5,
                    created_at=datetime.now(timezone.utc)
                )
                self.memory_system.store_memory_sync(mem)
                
                if self.consciousness_system is not None:
                    self.consciousness_system.add_knowledge(text, embedding)
        except Exception:
            pass
        
        if user_feedback is not None:
            self.feedback_buffer.append(interaction)
            self._update_from_feedback()
    
    def _update_from_feedback(self):
        """Update model based on user feedback"""
        if len(self.feedback_buffer) < 5:
            return
        
        # Sample recent feedback
        recent_feedback = list(self.feedback_buffer)[-5:]
        
        for interaction in recent_feedback:
            # Convert to embeddings using real generator if available
            if self.embedding_generator is not None:
                try:
                    user_emb_np = self.embedding_generator.generate_embedding(interaction['user_input'])
                    aura_emb_np = self.embedding_generator.generate_embedding(interaction['aura_response'])
                    user_emb = torch.tensor(user_emb_np, dtype=torch.float32).unsqueeze(0)
                    aura_emb = torch.tensor(aura_emb_np, dtype=torch.float32).unsqueeze(0)
                    # Align dims if needed
                    if user_emb.size(1) != self.hidden_size:
                        user_emb = torch.nn.functional.pad(user_emb, (0, max(0, self.hidden_size - user_emb.size(1))))[:, :self.hidden_size]
                    if aura_emb.size(1) != self.hidden_size:
                        aura_emb = torch.nn.functional.pad(aura_emb, (0, max(0, self.hidden_size - aura_emb.size(1))))[:, :self.hidden_size]
                except Exception:
                    user_emb = torch.randn(1, self.hidden_size)
                    aura_emb = torch.randn(1, self.hidden_size)
            else:
                user_emb = torch.randn(1, self.hidden_size)
                aura_emb = torch.randn(1, self.hidden_size)
            
            # Predict quality
            combined = torch.cat([user_emb, aura_emb], dim=1)
            predicted_quality = self.quality_predictor(combined)
            
            # Actual feedback as target
            target_quality = torch.tensor([[interaction['feedback']]], dtype=torch.float32)
            
            # Update quality predictor
            loss = nn.MSELoss()(predicted_quality, target_quality)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        if not self.conversation_buffer:
            return {'total_interactions': 0}
        
        feedback_scores = [i['feedback'] for i in self.conversation_buffer if i['feedback'] is not None]
        
        return {
            'total_interactions': len(self.conversation_buffer),
            'feedback_count': len(feedback_scores),
            'avg_feedback': sum(feedback_scores) / len(feedback_scores) if feedback_scores else 0.0,
            'recent_feedback': feedback_scores[-10:] if feedback_scores else []
        }
    
    def save_conversation_history(self, path: str):
        """Save conversation history to file"""
        with open(path, 'w') as f:
            json.dump(list(self.conversation_buffer), f, indent=2)
    
    def load_conversation_history(self, path: str):
        """Load conversation history from file"""
        try:
            with open(path, 'r') as f:
                history = json.load(f)
            self.conversation_buffer.extend(history)
            
            # Rebuild feedback buffer
            for interaction in history:
                if interaction.get('feedback') is not None:
                    self.feedback_buffer.append(interaction)
        except FileNotFoundError:
            pass
