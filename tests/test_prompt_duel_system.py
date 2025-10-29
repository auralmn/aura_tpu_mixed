#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for Prompt Duel Optimizer system.
"""

import unittest
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.prompt_duel.dueler import PromptDueler, OpenSearchRetriever
from aura.prompt_duel.judges import JudgeDecision, hybrid_judge, llm_pairwise_judge


class TestOpenSearchRetriever(unittest.TestCase):
    """Test OpenSearch retrieval functionality."""
    
    def setUp(self):
        self.retriever = OpenSearchRetriever(
            url="http://localhost:9200",
            index="test_index",
            topk=3
        )
    
    @patch('aura.prompt_duel.dueler.requests.get')
    def test_search_success(self, mock_get):
        """Test successful search operation."""
        # Mock successful response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"text": "First result"}},
                    {"_source": {"text": "Second result"}},
                    {"_source": {"text": "Third result"}}
                ]
            }
        }
        mock_get.return_value = mock_response
        
        results = self.retriever.search("test query")
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "First result")
        self.assertEqual(results[1], "Second result")
        self.assertEqual(results[2], "Third result")
    
    @patch('aura.prompt_duel.dueler.requests.get')
    def test_search_failure(self, mock_get):
        """Test search failure handling."""
        # Mock request failure
        mock_get.side_effect = Exception("Connection error")
        
        results = self.retriever.search("test query")
        
        self.assertEqual(results, [])
    
    @patch('aura.prompt_duel.dueler.requests.get')
    def test_search_empty_results(self, mock_get):
        """Test handling of empty search results."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"hits": {"hits": []}}
        mock_get.return_value = mock_response
        
        results = self.retriever.search("test query")
        
        self.assertEqual(results, [])


class TestJudges(unittest.TestCase):
    """Test judging system functionality."""
    
    def test_judge_decision_enum(self):
        """Test JudgeDecision enum values."""
        self.assertEqual(JudgeDecision.A_WINS.value, "A")
        self.assertEqual(JudgeDecision.B_WINS.value, "B")
        self.assertEqual(JudgeDecision.TIE.value, "TIE")
    
    @patch('aura.prompt_duel.judges.requests.post')
    def test_llm_pairwise_judge(self, mock_post):
        """Test LLM pairwise judging."""
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "A"}}]
        }
        mock_post.return_value = mock_response
        
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
            decision = llm_pairwise_judge(
                prompt="Test prompt",
                response_a="Response A",
                response_b="Response B",
                model="gpt-3.5-turbo"
            )
        
        self.assertEqual(decision, JudgeDecision.A_WINS)
    
    def test_hybrid_judge_oracle_wins(self):
        """Test hybrid judge when oracle is confident."""
        oracle_fn = lambda p, a, b: (JudgeDecision.A_WINS, 0.9)  # High confidence
        llm_fn = lambda p, a, b: JudgeDecision.B_WINS  # Different decision
        
        decision = hybrid_judge(
            prompt="test",
            response_a="A",
            response_b="B",
            oracle_judge=oracle_fn,
            llm_judge=llm_fn,
            confidence_threshold=0.8
        )
        
        self.assertEqual(decision, JudgeDecision.A_WINS)
    
    def test_hybrid_judge_llm_fallback(self):
        """Test hybrid judge falls back to LLM when oracle confidence is low."""
        oracle_fn = lambda p, a, b: (JudgeDecision.A_WINS, 0.6)  # Low confidence
        llm_fn = lambda p, a, b: JudgeDecision.B_WINS
        
        decision = hybrid_judge(
            prompt="test",
            response_a="A",
            response_b="B",
            oracle_judge=oracle_fn,
            llm_judge=llm_fn,
            confidence_threshold=0.8
        )
        
        self.assertEqual(decision, JudgeDecision.B_WINS)


class TestPromptDueler(unittest.TestCase):
    """Test PromptDueler functionality."""
    
    def setUp(self):
        self.dueler = PromptDueler(
            candidates=["prompt1", "prompt2", "prompt3"],
            data_source=lambda: ["question1", "question2"],
            generator_fn=lambda p, q: f"response to {q} with {p}",
            judge_fn=lambda p, a, b: JudgeDecision.A_WINS,
            alpha=1.0,
            beta=1.0
        )
    
    def test_initialization(self):
        """Test proper initialization of PromptDueler."""
        self.assertEqual(len(self.dueler.candidates), 3)
        self.assertEqual(len(self.dueler.successes), 3)
        self.assertEqual(len(self.dueler.failures), 3)
        self.assertTrue(all(s == 0 for s in self.dueler.successes))
        self.assertTrue(all(f == 0 for f in self.dueler.failures))
    
    def test_select_candidates(self):
        """Test candidate selection using Thompson sampling."""
        # Run selection multiple times to test randomness
        selections = []
        for _ in range(10):
            idx_a, idx_b = self.dueler._select_candidates()
            selections.append((idx_a, idx_b))
            self.assertNotEqual(idx_a, idx_b)
            self.assertIn(idx_a, range(3))
            self.assertIn(idx_b, range(3))
        
        # Should have some variation in selections
        unique_selections = set(selections)
        self.assertGreater(len(unique_selections), 1)
    
    def test_update_scores(self):
        """Test score updating after duel."""
        initial_successes = self.dueler.successes[0]
        initial_failures = self.dueler.failures[1]
        
        self.dueler._update_scores(winner_idx=0, loser_idx=1)
        
        self.assertEqual(self.dueler.successes[0], initial_successes + 1)
        self.assertEqual(self.dueler.failures[1], initial_failures + 1)
    
    def test_run_single_duel(self):
        """Test running a single duel."""
        result = self.dueler.run_single_duel()
        
        self.assertIn('winner_idx', result)
        self.assertIn('loser_idx', result)
        self.assertIn('question', result)
        self.assertIn('winner_response', result)
        self.assertIn('loser_response', result)
        self.assertIn('winner_prompt', result)
        self.assertIn('loser_prompt', result)
    
    def test_get_best_prompt(self):
        """Test getting the best prompt based on success rates."""
        # Manually set some scores to test
        self.dueler.successes = [5, 2, 8]
        self.dueler.failures = [1, 3, 2]
        
        best_idx, best_prompt, success_rate = self.dueler.get_best_prompt()
        
        # Should return the prompt with highest success rate
        # Prompt 2: 8/(8+2) = 0.8, Prompt 0: 5/(5+1) = 0.833..., Prompt 1: 2/(2+3) = 0.4
        self.assertEqual(best_idx, 0)
        self.assertEqual(best_prompt, "prompt1")
        self.assertAlmostEqual(success_rate, 5/6, places=2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full prompt duel system."""
    
    def test_end_to_end_duel(self):
        """Test end-to-end duel execution."""
        candidates = ["Be helpful and concise", "Be detailed and thorough", "Be creative and engaging"]
        questions = ["What is Python?", "How does machine learning work?"]
        
        def mock_generator(prompt, question):
            return f"Answer about {question.split()[2] if len(question.split()) > 2 else 'topic'} - {prompt[:10]}"
        
        def mock_judge(prompt, response_a, response_b):
            # Simple heuristic: prefer longer responses
            return JudgeDecision.A_WINS if len(response_a) > len(response_b) else JudgeDecision.B_WINS
        
        dueler = PromptDueler(
            candidates=candidates,
            data_source=lambda: questions,
            generator_fn=mock_generator,
            judge_fn=mock_judge
        )
        
        # Run several duels
        results = []
        for _ in range(5):
            result = dueler.run_single_duel()
            results.append(result)
        
        self.assertEqual(len(results), 5)
        
        # Check that scores were updated
        total_duels = sum(dueler.successes) + sum(dueler.failures)
        self.assertEqual(total_duels, 5)
        
        # Get final best prompt
        best_idx, best_prompt, success_rate = dueler.get_best_prompt()
        self.assertIn(best_idx, range(3))
        self.assertIn(best_prompt, candidates)
        self.assertGreaterEqual(success_rate, 0.0)
        self.assertLessEqual(success_rate, 1.0)


if __name__ == '__main__':
    unittest.main()
