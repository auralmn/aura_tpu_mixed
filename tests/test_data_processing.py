#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for data processing components.
"""

import unittest
import os
import sys
import tempfile
import json
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.ingestion.txt_loader import load_text_corpus, load_json_corpus, load_text_corpus_all, _affect_vector
from aura.data.hf_stream import get_stream_fn
from aura.data.hf_conversations import stream_conversations


class TestTxtLoader(unittest.TestCase):
    """Test text corpus loading functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_json_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.temp_json_dir)
    
    def test_affect_vector(self):
        """Test affect vector computation."""
        # Test text with joy keywords
        joy_text = "I am very happy and grateful for this wonderful day"
        vec = _affect_vector(joy_text)
        
        self.assertEqual(vec.shape, (8,))  # 8 emotions
        self.assertTrue(np.all(vec >= 0))
        self.assertAlmostEqual(np.sum(vec), 1.0, places=5)  # Should sum to 1
        self.assertGreater(vec[0], 0)  # Joy should be > 0
    
    def test_affect_vector_neutral(self):
        """Test affect vector for neutral text."""
        neutral_text = "The weather forecast shows clouds tomorrow"
        vec = _affect_vector(neutral_text)
        
        self.assertEqual(vec.shape, (8,))
        self.assertTrue(np.all(vec == 0))  # No emotional keywords
    
    def test_load_text_corpus_empty_dir(self):
        """Test loading from empty directory."""
        items = load_text_corpus(self.temp_dir)
        self.assertEqual(len(items), 0)
    
    def test_load_text_corpus_with_files(self):
        """Test loading text files."""
        # Create test files
        test_files = {
            'happy.txt': 'I am so happy and joyful today!',
            'sad.txt': 'I feel very sad and depressed.',
            'neutral.txt': 'This is a neutral statement about weather.',
            'empty.txt': '',
            'not_txt.pdf': 'This should be ignored'
        }
        
        for filename, content in test_files.items():
            with open(os.path.join(self.temp_dir, filename), 'w') as f:
                f.write(content)
        
        items = load_text_corpus(self.temp_dir)
        
        # Should load 3 non-empty .txt files
        self.assertEqual(len(items), 3)
        
        # Check that all items have text and affect vector
        for text, vec in items:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            self.assertEqual(vec.shape, (8,))
            self.assertTrue(np.all(vec >= 0))
    
    def test_load_json_corpus_empty_dir(self):
        """Test loading from empty JSON directory."""
        items = load_json_corpus(self.temp_json_dir)
        self.assertEqual(len(items), 0)
    
    def test_load_json_corpus_with_files(self):
        """Test loading JSON files."""
        # Create test JSON files
        json_data = [
            {'text': 'I am happy and excited!'},
            {'content': 'This makes me feel sad and lonely.'},
            {'body': 'Neutral information about topics.'},
            {'message': 'Another joyful and delightful message!'}
        ]
        
        # Regular JSON file
        with open(os.path.join(self.temp_json_dir, 'test.json'), 'w') as f:
            json.dump(json_data, f)
        
        # JSONL file
        with open(os.path.join(self.temp_json_dir, 'test.jsonl'), 'w') as f:
            for item in json_data:
                f.write(json.dumps(item) + '\n')
        
        items = load_json_corpus(self.temp_json_dir)
        
        # Should load all 8 items (4 from JSON + 4 from JSONL)
        self.assertEqual(len(items), 8)
        
        # Check that all items have text and affect vector
        for text, vec in items:
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            self.assertEqual(vec.shape, (8,))
    
    def test_load_json_corpus_custom_keys(self):
        """Test loading JSON with custom text keys."""
        custom_data = [
            {'title': 'Happy title'},
            {'description': 'Sad description'},
            {'summary': 'Neutral summary'}
        ]
        
        with open(os.path.join(self.temp_json_dir, 'custom.json'), 'w') as f:
            json.dump(custom_data, f)
        
        items = load_json_corpus(self.temp_json_dir, text_keys=['title', 'description', 'summary'])
        
        self.assertEqual(len(items), 3)
        
        # Check that we got the right texts
        texts = [item[0] for item in items]
        self.assertIn('Happy title', texts)
        self.assertIn('Sad description', texts)
        self.assertIn('Neutral summary', texts)
    
    def test_load_text_corpus_all(self):
        """Test loading from both text and JSON directories."""
        # Create text file
        with open(os.path.join(self.temp_dir, 'test.txt'), 'w') as f:
            f.write('Happy text content')
        
        # Create JSON file
        with open(os.path.join(self.temp_json_dir, 'test.json'), 'w') as f:
            json.dump([{'text': 'Sad JSON content'}], f)
        
        items = load_text_corpus_all(self.temp_dir, self.temp_json_dir)
        
        self.assertEqual(len(items), 2)
        texts = [item[0] for item in items]
        self.assertIn('Happy text content', texts)
        self.assertIn('Sad JSON content', texts)


class TestHFStream(unittest.TestCase):
    """Test Hugging Face streaming functionality."""
    
    def test_get_stream_fn_identity(self):
        """Test identity stream function."""
        stream_fn = get_stream_fn('identity')
        
        # Test with sample data
        sample_data = [
            {'text': 'Hello world'},
            {'text': 'Another message'}
        ]
        
        result = list(stream_fn(sample_data))
        self.assertEqual(result, sample_data)
    
    def test_get_stream_fn_conversations(self):
        """Test conversations stream function."""
        stream_fn = get_stream_fn('conversations')
        
        # Test with conversation data
        conversation_data = [{
            'conversations': [
                {'from': 'user', 'value': 'Hello'},
                {'from': 'assistant', 'value': 'Hi there!'},
                {'from': 'user', 'value': 'How are you?'},
                {'from': 'assistant', 'value': 'I am doing well!'}
            ]
        }]
        
        result = list(stream_fn(conversation_data))
        
        # Should flatten to individual turns
        self.assertGreater(len(result), 0)
        
        # Each result should have 'text' field
        for item in result:
            self.assertIn('text', item)
            self.assertIsInstance(item['text'], str)
    
    def test_get_stream_fn_unknown(self):
        """Test unknown stream function defaults to identity."""
        stream_fn = get_stream_fn('unknown_format')
        
        sample_data = [{'test': 'data'}]
        result = list(stream_fn(sample_data))
        self.assertEqual(result, sample_data)


class TestHFConversations(unittest.TestCase):
    """Test Hugging Face conversations processing."""
    
    def test_stream_conversations_basic(self):
        """Test basic conversation streaming."""
        data = [{
            'conversations': [
                {'from': 'user', 'value': 'What is AI?'},
                {'from': 'assistant', 'value': 'AI stands for Artificial Intelligence.'},
                {'from': 'user', 'value': 'Tell me more.'},
                {'from': 'assistant', 'value': 'AI involves creating systems that can think and learn.'}
            ]
        }]
        
        result = list(stream_conversations(data))
        
        self.assertGreater(len(result), 0)
        
        # Check that we get flattened conversation turns
        for item in result:
            self.assertIn('text', item)
            self.assertIsInstance(item['text'], str)
            self.assertGreater(len(item['text']), 0)
    
    def test_stream_conversations_empty(self):
        """Test streaming empty conversations."""
        data = [{'conversations': []}]
        
        result = list(stream_conversations(data))
        
        # Should handle empty conversations gracefully
        self.assertIsInstance(result, list)
    
    def test_stream_conversations_missing_field(self):
        """Test streaming with missing conversations field."""
        data = [{'other_field': 'value'}]
        
        result = list(stream_conversations(data))
        
        # Should handle missing field gracefully
        self.assertIsInstance(result, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for data processing pipeline."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_data_pipeline(self):
        """Test full data processing pipeline."""
        # Create mixed content
        texts = [
            "I am incredibly happy and joyful today! This is wonderful!",
            "I feel sad and depressed about the current situation.",
            "Fear and anxiety are overwhelming me right now.",
            "I trust that everything will work out fine in the end.",
            "This is neutral technical documentation about APIs."
        ]
        
        # Write to files
        for i, text in enumerate(texts):
            with open(os.path.join(self.temp_dir, f'text_{i}.txt'), 'w') as f:
                f.write(text)
        
        # Load and process
        items = load_text_corpus(self.temp_dir)
        
        self.assertEqual(len(items), 5)
        
        # Analyze affect vectors
        affect_vectors = [item[1] for item in items]
        
        # First text should have high joy
        self.assertGreater(affect_vectors[0][0], 0.3)  # Joy index
        
        # Second text should have high sadness
        self.assertGreater(affect_vectors[1][1], 0.3)  # Sadness index
        
        # Third text should have high fear
        self.assertGreater(affect_vectors[2][3], 0.3)  # Fear index
        
        # Fourth text should have high trust
        self.assertGreater(affect_vectors[3][6], 0.3)  # Trust index
        
        # Fifth text should be mostly neutral (low values)
        self.assertLess(np.max(affect_vectors[4]), 0.3)


if __name__ == '__main__':
    unittest.main()
