#!/usr/bin/env python3

import asyncio
import json
import logging
import hashlib
import pickle
import os
import re
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dotenv import load_dotenv
from .query_templates import BraveSearchQueryTemplates

# Flair imports
try:
    from flair.data import Sentence
    from flair.models import SequenceTagger
    FLAIR_AVAILABLE = True
except ImportError:
    FLAIR_AVAILABLE = False
    print("Flair not available. Please install with: pip install flair")

# Load environment variables
load_dotenv()

class SyncBraveHistoricalExtractor:
    """
    Synchronous historical extractor using regular Brave Search API
    Provides fast, accurate historical data extraction with intelligent caching
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache", response_limits: Dict[str, int] = None):
        # Clean and validate API key
        raw_api_key = api_key or os.getenv('BRAVE_API_KEY')
        if raw_api_key:
            # Remove any whitespace and clean the key
            self.api_key = raw_api_key.strip()
            if not self.api_key.startswith('BSA'):
                self.logger.warning("API key doesn't start with 'BSA' - please verify it's correct")
        else:
            self.api_key = None
            self.logger.error("No Brave API key provided!")
        
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Response length limits (in words)
        self.response_limits = response_limits or {
            # Basic fields
            'eventName': 10,
            'description': 50,
            'summary': 20,
            
            # Location fields
            'eventLocation': 10,
            'geopoliticalContext': 30,
            
            # Lists (per item)
            'precursors': 15,
            'outcomes': 15,
            'consequences': 15,
            'keyFigures_description': 10,
            
            # Impact fields
            'culturalImpact': 30,
            'economicImpact': 30,
            'socialImpact': 30,
            'environmentalImpact': 20,
            
            # Research fields
            'significance': 40,
            'modernEquivalent': 20,
            'historicalPattern': 30,
            'historiographicalDebates': 40,
            'lessonsFuture': 30,
            
            # Future predictions
            'futurePredictions': 10,  # per prediction
            
            # Sources
            'source_snippet': 20  # per source
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        
        # Load Flair NER model
        if FLAIR_AVAILABLE:
            try:
                self.logger.info("Loading Flair NER model...")
                self.ner_tagger = SequenceTagger.load("flair/ner-english")
                self.logger.info("Loaded Flair NER model: flair/ner-english")
            except Exception as e:
                self.logger.error(f"Failed to load Flair NER model: {e}")
                self.ner_tagger = None
        else:
            self.logger.error("Flair not available. Please install with: pip install flair")
            self.ner_tagger = None
        
        # Initialize query templates
        self.query_templates = BraveSearchQueryTemplates()
        
        # API Configuration
        self.api_timeout = 10
        self.api_host = "https://api.search.brave.com"
        
        # Cache files
        self.search_cache_file = os.path.join(self.cache_dir, "search_cache.pkl")
        self.enrichment_cache_file = os.path.join(self.cache_dir, "enrichment_cache.pkl")
        
        # Initialize caches
        self.search_cache = {}
        self.enrichment_cache = {}
        
        # Load existing caches
        self._load_caches()
    
    def _save_cache(self, cache_file: str, cache_data: dict):
        """Save specific cache to disk"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.logger.error(f"Error saving cache to {cache_file}: {e}")
    
    def truncate_text(self, text: str, field_name: str, add_ellipsis: bool = True) -> str:
        """Truncate text to specified word limits for each field"""
        if not text:
            return ""
        
        word_limit = self.response_limits.get(field_name, 20)  # Default 20 words
        
        # Split into words
        words = text.split()
        
        if len(words) <= word_limit:
            return text
        
        # Truncate to word limit
        truncated_words = words[:word_limit]
        
        # Try to end at sentence boundary if possible
        truncated_text = ' '.join(truncated_words)
        
        # Look for last complete sentence
        last_period = truncated_text.rfind('.')
        if last_period > len(truncated_text) * 0.7:  # If period is reasonably close to end
            truncated_text = truncated_text[:last_period + 1]
        else:
            # Just use the word limit
            truncated_text = ' '.join(truncated_words)
        
        if add_ellipsis and len(words) > word_limit:
            truncated_text += "..."
        
        return truncated_text
    
    def _load_caches(self):
        """Load caches from disk"""
        try:
            if os.path.exists(self.search_cache_file):
                with open(self.search_cache_file, 'rb') as f:
                    self.search_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.search_cache)} search cache entries")
            
            if os.path.exists(self.enrichment_cache_file):
                with open(self.enrichment_cache_file, 'rb') as f:
                    self.enrichment_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.enrichment_cache)} enrichment cache entries")
                
        except Exception as e:
            self.logger.warning(f"Error loading caches: {e}")
            self.search_cache = {}
            self.enrichment_cache = {}
    
    def _save_caches(self):
        """Save caches to disk"""
        try:
            with open(self.search_cache_file, 'wb') as f:
                pickle.dump(self.search_cache, f)
            with open(self.enrichment_cache_file, 'wb') as f:
                pickle.dump(self.enrichment_cache, f)
        except Exception as e:
            self.logger.warning(f"Error saving caches: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: dict) -> bool:
        """Check if cache entry is still valid (24 hours)"""
        try:
            timestamp = datetime.fromisoformat(cache_entry['timestamp'])
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            return age_hours < 24
        except:
            return False
        
        self.logger.info(f"Sync Brave Extractor initialized with {len(self.search_cache)} cached entries")
    
    def search_brave_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search Brave web API for comprehensive results"""
        # Validate query length (max 400 chars, 50 words)
        words = query.split()
        if len(words) > 50:
            query = ' '.join(words[:50])
            self.logger.warning(f"Query truncated to 50 words: {query}")
        
        if len(query) > 400:
            query = query[:400]
            self.logger.warning(f"Query truncated to 400 chars: {query}")
        
        cache_key = self._get_cache_key(query)
        
        # Check cache first
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.info(f"Using cached results for: {query}")
                return cache_entry['result']
        
        try:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.api_key
            }
            
            params = {
                'q': query,
                'count': max_results,
                'safesearch': 'moderate'
            }
            
            response = requests.get(
                f"{self.api_host}/res/v1/web/search",
                headers=headers,
                params=params,
                timeout=self.api_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                for result in data.get('web', {}).get('results', [])[:max_results]:
                    # Apply length limits to descriptions
                    description = result.get('description', '')
                    truncated_desc = self.truncate_text(description, 'source_snippet')
                    
                    results.append({
                        'title': self.truncate_text(result.get('title', ''), 'eventName'),
                        'description': truncated_desc,
                        'url': result.get('url', ''),
                        'snippet': truncated_desc,
                        'source': 'brave_search',
                        'full_text': truncated_desc
                    })
                
                # Cache the result
                self.search_cache[cache_key] = {
                    'result': results,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_caches()
                
                self.logger.info(f"Found {len(results)} Brave Search results for: {query}")
                return results
            else:
                self.logger.error(f"Brave Search API error: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching Brave: {e}")
            return []
    
    def determine_era_sync(self, text: str, dates: List[Dict] = None) -> str:
        """Determine historical era using Brave Search"""
        # Create very short era detection query (max 50 words, 400 chars)
        # Extract key historical terms from text
        words = text.split()[:10]  # Take first 10 words max
        era_query = f"historical era {' '.join(words)}"
        
        # Check cache first
        cache_key = self._get_cache_key(f"era_{era_query}")
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['era_result']
        
        try:
            results = self.search_brave_web(era_query, max_results=3)
            
            # Extract era from search results
            era_info = self._extract_era_from_results(results, text, dates)
            
            # Cache the result
            self.search_cache[cache_key] = {
                'era_result': era_info,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(self.search_cache_file, self.search_cache)
            
            return era_info
            
        except Exception as e:
            self.logger.error(f"Error determining era via Brave: {e}")
        
        # Fallback to simple year-based detection
        if dates:
            for date_info in dates:
                year = date_info.get('year', 0)
                return self._determine_era_by_year(year)
        
        return "Unknown Era"
    
    def detect_event_type_sync(self, text: str) -> str:
        """Detect the type of historical event using Brave Search"""
        # Create very short event type detection query (max 50 words, 400 chars)
        words = text.split()[:10]  # Take first 10 words max
        event_query = f"event type {' '.join(words)}"
        
        # Check cache first
        cache_key = self._get_cache_key(f"event_type_{event_query}")
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['event_type_result']
        
        try:
            results = self.search_brave_web(event_query, max_results=3)
            
            # Extract event type from search results
            event_type = self._extract_event_type_from_results(results, text)
            
            # Cache the result
            self.search_cache[cache_key] = {
                'event_type_result': event_type,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache(self.search_cache_file, self.search_cache)
            
            return event_type
            
        except Exception as e:
            self.logger.error(f"Error determining event type via Brave: {e}")
        
        # Fallback to pattern-based detection
        return self._detect_event_type_fallback(text)
    
    def _extract_era_from_results(self, results: List[Dict], text: str, dates: List[Dict] = None) -> str:
        """Extract era information from search results"""
        era_keywords = {
            'Ancient': ['ancient', 'antiquity', 'classical', 'greek', 'roman', 'egyptian', 'mesopotamian', 'bce', 'bc', 'before christ'],
            'Medieval': ['medieval', 'middle ages', 'dark ages', 'feudal', 'crusades', 'viking', 'gothic', 'byzantine'],
            'Renaissance': ['renaissance', 'reformation', 'humanism', 'michaelangelo', 'leonardo', 'da vinci'],
            'Early Modern': ['early modern', 'age of discovery', 'columbus', 'exploration', 'colonial', '16th century', '17th century'],
            'Modern': ['modern', 'industrial revolution', '19th century', '20th century', 'contemporary'],
            'Classical Antiquity': ['alexander the great', 'macedonia', 'hellenistic', 'bce', 'bc']
        }
        
        # Score each era based on keyword matches
        era_scores = defaultdict(int)
        
        for result in results:
            content = f"{result.get('title', '')} {result.get('description', '')}".lower()
            
            for era, keywords in era_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        era_scores[era] += 1
        
        # Return the era with the highest score
        if era_scores:
            return max(era_scores.items(), key=lambda x: x[1])[0]
        
        return "Unknown Era"
    
    def _extract_event_type_from_results(self, results: List[Dict], text: str) -> str:
        """Extract event type from search results"""
        event_keywords = {
            'war': ['war', 'battle', 'conflict', 'military', 'army', 'navy', 'soldier', 'combat'],
            'conquest': ['conquered', 'conquest', 'invasion', 'military campaign', 'conqueror'],
            'political': ['political', 'government', 'politics', 'election', 'republic', 'democracy', 'monarchy'],
            'cultural': ['cultural', 'art', 'literature', 'philosophy', 'religion', 'tradition'],
            'economic': ['economic', 'trade', 'commerce', 'economy', 'financial', 'business'],
            'disaster': ['disaster', 'catastrophe', 'plague', 'famine', 'earthquake', 'flood'],
            'discovery': ['discovery', 'exploration', 'invention', 'scientific', 'technology']
        }
        
        # Score each event type based on keyword matches
        event_scores = defaultdict(int)
        
        for result in results:
            content = f"{result.get('title', '')} {result.get('description', '')}".lower()
            
            for event_type, keywords in event_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        event_scores[event_type] += 1
        
        # Return the event type with the highest score
        if event_scores:
            return max(event_scores.items(), key=lambda x: x[1])[0]
        
        return self._detect_event_type_fallback(text)
    
    def _detect_event_type_fallback(self, text: str) -> str:
        """Fallback event type detection using text patterns"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['war', 'battle', 'conflict', 'military']):
            return 'war'
        elif any(word in text_lower for word in ['conquered', 'conquest', 'invasion']):
            return 'conquest'
        elif any(word in text_lower for word in ['political', 'government', 'election']):
            return 'political'
        elif any(word in text_lower for word in ['cultural', 'art', 'literature']):
            return 'cultural'
        elif any(word in text_lower for word in ['economic', 'trade', 'commerce']):
            return 'economic'
        elif any(word in text_lower for word in ['disaster', 'plague', 'famine']):
            return 'disaster'
        elif any(word in text_lower for word in ['discovery', 'exploration', 'invention']):
            return 'discovery'
        else:
            return 'unknown'
    
    def _determine_era_by_year(self, year: int) -> str:
        """Determine era based on year"""
        if year < 500:
            return "Ancient"
        elif year < 1500:
            return "Medieval"
        elif year < 1800:
            return "Early Modern"
        else:
            return "Modern"
    
    def _merge_adjacent_entities(self, raw_entities: List[Dict], text: str) -> List[Dict]:
        """Merge adjacent entities of the same type into single entities"""
        if not raw_entities:
            return []
        
        # Sort entities by start position
        sorted_entities = sorted(raw_entities, key=lambda x: x['start'])
        merged = []
        
        i = 0
        while i < len(sorted_entities):
            current = sorted_entities[i]
            merged_text = current['text']
            merged_confidence = current['confidence']
            start_pos = current['start']
            end_pos = current['end']
            
            # Look for adjacent entities of the same type
            j = i + 1
            while j < len(sorted_entities):
                next_entity = sorted_entities[j]
                
                # Check if they're adjacent and same type
                if (next_entity['label'] == current['label'] and 
                    next_entity['start'] <= end_pos + 10):  # Allow larger gap for "the", "of", etc.
                    
                    # Check if the gap contains connecting words
                    gap_text = text[end_pos:next_entity['start']].strip().lower()
                    connecting_words = {'the', 'of', 'in', 'from', 'to', 'and', 'de', 'la', 'le', 'von', 'van'}
                    
                    # Only merge if gap is small or contains connecting words
                    if len(gap_text) <= 3 or any(word in gap_text for word in connecting_words):
                        # Merge the text by taking the substring from original text
                        merged_text = text[start_pos:next_entity['end']].strip()
                        merged_confidence = max(merged_confidence, next_entity['confidence'])
                        end_pos = next_entity['end']
                        j += 1
                    else:
                        break
                else:
                    break
            
            # Add merged entity
            merged.append({
                'text': merged_text,
                'label': current['label'],
                'confidence': merged_confidence,
                'start': start_pos,
                'end': end_pos
            })
            
            i = j
        
        return merged

    def extract_entities_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities from text using Flair NER model"""
        entities = {
            'PERSON': [],
            'GPE': [],
            'EVENT': [],
            'ORG': [],
            'DATE': []
        }
        
        if not self.ner_tagger:
            self.logger.warning("Flair NER tagger not available, returning empty entities")
            return entities
        
        try:
            # Create sentence object and predict NER tags
            sentence = Sentence(text)
            self.ner_tagger.predict(sentence)
            
            # Extract entities from Flair spans
            for span in sentence.get_spans('ner'):
                entity_text = span.text
                entity_label = span.get_label('ner').value
                confidence = span.get_label('ner').score
                start = span.start_position
                end = span.end_position
                
                # Skip very short entities
                if len(entity_text) < 2:
                    continue
                    
                # Skip common non-historical words
                skip_words = {
                    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                    'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
                    'above', 'below', 'between', 'among', 'under', 'over', 'around', 'near',
                    'this', 'that', 'these', 'those', 'some', 'any', 'all', 'each', 'every',
                    'many', 'much', 'few', 'little', 'more', 'most', 'less', 'least',
                    'first', 'last', 'next', 'previous', 'other', 'another', 'such', 'same'
                }
                
                if entity_text.lower() in skip_words:
                    continue
                
                # Map Flair labels to our categories
                if entity_label == "PER":  # Person
                    entities['PERSON'].append({
                        'text': entity_text,
                        'start': start,
                        'end': end,
                        'confidence': confidence
                    })
                elif entity_label == "LOC":  # Location (Geopolitical entities)
                    entities['GPE'].append({
                        'text': entity_text,
                        'start': start,
                        'end': end,
                        'confidence': confidence
                    })
                elif entity_label == "ORG":  # Organization
                    entities['ORG'].append({
                        'text': entity_text,
                        'start': start,
                        'end': end,
                        'confidence': confidence
                    })
                elif entity_label == "MISC":  # Miscellaneous - could be events
                    # Check if it looks like a historical event
                    if any(keyword in entity_text.lower() for keyword in 
                           ['war', 'battle', 'treaty', 'revolution', 'conquest', 'rebellion']):
                        entities['EVENT'].append({
                            'text': entity_text,
                            'start': start,
                            'end': end,
                            'confidence': confidence
                        })
                    else:
                        entities['ORG'].append({
                            'text': entity_text,
                            'start': start,
                            'end': end,
                            'confidence': confidence
                        })
            
        except Exception as e:
            self.logger.error(f"Error in Flair NER extraction: {e}")
        
        # Add some basic event detection using patterns (for things BERT might miss)
        event_patterns = [
            r'\b(?:World War|Civil War|Revolutionary War|War of|Battle of|Siege of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b(?:Treaty of|Declaration of|Convention of|Act of)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            r'\b(?:Renaissance|Reformation|Enlightenment|Industrial Revolution|Great Depression|Cold War)\b'
        ]
        
        found_events = set()
        for pattern in event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                event_text = match.group().strip()
                if event_text not in found_events and len(event_text) > 3:
                    found_events.add(event_text)
                    entities['EVENT'].append({
                        'text': event_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8
                    })
        
        # Extract dates using patterns (BERT doesn't handle dates well)
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{4}\b',
            r'\b\d{4}\b',
            r'\b(?:c\.|circa|around)\s+\d{4}\b',
            r'\b\d{4}\s*(?:BCE?|AD|CE)\b',
            r'\b(?:early|mid|late)\s+(?:century|C)\b'
        ]
        
        found_dates = set()
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group().strip()
                if date_text not in found_dates:
                    found_dates.add(date_text)
                    entities['DATE'].append({
                        'text': date_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9
                    })
        
        return entities
    
    def _generate_entity_queries(self, entity_text: str, entity_type: str) -> List[str]:
        """No enrichment - return empty list to skip all searches"""
        return []  # Skip all entity enrichment
    
    def enrich_entity_sync(self, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """Enrich entity using Brave Search"""
        cache_key = self._get_cache_key(f"enrich_{entity_type}_{entity_text}")
        
        # Check cache first
        if cache_key in self.enrichment_cache:
            cache_entry = self.enrichment_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['result']
        
        queries = self._generate_entity_queries(entity_text, entity_type)
        
        # Skip enrichment if no queries (for common entities)
        if not queries:
            return {
                'entity': entity_text,
                'entity_type': entity_type,
                'sources': [],
                'data': {
                    'significance': None,
                    'description': None,
                    'birth_year': None,
                    'death_year': None,
                    'occupation': None,
                    'founding_date': None,
                    'historical_period': None,
                    'rulers': None,
                    'civilization': None
                }
            }
        
        # Use only first query and limit results
        all_data = []
        results = self.search_brave_web(queries[0], max_results=2)
        all_data.extend(results)
        
        enrichment = {
            'entity': entity_text,
            'entity_type': entity_type,
            'sources': all_data,
            'data': self._extract_entity_data(all_data, entity_type)
        }
        
        # Cache the result
        self.enrichment_cache[cache_key] = {
            'result': enrichment,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache(self.enrichment_cache_file, self.enrichment_cache)
        
        return enrichment
    
    def _extract_entity_data(self, data_list: List[Dict], entity_type: str) -> Dict[str, Any]:
        """Extract structured data from enrichment results based on entity type"""
        extracted = {
            'significance': None,
            'description': None
        }
        
        # Extract based on entity type
        if entity_type == 'PERSON':
            extracted.update({
                'birth_year': None,
                'death_year': None,
                'occupation': None
            })
        elif entity_type == 'GPE':
            extracted.update({
                'founding_date': None,
                'historical_period': None,
                'rulers': None,
                'civilization': None
            })
        elif entity_type == 'EVENT':
            extracted.update({
                'event_date': None,
                'duration': None,
                'causes': None,
                'consequences': None
            })
        
        return extracted
    
    async def process_text_async(self, text: str) -> Dict[str, Any]:
        """Process text synchronously (wrapper for compatibility)"""
        return self.process_text_sync(text)
    
    def process_text_sync(self, text: str) -> Dict[str, Any]:
        """Process text synchronously"""
        self.logger.info(f"Processing text: {text[:100]}...")
        
        # Extract basic entities
        entities = self.extract_entities_from_text(text)
        
        # Determine era and event type
        era = self.determine_era_sync(text)
        event_type = self.detect_event_type_sync(text)
        
        # Enrich entities
        enriched_entities = {}
        for entity_type, entity_list in entities.items():
            if entity_list:
                enriched_entities[entity_type] = []
                for entity in entity_list:
                    enriched = self.enrich_entity_sync(entity['text'], entity_type)
                    enriched_entities[entity_type].append(enriched)
        
        return {
            'text': text,
            'era': era,
            'event_type': event_type,
            'entities': enriched_entities,
            'processing_metadata': {
                'entities_enriched': sum(len(ents) for ents in enriched_entities.values()),
                'timestamp': datetime.now().isoformat()
            }
        }
