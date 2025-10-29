#!/usr/bin/env python3

import asyncio
import json
import logging
import hashlib
import pickle
import os
from urllib.parse import urljoin
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

from aiohttp import ClientSession, ClientTimeout, TCPConnector
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BraveAsyncExtractor:
    """
    Async Brave Search extractor with summarizer capabilities
    Provides fast, comprehensive historical data extraction
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache"):
        self.api_key = api_key or os.getenv('BRAVE_API_KEY')
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        
        # API Configuration
        self.api_max_concurrent_requests = 3
        self.api_rps = 2  # Requests per second
        self.api_rate_limit = AsyncLimiter(self.api_rps, 1)
        self.api_timeout = 20
        
        # Brave Search API endpoints
        self.api_host = "https://api.search.brave.com"
        self.api_paths = {
            "web": urljoin(self.api_host, "res/v1/web/search"),
            "summarizer_search": urljoin(self.api_host, "res/v1/summarizer/search"),
        }
        
        # API Headers
        self.api_headers = {
            "web": {"X-Subscription-Token": self.api_key, "Api-Version": "2023-10-11"},
            "summarizer": {"X-Subscription-Token": self.api_key, "Api-Version": "2024-04-23"},
        }
        
        # Load caches
        self.search_cache_file = os.path.join(cache_dir, "brave_async_cache.pkl")
        self.search_cache = self._load_cache()
        
        self.logger.info(f"Brave Async Extractor initialized with {len(self.search_cache)} cached entries")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file"""
        try:
            if os.path.exists(self.search_cache_file):
                with open(self.search_cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.search_cache_file, 'wb') as f:
                pickle.dump(self.search_cache, f)
        except Exception as e:
            self.logger.error(f"Could not save cache: {e}")
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any], max_age_hours: int = 24) -> bool:
        """Check if cache entry is still valid"""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        age = datetime.now() - cache_time
        return age.total_seconds() < (max_age_hours * 3600)
    
    async def search_with_summary(self, query: str) -> Dict[str, Any]:
        """Search Brave with summarizer for comprehensive results"""
        cache_key = self._get_cache_key(query)
        
        # Check cache first
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                self.logger.info(f"Using cached results for: {query}")
                return cache_entry['result']
        
        async with self.api_rate_limit:
            async with ClientSession(
                connector=TCPConnector(limit=self.api_max_concurrent_requests, ssl=False),
                timeout=ClientTimeout(self.api_timeout),
            ) as session:
                
                # Step 1: Get web search results with summary key
                web_results = await self._get_web_search(session, query)
                if not web_results:
                    return {'error': 'Failed to get web search results'}
                
                # Step 2: Get summary if available
                summary_data = None
                summary_key = web_results.get("summarizer", {}).get("key")
                if summary_key:
                    summary_data = await self._get_summary(session, summary_key)
                
                # Combine results
                result = {
                    'query': query,
                    'web_results': web_results.get('web', {}).get('results', []),
                    'summary': summary_data,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Cache the result
                self.search_cache[cache_key] = {
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_cache()
                
                return result
    
    async def _get_web_search(self, session: ClientSession, query: str) -> Dict[str, Any]:
        """Get web search results from Brave API"""
        params = {
            "q": query,
            "summary": 1,
            "count": 10,
            "safesearch": "moderate"
        }
        
        try:
            async with session.get(
                self.api_paths["web"],
                params=params,
                headers=self.api_headers["web"],
            ) as response:
                self.logger.info(f"Web search query: {query}")
                
                if response.status != 200:
                    self.logger.error(f"Web search failed with status {response.status}")
                    return {}
                
                data = await response.json()
                return data
                
        except Exception as e:
            self.logger.error(f"Error in web search: {e}")
            return {}
    
    async def _get_summary(self, session: ClientSession, summary_key: str) -> Dict[str, Any]:
        """Get summary from Brave summarizer API"""
        params = {
            "key": summary_key,
            "entity_info": 1
        }
        
        try:
            async with session.get(
                self.api_paths["summarizer_search"],
                params=params,
                headers=self.api_headers["summarizer"],
            ) as response:
                self.logger.info(f"Getting summary for key: {summary_key[:20]}...")
                
                if response.status != 200:
                    self.logger.error(f"Summary search failed with status {response.status}")
                    return {}
                
                data = await response.json()
                return data
                
        except Exception as e:
            self.logger.error(f"Error in summary search: {e}")
            return {}
    
    def generate_historical_queries(self, entity_text: str, entity_type: str) -> List[str]:
        """Generate comprehensive historical queries"""
        queries = []
        
        # Base comprehensive query
        queries.append(f"{entity_text} history biography facts")
        
        if entity_type == 'PERSON':
            queries.extend([
                f"{entity_text} birth death years occupation achievements",
                f"{entity_text} historical era period significance",
                f"{entity_text} major contributions impact on history",
                f"{entity_text} contemporaries relationships timeline"
            ])
        elif entity_type == 'GPE':
            queries.extend([
                f"{entity_text} history timeline development",
                f"{entity_text} historical significance importance",
                f"{entity_text} rise fall empire kingdom major events"
            ])
        elif entity_type == 'EVENT':
            queries.extend([
                f"{entity_text} causes consequences historical impact",
                f"{entity_text} historical context significance timeline"
            ])
        
        return queries[:4]  # Limit to 4 focused queries
    
    async def extract_historical_data(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive historical data from text"""
        self.logger.info(f"Extracting historical data from: {text[:50]}...")
        
        # Generate queries for era and event type detection
        era_query = f"what historical era period is this: {text}"
        event_type_query = f"what type of historical event is this: {text}"
        
        # Extract entities (simple pattern-based for now)
        entities = self._extract_entities_simple(text)
        
        # Run searches concurrently
        tasks = [
            self.search_with_summary(era_query),
            self.search_with_summary(event_type_query)
        ]
        
        # Add entity-specific queries
        for entity_text, entity_type in entities[:3]:  # Top 3 entities
            queries = self.generate_historical_queries(entity_text, entity_type)
            for query in queries:
                tasks.append(self.search_with_summary(query))
        
        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        era_data = results[0] if len(results) > 0 else {}
        event_type_data = results[1] if len(results) > 1 else {}
        entity_results = results[2:] if len(results) > 2 else []
        
        # Extract era and event type
        era = self._extract_era_from_results(era_data)
        event_type = self._extract_event_type_from_results(event_type_data)
        
        # Process entity results
        enriched_entities = self._process_entity_results(entities, entity_results)
        
        return {
            'text': text,
            'era': era,
            'event_type': event_type,
            'entities': enriched_entities,
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'extractor': 'brave_async_v1.0',
                'queries_executed': len(tasks),
                'cache_entries': len(self.search_cache)
            }
        }
    
    def _extract_entities_simple(self, text: str) -> List[tuple]:
        """Simple entity extraction (can be enhanced with spaCy later)"""
        entities = []
        
        # Simple patterns for historical entities
        import re
        
        # Person patterns
        person_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+da\s+[A-Z][a-z]+\b',
            r'\b([A-Z][a-z]+)\s+the\s+Great\b',
            r'\b([A-Z][a-z]+)\s+Caesar\b',
            r'\b([A-Z][a-z]+)\s+Bonaparte\b'
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append((match.group(), 'PERSON'))
        
        # Place patterns
        place_patterns = [
            r'\b(Florence|Macedonia|Rome|Athens|Sparta|Egypt|Italy|Greece|Persia|France|India|Europe)\b'
        ]
        
        for pattern in place_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append((match.group(), 'GPE'))
        
        return entities[:5]  # Limit to 5 entities
    
    def _extract_era_from_results(self, era_data: Dict[str, Any]) -> str:
        """Extract historical era from search results"""
        if not era_data or 'error' in era_data:
            return "Unknown Era"
        
        # Check summary first
        summary = era_data.get('summary', {})
        if summary and 'summaries' in summary:
            summary_text = ' '.join([s.get('text', '') for s in summary['summaries']]).lower()
        else:
            summary_text = ""
        
        # Check web results
        web_results = era_data.get('web_results', [])
        web_text = ' '.join([r.get('description', '') for r in web_results]).lower()
        
        # Combine all text
        all_text = f"{summary_text} {web_text}".lower()
        
        # Era keywords
        era_keywords = {
            'Renaissance': ['renaissance', 'italian renaissance'],
            'Classical Antiquity': ['classical antiquity', 'ancient greece', 'ancient rome'],
            'Late Middle Ages': ['late middle ages', '14th century', 'medieval'],
            'Industrial Revolution': ['industrial revolution', '18th century', '19th century'],
            'Enlightenment': ['enlightenment', 'scientific revolution', '17th century'],
            'Modern Era': ['modern era', '20th century', 'contemporary']
        }
        
        era_scores = defaultdict(int)
        for era, keywords in era_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    era_scores[era] += 1
        
        if era_scores:
            return max(era_scores, key=era_scores.get)
        
        return "Unknown Era"
    
    def _extract_event_type_from_results(self, event_data: Dict[str, Any]) -> str:
        """Extract event type from search results"""
        if not event_data or 'error' in event_data:
            return "general"
        
        # Check summary first
        summary = event_data.get('summary', {})
        if summary and 'summaries' in summary:
            summary_text = ' '.join([s.get('text', '') for s in summary['summaries']]).lower()
        else:
            summary_text = ""
        
        # Check web results
        web_results = event_data.get('web_results', [])
        web_text = ' '.join([r.get('description', '') for r in web_results]).lower()
        
        # Combine all text
        all_text = f"{summary_text} {web_text}".lower()
        
        # Event type keywords
        event_keywords = {
            'cultural': ['art', 'painting', 'renaissance', 'cultural', 'literature'],
            'political': ['conquest', 'empire', 'political', 'government', 'kingdom'],
            'disaster': ['plague', 'death', 'disaster', 'famine', 'disease'],
            'discovery': ['invention', 'discovery', 'scientific', 'formulated', 'laws'],
            'war': ['war', 'battle', 'conflict', 'conquest', 'invasion']
        }
        
        event_scores = defaultdict(int)
        for event_type, keywords in event_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    event_scores[event_type] += 1
        
        if event_scores:
            return max(event_scores, key=event_scores.get)
        
        return "general"
    
    def _process_entity_results(self, entities: List[tuple], entity_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Process entity search results"""
        enriched_entities = {
            'PERSON': [],
            'GPE': [],
            'EVENT': []
        }
        
        # Group results by entity
        result_index = 0
        for entity_text, entity_type in entities:
            if result_index < len(entity_results):
                result = entity_results[result_index]
                result_index += 1
                
                enrichment = {
                    'text': entity_text,
                    'type': entity_type,
                    'summary': result.get('summary', {}),
                    'web_results': result.get('web_results', []),
                    'cached': False
                }
                
                enriched_entities[entity_type].append(enrichment)
        
        return enriched_entities

async def test_brave_async_extractor():
    """Test the async Brave extractor"""
    print("🚀 Testing Brave Async Extractor")
    print("=" * 50)
    
    extractor = BraveAsyncExtractor()
    
    if not extractor.api_key:
        print("❌ No Brave API key found. Please set BRAVE_API_KEY in .env file")
        return
    
    # Test cases
    test_cases = [
        "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance.",
        "Alexander the Great conquered the Persian Empire between 334 and 323 BCE.",
        "The Black Death swept through Europe in the 14th century, killing millions."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {text[:50]}...")
        
        result = await extractor.extract_historical_data(text)
        
        print(f"   📊 Era: {result['era']}")
        print(f"   📊 Event Type: {result['event_type']}")
        print(f"   📊 Entities Found: {sum(len(ents) for ents in result['entities'].values())}")
        print(f"   📊 Queries Executed: {result['extraction_metadata']['queries_executed']}")
        
        # Show entity enrichments
        for entity_type, entities in result['entities'].items():
            if entities:
                print(f"   {entity_type}:")
                for entity in entities:
                    web_results = len(entity.get('web_results', []))
                    summary = entity.get('summary', {})
                    has_summary = bool(summary and summary.get('summaries'))
                    print(f"     • {entity['text']} (Web: {web_results}, Summary: {'✅' if has_summary else '❌'})")

if __name__ == "__main__":
    asyncio.run(test_brave_async_extractor())
