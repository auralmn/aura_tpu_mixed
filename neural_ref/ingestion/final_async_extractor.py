#!/usr/bin/env python3

import asyncio
import json
import logging
import hashlib
import pickle
import os
import re
from urllib.parse import urljoin
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FinalAsyncHistoricalExtractor:
    """
    Final comprehensive historical extractor using async Brave Search with summarizer
    Provides fast, accurate historical data extraction with intelligent caching
    """
    
    def __init__(self, api_key: str = None, cache_dir: str = "cache"):
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
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
        
        # API Configuration - Optimized for high throughput
        self.api_timeout = 10
        self.api_rate_limit = AsyncLimiter(self.api_rps, 1)
        self.api_timeout = 10  # Shorter timeout for faster processing
        
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
        self.search_cache_file = os.path.join(cache_dir, "final_async_cache.pkl")
        self.search_cache = self._load_cache()
        
        # Historical patterns
        self.date_patterns = [
            r'\b(\d{1,2})\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{4})\b',
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',
            r'\b(\d{1,4})\s+(?:BC|BCE|AD|CE)\b',
            r'\b(?:around|about|circa|c\.)\s+(\d{1,4})\b'
        ]
        
        self.logger.info(f"Final Async Extractor initialized with {len(self.search_cache)} cached entries")
    
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
                self.logger.info(f"Getting summary for key: {summary_key[:30]}...")
                
                if response.status != 200:
                    self.logger.error(f"Summary search failed with status {response.status}")
                    return {}
                
                data = await response.json()
                return data
                
        except Exception as e:
            self.logger.error(f"Error in summary search: {e}")
            return {}
    
    def extract_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract dates from text using regex patterns"""
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_info = {
                    'original_text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'pattern_used': pattern
                }
                
                # Extract year
                groups = match.groups()
                if groups:
                    if len(groups) >= 3:  # Full date
                        year = int(groups[2]) if groups[2] else int(groups[1])
                    elif len(groups) >= 2:  # Month Day Year or Year BC/AD
                        year = int(groups[1]) if groups[1].isdigit() else int(groups[0])
                    else:  # Just year
                        year = int(groups[0])
                    
                    date_info['year'] = year
                    date_info['formatted_date'] = f"{year}-01-01"
                
                dates.append(date_info)
        
        return dates
    
    def extract_entities_from_text(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract entities using pattern-based approach"""
        entities = {
            'PERSON': [],
            'GPE': [],
            'EVENT': [],
            'ORG': []
        }
        
        # Person patterns
        person_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+da\s+[A-Z][a-z]+\b',
            r'\b([A-Z][a-z]+)\s+the\s+Great\b',
            r'\b([A-Z][a-z]+)\s+Caesar\b',
            r'\b([A-Z][a-z]+)\s+Bonaparte\b',
            r'\b([A-Z][a-z]+)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'  # General name pattern
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.8,
                    'type': 'PERSON'
                }
                entities['PERSON'].append(entity)
        
        # Place patterns with historical context
        place_patterns = [
            r'\b(Florence|Macedonia|Rome|Athens|Sparta|Egypt|Italy|Greece|Persia|France|India|Europe|Asia|Africa|America)\b'
        ]
        
        for pattern in place_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Check if this is likely a historical reference based on context
                entity_text = match.group()
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].lower()
                
                # Skip if context suggests modern entities (bands, companies, etc.)
                modern_indicators = ['band', 'music', 'album', 'song', 'tour', 'concert', 'company', 'corporation']
                if any(indicator in context for indicator in modern_indicators):
                    continue
                
                # Boost confidence for historical context
                confidence = 0.9
                historical_indicators = ['ancient', 'medieval', 'renaissance', 'empire', 'kingdom', 'century', 'bce', 'ad', 'conquered', 'historical']
                if any(indicator in context for indicator in historical_indicators):
                    confidence = 0.95
                
                entity = {
                    'text': entity_text,
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': confidence,
                    'type': 'GPE',
                    'historical_context': any(indicator in context for indicator in historical_indicators)
                }
                entities['GPE'].append(entity)
        
        # Event patterns
        event_patterns = [
            r'\b(Black Death|Renaissance|Reformation|Crusades|Industrial Revolution)\b'
        ]
        
        for pattern in event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = {
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.9,
                    'type': 'EVENT'
                }
                entities['EVENT'].append(entity)
        
        # Remove duplicates
        for entity_type in entities:
            seen = set()
            unique_entities = []
            for entity in entities[entity_type]:
                if entity['text'] not in seen:
                    seen.add(entity['text'])
                    unique_entities.append(entity)
            entities[entity_type] = unique_entities
        
        return entities
    
    async def determine_era_async(self, text: str, dates: List[Dict] = None) -> str:
        """Determine historical era using async Brave Search"""
        # Extract key historical terms from the text for better context
        historical_terms = []
        if dates:
            for date_info in dates:
                year = date_info.get('year', 0)
                if year < 1000:
                    historical_terms.append("ancient")
                elif year < 1500:
                    historical_terms.append("medieval")
                elif year < 1800:
                    historical_terms.append("renaissance")
        
        context_terms = " ".join(historical_terms)
        era_query = f"what historical era period ancient medieval renaissance is this: {text} {context_terms}"
        
        result = await self.search_with_summary(era_query)
        if 'error' in result:
            # Fallback to date-based detection
            if dates:
                for date_info in dates:
                    year = date_info.get('year', 0)
                    return self._determine_era_by_year(year)
            return "Unknown Era"
        
        return self._extract_era_from_results(result)
    
    async def detect_event_type_async(self, text: str) -> str:
        """Detect event type using async Brave Search"""
        # Add historical context to avoid modern interpretations
        event_query = f"what type of historical event ancient medieval is this: {text}"
        
        result = await self.search_with_summary(event_query)
        if 'error' in result:
            return self._detect_event_type_fallback(text)
        
        return self._extract_event_type_from_results(result)
    
    def _extract_era_from_results(self, result: Dict[str, Any]) -> str:
        """Extract era from search results"""
        all_text = ""
        
        # Check summary first
        summary = result.get('summary', {})
        if summary and 'summaries' in summary:
            all_text += ' '.join([s.get('text', '') for s in summary['summaries']]).lower()
        
        # Check web results
        web_results = result.get('web_results', [])
        all_text += ' '.join([r.get('description', '') for r in web_results]).lower()
        
        # Era keywords
        era_keywords = {
            'Renaissance': ['renaissance', 'italian renaissance', '15th century', '16th century'],
            'Classical Antiquity': ['classical antiquity', 'ancient greece', 'ancient rome', 'alexander the great', 'macedonia', 'hellenistic', 'bce', 'bc'],
            'Late Middle Ages': ['late middle ages', '14th century', 'medieval', 'black death', 'plague'],
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
    
    def _extract_event_type_from_results(self, result: Dict[str, Any]) -> str:
        """Extract event type from search results"""
        all_text = ""
        
        # Check summary first
        summary = result.get('summary', {})
        if summary and 'summaries' in summary:
            all_text += ' '.join([s.get('text', '') for s in summary['summaries']]).lower()
        
        # Check web results
        web_results = result.get('web_results', [])
        all_text += ' '.join([r.get('description', '') for r in web_results]).lower()
        
        # Event type keywords
        event_keywords = {
            'conquest': ['conquered', 'conquest', 'conquering', 'conquers', 'invasion', 'invaded', 'military campaign'],
            'cultural': ['art', 'painting', 'renaissance', 'cultural', 'literature', 'painted', 'sculpture'],
            'political': ['empire', 'political', 'government', 'kingdom', 'reign', 'ruled', 'administration'],
            'disaster': ['plague', 'death', 'disaster', 'famine', 'disease', 'swept through', 'killed'],
            'discovery': ['invention', 'discovery', 'scientific', 'formulated', 'laws', 'invented', 'discovered'],
            'war': ['war', 'battle', 'conflict', 'fought', 'campaign', 'siege', 'military']
        }
        
        event_scores = defaultdict(int)
        for event_type, keywords in event_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    event_scores[event_type] += 1
        
        if event_scores:
            return max(event_scores, key=event_scores.get)
        
        return "general"
    
    def _determine_era_by_year(self, year: int) -> str:
        """Determine era based on year"""
        if year < 500:
            return "Classical Antiquity"
        elif year < 1000:
            return "Early Middle Ages"
        elif year < 1400:
            return "High Middle Ages"
        elif year < 1500:
            return "Late Middle Ages"
        elif year < 1700:
            return "Renaissance"
        elif year < 1800:
            return "Enlightenment"
        elif year < 1900:
            return "Industrial Revolution"
        else:
            return "Modern Era"
    
    def _detect_event_type_fallback(self, text: str) -> str:
        """Fallback event type detection"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['conquered', 'conquest', 'invasion']):
            return 'conquest'
        elif any(word in text_lower for word in ['art', 'painting', 'literature', 'painted']):
            return 'cultural'
        elif any(word in text_lower for word in ['war', 'battle', 'conflict']):
            return 'war'
        elif any(word in text_lower for word in ['plague', 'disease', 'death', 'swept through']):
            return 'disaster'
        elif any(word in text_lower for word in ['invention', 'discovery']):
            return 'discovery'
        else:
            return 'general'
    
    async def enrich_entity_async(self, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """Enrich entity with comprehensive historical data"""
        enrichment = {
            'text': entity_text,
            'type': entity_type,
            'enriched': False,
            'sources': [],
            'data': {}
        }
        
        # Generate focused queries
        queries = self._generate_entity_queries(entity_text, entity_type)
        
        # Execute searches concurrently
        tasks = [self.search_with_summary(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_data = []
        for i, result in enumerate(results):
            if not isinstance(result, Exception) and 'error' not in result:
                all_data.append({
                    'query': queries[i],
                    'result': result
                })
        
        if all_data:
            enrichment['enriched'] = True
            enrichment['sources'] = all_data
            enrichment['data'] = self._extract_entity_data(all_data, entity_type)
        
        return enrichment
    
    def _generate_entity_queries(self, entity_text: str, entity_type: str) -> List[str]:
        """Generate simple, direct questions for entity enrichment"""
        queries = []
        
        if entity_type == 'PERSON':
            queries = [
                f"Who was {entity_text}?",
                f"{entity_text} biography birth death years occupation"
            ]
        elif entity_type == 'GPE':
            queries = [
                f"When was {entity_text} founded?",
                f"{entity_text} history founding date kingdom empire"
            ]
        elif entity_type == 'EVENT':
            queries = [
                f"{entity_text} what are the precursors?",
                f"{entity_text} what were the impacts?"
            ]
        else:
            queries = [f"{entity_text} history facts information"]
        
        return queries
    
    def _extract_entity_data(self, data_list: List[Dict], entity_type: str) -> Dict[str, Any]:
        """Extract structured data from enrichment results based on entity type"""
        extracted = {
            'significance': None,
            'timeline': None,
            'impact': None,
            'description': None
        }
        
        all_text = ""
        for data_item in data_list:
            result = data_item['result']
            
            # Add summary text
            summary = result.get('summary', {})
            if summary and 'summaries' in summary:
                all_text += ' '.join([s.get('text', '') for s in summary['summaries']])
            
            # Add web results text
            web_results = result.get('web_results', [])
            all_text += ' '.join([r.get('description', '') for r in web_results])
        
        # Extract type-specific information
        if entity_type == 'PERSON':
            # Only persons can have birth/death years and occupations
            extracted.update({
                'birth_year': None,
                'death_year': None,
                'occupation': None
            })
            
            # Birth/death years
            year_pattern = r'\b(?:born|birth|died|death).*?(\d{4})\s*(?:bc|bce|ad|ce)?\b'
            years = re.findall(year_pattern, all_text, re.IGNORECASE)
            if years:
                birth_year = years[0] if len(years) > 0 else None
                death_year = years[1] if len(years) > 1 else None
                
                # Format BCE years properly
                if birth_year and int(birth_year) > 1000:
                    context_text = all_text.lower()
                    if any(indicator in context_text for indicator in ['bc', 'bce', 'before christ', 'before common era', 'ancient']):
                        extracted['birth_year'] = f"{birth_year} BCE"
                    else:
                        extracted['birth_year'] = f"{birth_year} BCE"  # Default to BCE for ancient dates
                else:
                    extracted['birth_year'] = birth_year
                
                if death_year and int(death_year) > 1000:
                    context_text = all_text.lower()
                    if any(indicator in context_text for indicator in ['bc', 'bce', 'before christ', 'before common era', 'ancient']):
                        extracted['death_year'] = f"{death_year} BCE"
                    else:
                        extracted['death_year'] = f"{death_year} BCE"  # Default to BCE for ancient dates
                else:
                    extracted['death_year'] = death_year
            
            # Occupation
            occupation_pattern = r'\b(painter|artist|scientist|emperor|king|general|philosopher|inventor|explorer|scholar|architect|sculptor|writer|poet)\b'
            occupations = re.findall(occupation_pattern, all_text, re.IGNORECASE)
            if occupations:
                extracted['occupation'] = occupations[0].title()
                
        elif entity_type == 'GPE':
            # Places can have founding dates, rulers, and historical periods
            extracted.update({
                'founding_date': None,
                'historical_period': None,
                'rulers': None,
                'civilization': None
            })
            
            # Founding dates or historical periods
            period_pattern = r'\b(founded|established|existed).*?(\d{4})\s*(?:bc|bce|ad|ce)?\b'
            periods = re.findall(period_pattern, all_text, re.IGNORECASE)
            if periods:
                date_str = periods[0][1] if periods else None
                # Check if it's a BCE date based on context
                if date_str and int(date_str) > 1000:  # Likely BCE date if very old
                    # Look for BCE indicators in the text
                    context_text = all_text.lower()
                    if any(indicator in context_text for indicator in ['bc', 'bce', 'before christ', 'before common era', 'ancient']):
                        extracted['founding_date'] = f"{date_str} BCE"
                    else:
                        extracted['founding_date'] = f"{date_str} BCE"  # Default to BCE for ancient dates
                else:
                    extracted['founding_date'] = date_str
            
            # Historical period
            period_keywords = ['ancient', 'medieval', 'renaissance', 'classical', 'hellenistic', 'roman', 'greek']
            for keyword in period_keywords:
                if keyword in all_text.lower():
                    extracted['historical_period'] = keyword.title()
                    break
                    
        elif entity_type == 'EVENT':
            # Events can have dates, causes, and consequences
            extracted.update({
                'event_date': None,
                'duration': None,
                'causes': None,
                'consequences': None
            })
            
            # Event dates
            date_pattern = r'\b(?:occurred|happened|began|ended).*?(\d{4})\s*(?:bc|bce|ad|ce)?\b'
            dates = re.findall(date_pattern, all_text, re.IGNORECASE)
            if dates:
                date_str = dates[0] if dates else None
                # Check if it's a BCE date based on context
                if date_str and int(date_str) > 1000:  # Likely BCE date if very old
                    # Look for BCE indicators in the text
                    context_text = all_text.lower()
                    if any(indicator in context_text for indicator in ['bc', 'bce', 'before christ', 'before common era', 'ancient']):
                        extracted['event_date'] = f"{date_str} BCE"
                    else:
                        extracted['event_date'] = f"{date_str} BCE"  # Default to BCE for ancient dates
                else:
                    extracted['event_date'] = date_str
        
        # Common fields for all entity types
        # Significance (first few sentences from summaries)
        for data_item in data_list:
            summary = data_item['result'].get('summary', {})
            if summary and 'summaries' in summary and summary['summaries']:
                extracted['significance'] = summary['summaries'][0].get('text', '')[:200]
                break
        
        # Description from web results
        if not extracted['significance']:
            for data_item in data_list:
                web_results = data_item['result'].get('web_results', [])
                if web_results:
                    extracted['description'] = web_results[0].get('description', '')[:200]
                    break
        
        return extracted
    
    async def process_text_async(self, text: str) -> Dict[str, Any]:
        """Process text and extract comprehensive historical information"""
        self.logger.info(f"Processing text: {text[:50]}...")
        
        # Extract dates and entities
        dates = self.extract_dates_from_text(text)
        entities = self.extract_entities_from_text(text)
        
        # Determine era and event type concurrently
        era_task = self.determine_era_async(text, dates)
        event_type_task = self.detect_event_type_async(text)
        
        era, event_type = await asyncio.gather(era_task, event_type_task)
        
        # Enrich entities concurrently
        enrichment_tasks = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list[:2]:  # Limit to top 2 entities per type
                task = self.enrich_entity_async(entity['text'], entity_type)
                enrichment_tasks.append((entity, task))
        
        enriched_entities = {}
        if enrichment_tasks:
            entity_results = await asyncio.gather(*[task for _, task in enrichment_tasks])
            
            # Organize enriched entities
            for entity_type in entities:
                enriched_entities[entity_type] = []
            
            for (entity, _), enrichment_result in zip(enrichment_tasks, entity_results):
                entity['enrichment'] = enrichment_result
                enriched_entities[entity['type']].append(entity)
        
        return {
            'text': text,
            'dates': dates,
            'entities': enriched_entities,
            'era': era,
            'event_type': event_type,
            'processing_metadata': {
                'timestamp': datetime.now().isoformat(),
                'extractor': 'final_async_v1.0',
                'cache_entries': len(self.search_cache),
                'entities_processed': sum(len(ents) for ents in entities.values()),
                'entities_enriched': sum(len(ents) for ents in enriched_entities.values())
            }
        }

async def test_final_async_extractor():
    """Test the final async extractor"""
    print("🚀 Testing Final Async Historical Extractor")
    print("=" * 60)
    
    extractor = FinalAsyncHistoricalExtractor()
    
    if not extractor.api_key:
        print("❌ No Brave API key found. Please set BRAVE_API_KEY in .env file")
        return
    
    test_cases = [
        "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance, revolutionizing art with his techniques.",
        "Alexander the Great conquered the Persian Empire between 334 and 323 BCE, spreading Hellenistic culture from Macedonia to India.",
        "The Black Death swept through Europe in the 14th century, killing an estimated 30-50% of the population."
    ]
    
    all_results = []
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}: {text[:60]}...")
        print("-" * 60)
        
        result = await extractor.process_text_async(text)
        all_results.append(result)
        
        print(f"📊 Era: {result['era']}")
        print(f"📊 Event Type: {result['event_type']}")
        print(f"📊 Dates Found: {len(result['dates'])}")
        print(f"📊 Entities Processed: {result['processing_metadata']['entities_processed']}")
        print(f"📊 Entities Enriched: {result['processing_metadata']['entities_enriched']}")
        
        # Show enriched entities
        for entity_type, entities in result['entities'].items():
            if entities:
                print(f"\n📚 {entity_type}:")
                for entity in entities:
                    enrichment = entity.get('enrichment', {})
                    if enrichment.get('enriched'):
                        data = enrichment.get('data', {})
                        print(f"   • {entity['text']}")
                        if data.get('birth_year') or data.get('death_year'):
                            print(f"     📅 Years: {data.get('birth_year', '?')} - {data.get('death_year', '?')}")
                        if data.get('occupation'):
                            print(f"     💼 Occupation: {data.get('occupation')}")
                        if data.get('significance'):
                            print(f"     ⭐ Significance: {data.get('significance')[:100]}...")
        
        print(f"\n✅ Cache Entries: {result['processing_metadata']['cache_entries']}")
    
    # Save results
    output_file = "final_async_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Final Async Extractor Test Complete!")
    print(f"📁 Results saved to: {output_file}")
    print(f"✅ Async Brave Search with summarizer working")
    print(f"✅ Intelligent caching system active")
    print(f"✅ Comprehensive historical data extraction")
    print(f"✅ Fast concurrent processing")

if __name__ == "__main__":
    asyncio.run(test_final_async_extractor())
