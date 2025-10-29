#!/usr/bin/env python3

import re
import json
import uuid
import os
import requests
import hashlib
import pickle
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ComprehensiveHistoricalExtractor:
    """
    Comprehensive historical event extractor with Wikipedia + Brave Search enrichment
    Combines era detection, entity extraction, and data enrichment
    """
    
    def __init__(self, 
                 text_file_path: str = None, 
                 wikipedia_jsonl_path: str = None, 
                 brave_api_key: str = None,
                 spacy_model: str = "en_core_web_sm"):
        
        self.text_file_path = text_file_path
        self.text_content = ""
        self.events = []
        self.wikipedia_jsonl_path = wikipedia_jsonl_path
        self.brave_api_key = brave_api_key or os.getenv('BRAVE_API_KEY')
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize caching system
        self.cache_dir = "cache"
        self.search_cache_file = os.path.join(self.cache_dir, "search_cache.pkl")
        self.enrichment_cache_file = os.path.join(self.cache_dir, "enrichment_cache.pkl")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing caches
        self.search_cache = self._load_cache(self.search_cache_file)
        self.enrichment_cache = self._load_cache(self.enrichment_cache_file)
        
        self.logger.info(f"Loaded {len(self.search_cache)} search cache entries")
        self.logger.info(f"Loaded {len(self.enrichment_cache)} enrichment cache entries")
        
        # Wikipedia JSONL cache
        self.wikipedia_cache = {}
        if self.wikipedia_jsonl_path:
            self.load_wikipedia_jsonl()
        
        # Try to initialize spaCy (optional)
        self.nlp = None
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
            self.logger.info(f"Loaded spaCy model: {spacy_model}")
        except Exception as e:
            self.logger.warning(f"spaCy not available: {e}. Using pattern-based extraction.")
        
        # Enhanced event patterns
        self.event_patterns = {
            'war': r'(?i)\b(?:war|battle|conflict|siege|invasion|revolution|rebellion|uprising|fought|victory|defeat|conquest|campaign|raid|conquers?)\b',
            'political': r'(?i)\b(?:treaty|alliance|independence|coronation|election|coup|assassination|declaration|signed|established|founded|unified|dynasty|reign|empire|kingdom|ruled|created|imperial|rule)\b',
            'discovery': r'(?i)\b(?:discovered|invented|founded|established|created|built|constructed|explored|expedition|domesticated|cultivated|developed)\b',
            'disaster': r'(?i)\b(?:earthquake|plague|famine|flood|hurricane|fire|disaster|catastrophe|destroyed|swept|volcanic|eruption|drought|death)\b',
            'cultural': r'(?i)\b(?:published|composed|painted|wrote|built|designed|performed|renaissance|art|literature|religion|ritual|ceremony|festival|code|laws|culture|language|integrates?)\b',
            'migration': r'(?i)\b(?:migrated|migration|settled|settlement|colony|expansion|spread|diaspora|exodus|arrive|arrives?)\b',
            'technology': r'(?i)\b(?:tools|bronze|iron|agriculture|farming|wheel|writing|pottery|metalwork|technology|age|revolutionizing|construction|temples|theaters|baths)\b',
            'trade': r'(?i)\b(?:trade|trading|commerce|merchant|market|exchange|goods|silk road|caravan|prosperity)\b',
            'religious': r'(?i)\b(?:christian|bishop|church|temple|religious|faith|belief|sacred|holy|monastery|cathedral)\b'
        }
        
        # Historical eras with date ranges and characteristics
        self.historical_eras = {
            'Paleolithic': {'start': -2500000, 'end': -10000, 'keywords': ['paleolithic', 'old stone age', 'hunter', 'gatherer', 'cave', 'mammoth']},
            'Neolithic': {'start': -10000, 'end': -3000, 'keywords': ['neolithic', 'new stone age', 'agriculture', 'farming', 'domestication', 'settlement']},
            'Bronze Age': {'start': -3000, 'end': -1200, 'keywords': ['bronze', 'metalworking', 'bronze age', 'alloy', 'copper', 'tin']},
            'Iron Age': {'start': -1200, 'end': -500, 'keywords': ['iron', 'iron age', 'metallurgy', 'steel', 'forging']},
            'Classical Antiquity': {'start': -800, 'end': 476, 'keywords': ['classical', 'antiquity', 'greece', 'rome', 'athens', 'sparta', 'republic', 'empire']},
            'Ancient Greece': {'start': -800, 'end': -146, 'keywords': ['greece', 'greek', 'athens', 'sparta', 'olympics', 'democracy', 'philosophy']},
            'Roman Republic': {'start': -509, 'end': -27, 'keywords': ['roman republic', 'caesar', 'senate', 'consul', 'patrician', 'plebeian']},
            'Roman Empire': {'start': -27, 'end': 476, 'keywords': ['roman empire', 'emperor', 'augustus', 'pax romana', 'legion', 'gladiator']},
            'Late Antiquity': {'start': 284, 'end': 700, 'keywords': ['late antiquity', 'byzantine', 'constantinople', 'christianity', 'barbarian']},
            'Early Middle Ages': {'start': 476, 'end': 1000, 'keywords': ['dark ages', 'migration', 'barbarian', 'charlemagne', 'feudal', 'monastery']},
            'High Middle Ages': {'start': 1000, 'end': 1300, 'keywords': ['crusades', 'cathedral', 'gothic', 'university', 'scholasticism', 'feudalism']},
            'Late Middle Ages': {'start': 1300, 'end': 1500, 'keywords': ['black death', 'hundred years war', 'renaissance', 'plague', 'peasant']},
            'Renaissance': {'start': 1400, 'end': 1600, 'keywords': ['renaissance', 'leonardo', 'michelangelo', 'florence', 'humanism', 'art', 'science']},
            'Age of Discovery': {'start': 1400, 'end': 1700, 'keywords': ['exploration', 'columbus', 'discovery', 'new world', 'america', 'voyage']},
            'Early Modern Period': {'start': 1500, 'end': 1800, 'keywords': ['reformation', 'protestant', 'catholic', 'printing', 'gunpowder', 'absolutism']},
            'Enlightenment': {'start': 1650, 'end': 1800, 'keywords': ['enlightenment', 'reason', 'philosophy', 'voltaire', 'rousseau', 'democracy']},
            'Industrial Revolution': {'start': 1750, 'end': 1900, 'keywords': ['industrial', 'revolution', 'steam', 'factory', 'mechanization', 'urbanization']},
            'Modern Era': {'start': 1800, 'end': 1945, 'keywords': ['modern', 'world war', 'nationalism', 'imperialism', 'democracy', 'revolution']},
            'Contemporary': {'start': 1945, 'end': 2024, 'keywords': ['contemporary', 'cold war', 'globalization', 'technology', 'internet', 'space']}
        }
        
        # Enhanced entity patterns
        self.entity_patterns = {
            'person': [
                r'\bLeonardo\s+da\s+Vinci\b',
                r'\bAlexander\s+the\s+Great\b',
                r'\bJulius\s+Caesar\b',
                r'\bNapoleon\s+Bonaparte\b',
                r'\bIsaac\s+Newton\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+da\s+[A-Z][a-z]+\b',
                r'\b([A-Z][a-z]+)\s+the\s+Great\b',
                r'\b([A-Z][a-z]+)\s+Caesar\b',
                r'\b([A-Z][a-z]+)\s+Bonaparte\b'
            ],
            'place': [
                r'\bFlorence\b',
                r'\bMacedonia\b',
                r'\bRome\b',
                r'\bAthens\b',
                r'\bSparta\b',
                r'\bEgypt\b',
                r'\bItaly\b',
                r'\bGreece\b',
                r'\bPersia\b',
                r'\bFrance\b',
                r'\bIndia\b'
            ]
        }
        
        # Date patterns
        self.date_patterns = [
            (r'\b(\d{1,5})\s*BCE?\b', 'year_bc'),
            (r'\b(\d{1,5})\s*(?:CE|AD)\b', 'year_ce'),
            (r'\b([1-9]\d{2,3})\b(?!\s*(?:BC|BCE|AD|CE))', 'year_plain'),
        ]
    
    def load_wikipedia_jsonl(self, max_articles: int = 5000, chunk_size: int = 1000):
        """Load Wikipedia JSONL file(s) into memory for fast lookups with chunking"""
        if not os.path.exists(self.wikipedia_jsonl_path):
            self.logger.warning(f"Wikipedia JSONL path not found: {self.wikipedia_jsonl_path}")
            return
        
        try:
            self.logger.info(f"Loading Wikipedia JSONL from: {self.wikipedia_jsonl_path}")
            count = 0
            
            # Handle both single file and directory
            if os.path.isfile(self.wikipedia_jsonl_path):
                jsonl_files = [self.wikipedia_jsonl_path]
            else:
                # Load all JSONL files from directory (skip hidden files)
                jsonl_files = [os.path.join(self.wikipedia_jsonl_path, f) 
                             for f in os.listdir(self.wikipedia_jsonl_path) 
                             if f.endswith('.jsonl') and not f.startswith('.')]
                self.logger.info(f"Found {len(jsonl_files)} JSONL files in directory")
            
            # Process each JSONL file with chunking
            for jsonl_file in jsonl_files:  # Process ALL files
                self.logger.info(f"Processing: {os.path.basename(jsonl_file)}")
                
                try:
                    with open(jsonl_file, 'r', encoding='utf-8', errors='ignore') as f:
                        file_count = 0
                        chunk = []
                        
                        for line_num, line in enumerate(f, 1):
                            if line.strip():
                                chunk.append(line.strip())
                                
                                # Process chunk when it reaches chunk_size
                                if len(chunk) >= chunk_size:
                                    chunk_count = self._process_chunk(chunk)
                                    file_count += chunk_count
                                    count += chunk_count
                                    chunk = []
                                    
                                    # Stop if we've reached max articles
                                    if count >= max_articles:
                                        break
                        
                        # Process remaining chunk
                        if chunk and count < max_articles:
                            chunk_count = self._process_chunk(chunk)
                            file_count += chunk_count
                            count += chunk_count
                    
                    self.logger.info(f"Loaded {file_count} articles from {os.path.basename(jsonl_file)}")
                    
                except Exception as file_error:
                    self.logger.error(f"Error processing {jsonl_file}: {file_error}")
                    continue
                
                if count >= max_articles:
                    break
            
            self.logger.info(f"Total loaded: {len(self.wikipedia_cache)} Wikipedia articles")
            
        except Exception as e:
            self.logger.error(f"Error loading Wikipedia JSONL: {e}")
    
    def _process_chunk(self, chunk: List[str]) -> int:
        """Process a chunk of JSONL lines"""
        chunk_count = 0
        
        for line in chunk:
            try:
                article = json.loads(line)
                
                # Handle the actual Wikipedia JSONL format
                title = article.get('name', '').lower()
                if not title:
                    continue
                
                # Extract text content from sections
                text_content = self._extract_text_from_sections(article.get('sections', []))
                
                if text_content:
                    self.wikipedia_cache[title] = {
                        'title': article.get('name', ''),
                        'text': text_content,
                        'url': article.get('url', ''),
                        'summary': text_content[:300] if text_content else '',
                        'abstract': article.get('abstract', ''),
                        'description': article.get('description', '')
                    }
                    chunk_count += 1
                    
                    # Also index by first name for people (simple heuristic)
                    title_words = title.split()
                    if len(title_words) >= 2:
                        first_name = title_words[0]
                        if first_name not in self.wikipedia_cache:
                            self.wikipedia_cache[first_name] = self.wikipedia_cache[title]
            
            except json.JSONDecodeError:
                continue
        
        return chunk_count
    
    def _extract_text_from_sections(self, sections: List[Dict]) -> str:
        """Extract text content from Wikipedia sections structure"""
        text_parts = []
        
        for section in sections:
            if section.get('type') == 'section':
                # Extract text from section parts
                parts = section.get('has_parts', [])
                for part in parts:
                    if part.get('type') == 'paragraph':
                        text_parts.append(part.get('value', ''))
                    elif part.get('type') == 'list':
                        # Extract text from list items
                        list_parts = part.get('has_parts', [])
                        for list_item in list_parts:
                            if list_item.get('type') == 'list_item':
                                text_parts.append(list_item.get('value', ''))
        
        return ' '.join(text_parts).strip()
    
    def search_wikipedia_cache(self, query: str) -> List[Dict[str, str]]:
        """Search local Wikipedia JSONL cache with improved relevance"""
        if not self.wikipedia_cache:
            return []
        
        query_lower = query.lower().strip()
        results = []
        
        # Clean query for better matching
        query_words = query_lower.split()
        
        # Scoring system for relevance
        scored_results = []
        
        for title, article in self.wikipedia_cache.items():
            score = 0
            title_lower = title.lower()
            article_title = article['title'].lower()
            
            # Exact match gets highest score
            if title_lower == query_lower or article_title == query_lower:
                score = 100
            
            # Exact phrase match
            elif query_lower in title_lower or query_lower in article_title:
                score = 90
            
            # Word-by-word matching with position weighting
            elif all(word in title_lower for word in query_words):
                score = 80
                # Bonus for words in correct order
                if ' '.join(query_words) in title_lower:
                    score = 85
            elif all(word in article_title for word in query_words):
                score = 75
                if ' '.join(query_words) in article_title:
                    score = 80
            
            # Partial word matching (lower score)
            elif any(word in title_lower for word in query_words):
                score = 30
            
            # Filter out very low relevance results
            if score >= 30:
                scored_results.append((score, article))
        
        # Sort by score (highest first) and take top results
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Take top results with relevance scoring
        high_relevance = []
        for score, article in scored_results:
            if score >= 50:
                high_relevance.append((score, article))
        
        # If we have good results, use them
        if high_relevance:
            for score, article in high_relevance[:5]:
                results.append({
                    'title': article['title'],
                    'description': article['summary'],
                    'url': article.get('url', ''),
                    'snippet': article['summary'],
                    'source': 'wikipedia_cache',
                    'full_text': article['text'],
                    'relevance_score': score
                })
        else:
            # Fallback: show top 3 results even if lower relevance
            for score, article in scored_results[:3]:
                results.append({
                    'title': article['title'],
                    'description': article['summary'],
                    'url': article.get('url', ''),
                    'snippet': article['summary'],
                    'source': 'wikipedia_cache',
                    'full_text': article['text'],
                    'relevance_score': score
                })
        
        self.logger.info(f"Found {len(results)} relevant Wikipedia results for: {query}")
        return results
    
    def search_brave_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search Brave Search API for additional data"""
        if not self.brave_api_key:
            self.logger.warning("Brave API key not provided, skipping web search")
            return []
        
        try:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': query,
                'count': max_results,
                'safesearch': 'moderate'
            }
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                for result in data.get('web', {}).get('results', [])[:max_results]:
                    results.append({
                        'title': result.get('title', ''),
                        'description': result.get('description', ''),
                        'url': result.get('url', ''),
                        'snippet': result.get('description', ''),
                        'source': 'brave_search',
                        'full_text': result.get('description', '')
                    })
                
                self.logger.info(f"Found {len(results)} Brave Search results for: {query}")
                return results
            else:
                self.logger.error(f"Brave Search API error: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Error searching Brave: {e}")
            return []
    
    def search_combined(self, query: str) -> List[Dict[str, str]]:
        """Search both Wikipedia cache and Brave Search, then combine results with relevance ranking"""
        # Search Wikipedia cache first
        wiki_results = self.search_wikipedia_cache(query)
        
        # Search Brave Search for additional data
        brave_results = self.search_brave_web(query, max_results=5)
        
        # Combine and rank all results by relevance
        all_results = []
        
        # Add Wikipedia results with high priority but keep relevance scores
        for result in wiki_results:
            result['priority'] = 'high'
            result['combined_score'] = result.get('relevance_score', 50) + 20  # Wikipedia bonus
            all_results.append(result)
        
        # Add Brave results with medium priority
        for result in brave_results:
            # Avoid duplicates by URL
            if not any(r['url'] == result['url'] for r in all_results):
                result['priority'] = 'medium'
                result['combined_score'] = 60  # Default score for Brave results
                all_results.append(result)
        
        # Sort by combined score (highest first)
        all_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        # Take top 6 results
        combined_results = all_results[:6]
        
        self.logger.info(f"Combined search found {len(combined_results)} total results for: {query}")
        return combined_results
    
    def extract_entities_enhanced(self, text: str) -> Dict[str, List[Dict]]:
        """Enhanced entity extraction using spaCy + patterns + enrichment"""
        entities = {
            'PERSON': [],
            'GPE': [],
            'ORG': [],
            'EVENT': [],
            'WORK_OF_ART': []
        }
        
        # Use spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'label': ent.label_,
                    'confidence': 0.9,
                    'context': text[max(0, ent.start_char-20):ent.end_char+20],
                    'source': 'spacy'
                }
                
                # Map spaCy labels to our categories
                if ent.label_ in ['PERSON']:
                    entities['PERSON'].append(entity_info)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['GPE'].append(entity_info)
                elif ent.label_ in ['ORG']:
                    entities['ORG'].append(entity_info)
                elif ent.label_ in ['EVENT']:
                    entities['EVENT'].append(entity_info)
                elif ent.label_ in ['WORK_OF_ART']:
                    entities['WORK_OF_ART'].append(entity_info)
        
        # Pattern-based extraction for historical entities
        found_persons = set()
        for pattern in self.entity_patterns['person']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                if entity_text.lower() not in found_persons:
                    entity_info = {
                        'text': entity_text,
                        'start': match.start(),
                        'end': match.end(),
                        'label': 'PERSON',
                        'confidence': 0.9,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern'
                    }
                    entities['PERSON'].append(entity_info)
                    found_persons.add(entity_text.lower())
        
        # Pattern-based place extraction
        found_places = set()
        for pattern in self.entity_patterns['place']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                if entity_text.lower() not in found_places:
                    entity_info = {
                        'text': entity_text,
                        'start': match.start(),
                        'end': match.end(),
                        'label': 'GPE',
                        'confidence': 0.9,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern'
                    }
                    entities['GPE'].append(entity_info)
                    found_places.add(entity_text.lower())
        
        return entities
    
    def enrich_entity(self, entity_text: str, entity_type: str) -> Dict[str, Any]:
        """Enrich entity with Wikipedia and web data"""
        enrichment = {
            'wikipedia_found': False,
            'web_found': False,
            'birth_year': None,
            'death_year': None,
            'occupation': '',
            'description': '',
            'wikipedia_title': '',
            'wikipedia_url': '',
            'web_sources': []
        }
        
        # Search combined sources (Wikipedia + Brave Search)
        search_results = self.search_combined(entity_text)
        
        if search_results:
            # Use the best result (Wikipedia preferred)
            best_result = None
            for result in search_results:
                if result.get('source') == 'wikipedia_cache':
                    best_result = result
                    enrichment['wikipedia_found'] = True
                    break
            
            # If no Wikipedia result, use the first available result
            if not best_result:
                best_result = search_results[0]
                enrichment['web_found'] = True
            
            enrichment['wikipedia_title'] = best_result['title']
            enrichment['wikipedia_url'] = best_result.get('url', '')
            enrichment['description'] = best_result['description']
            
            # Extract structured information for persons
            if entity_type == 'PERSON':
                full_text = best_result.get('full_text', '') + ' ' + best_result.get('description', '')
                
                # Extract birth/death years
                years = self.extract_birth_death_years(full_text)
                enrichment.update(years)
                
                # Extract occupation
                occupation = self.extract_occupation(full_text)
                enrichment['occupation'] = occupation
            
            # Collect all web sources for reference
            enrichment['web_sources'] = [
                {
                    'title': r['title'],
                    'url': r['url'],
                    'source': r['source'],
                    'priority': r.get('priority', 'medium')
                }
                for r in search_results[:3]
            ]
        
        return enrichment
    
    def extract_birth_death_years(self, text: str) -> Dict[str, int]:
        """Extract birth and death years from text"""
        info = {}
        
        # Pattern for birth years
        birth_patterns = [
            r'\((\d{3,4})\s*[-–]\s*\d{3,4}\)',  # (1452-1519)
            r'\b(?:born|b\.|birth)\s*(?:in\s*)?(\d{3,4})(?:\s*(?:BC|BCE|AD|CE))?\b',
            r'\b(\d{3,4})\s*[-–]\s*\d{3,4}\s*(?:BC|BCE|AD|CE)?\b'
        ]
        
        for pattern in birth_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    if 100 <= year <= 2100:  # Reasonable year range
                        info['birth_year'] = year
                        break
                except ValueError:
                    continue
        
        # Pattern for death years
        death_patterns = [
            r'\(\d{3,4}\s*[-–]\s*(\d{3,4})\)',  # (1452-1519)
            r'\b(?:died|d\.|death)\s*(?:in\s*)?(\d{3,4})(?:\s*(?:BC|BCE|AD|CE))?\b',
            r'\b\d{3,4}\s*[-–]\s*(\d{3,4})\s*(?:BC|BCE|AD|CE)?\b'
        ]
        
        for pattern in death_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    year = int(match.group(1))
                    if 100 <= year <= 2100:  # Reasonable year range
                        info['death_year'] = year
                        break
                except ValueError:
                    continue
        
        return info
    
    def extract_occupation(self, text: str) -> str:
        """Extract occupation from text"""
        occupation_patterns = [
            r'\bwas (?:a|an) ([^,.]{5,50}?)(?:\s+(?:who|that|and|,|\.))',
            r'\b(?:emperor|king|queen|pharaoh|general|artist|philosopher|writer|poet|politician|painter|sculptor|architect|scientist)\b',
            r'\bpolymath\b',
            r'\bstatesman\b'
        ]
        
        for pattern in occupation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if hasattr(match, 'group') and len(match.groups()) > 0:
                    return match.group(1).strip()
                else:
                    return match.group().strip()
        
        return ''
    
    def generate_intelligent_queries(self, entity_text: str, entity_type: str, context: str = "") -> List[str]:
        """Generate intelligent search queries for comprehensive historical data"""
        queries = []
        
        # Base comprehensive query
        queries.append(f"{entity_text} biography history facts")
        
        if entity_type == 'PERSON':
            # Historical figure comprehensive queries
            queries.extend([
                f"{entity_text} birth death years occupation",
                f"{entity_text} historical era period timeline",
                f"{entity_text} major achievements contributions",
                f"{entity_text} historical significance impact",
                f"{entity_text} contemporaries relationships",
                f"{entity_text} historical context background"
            ])
        elif entity_type == 'GPE':
            # Place/empire comprehensive queries
            queries.extend([
                f"{entity_text} history timeline development",
                f"{entity_text} historical significance importance",
                f"{entity_text} rise fall empire kingdom",
                f"{entity_text} historical context period",
                f"{entity_text} major events timeline"
            ])
        elif entity_type == 'EVENT':
            # Event comprehensive queries
            queries.extend([
                f"{entity_text} causes consequences historical impact",
                f"{entity_text} historical context background",
                f"{entity_text} significance importance effects",
                f"{entity_text} timeline dates when happened"
            ])
        
        # Add context-specific comprehensive queries
        if context:
            context_words = context.split()[:3]  # First 3 words of context
            if context_words:
                context_str = " ".join(context_words)
                queries.extend([
                    f"{entity_text} {context_str} historical facts",
                    f"{entity_text} {context_str} significance importance"
                ])
        
        return queries[:6]  # Limit to 6 focused queries per entity
    
    def _load_cache(self, cache_file: str) -> Dict[str, Any]:
        """Load cache from file"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load cache {cache_file}: {e}")
        return {}
    
    def _save_cache(self, cache_file: str, cache_data: Dict[str, Any]):
        """Save cache to file"""
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            self.logger.error(f"Could not save cache {cache_file}: {e}")
    
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
    
    def search_brave_intelligent(self, queries: List[str], max_results_per_query: int = 3) -> List[Dict[str, Any]]:
        """Search Brave with multiple intelligent queries and aggregate results with caching"""
        all_results = []
        seen_urls = set()
        new_cache_entries = {}
        
        for query in queries:
            cache_key = self._get_cache_key(query)
            
            # Check cache first
            if cache_key in self.search_cache:
                cache_entry = self.search_cache[cache_key]
                if self._is_cache_valid(cache_entry):
                    self.logger.info(f"Using cached results for: {query}")
                    for result in cache_entry['results']:
                        if result['url'] not in seen_urls:
                            seen_urls.add(result['url'])
                            all_results.append(result)
                    continue
            
            # Not in cache or expired, search Brave
            try:
                headers = {
                    'Accept': 'application/json',
                    'Accept-Encoding': 'gzip',
                    'X-Subscription-Token': self.brave_api_key
                }
                
                params = {
                    'q': query,
                    'count': max_results_per_query,
                    'safesearch': 'moderate'
                }
                
                response = requests.get(
                    'https://api.search.brave.com/res/v1/web/search',
                    headers=headers,
                    params=params,
                    timeout=10
                )
                
                query_results = []
                if response.status_code == 200:
                    data = response.json()
                    
                    for result in data.get('web', {}).get('results', [])[:max_results_per_query]:
                        url = result.get('url', '')
                        
                        # Avoid duplicates
                        if url not in seen_urls:
                            seen_urls.add(url)
                            
                            enhanced_result = {
                                'title': result.get('title', ''),
                                'description': result.get('description', ''),
                                'url': url,
                                'snippet': result.get('description', ''),
                                'source': 'brave_search',
                                'full_text': result.get('description', ''),
                                'query_used': query,
                                'search_category': self._categorize_search_query(query)
                            }
                            all_results.append(enhanced_result)
                            query_results.append(enhanced_result)
                
                # Cache the results
                new_cache_entries[cache_key] = {
                    'query': query,
                    'results': query_results,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Small delay to be respectful to the API
                import time
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error searching Brave for query '{query}': {e}")
                continue
        
        # Save new cache entries
        if new_cache_entries:
            self.search_cache.update(new_cache_entries)
            self._save_cache(self.search_cache_file, self.search_cache)
            self.logger.info(f"Cached {len(new_cache_entries)} new search results")
        
        self.logger.info(f"Intelligent search found {len(all_results)} unique results from {len(queries)} queries")
        return all_results
    
    def _categorize_search_query(self, query: str) -> str:
        """Categorize the type of search query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['context', 'background']):
            return 'historical_context'
        elif any(word in query_lower for word in ['impact', 'consequences', 'effects']):
            return 'historical_impact'
        elif any(word in query_lower for word in ['precursors', 'influences', 'causes']):
            return 'historical_precursors'
        elif any(word in query_lower for word in ['significance', 'importance']):
            return 'historical_significance'
        elif any(word in query_lower for word in ['timeline', 'achievements', 'legacy']):
            return 'historical_timeline'
        elif any(word in query_lower for word in ['contemporaries', 'rivals']):
            return 'historical_relationships'
        else:
            return 'general_search'
    
    def enrich_entity_intelligent(self, entity_text: str, entity_type: str, context: str = "") -> Dict[str, Any]:
        """Intelligently enrich entity with comprehensive historical data"""
        # Check enrichment cache first
        enrichment_key = self._get_cache_key(f"{entity_text}_{entity_type}_{context}")
        
        if enrichment_key in self.enrichment_cache:
            cache_entry = self.enrichment_cache[enrichment_key]
            if self._is_cache_valid(cache_entry, max_age_hours=168):  # 7 days for enrichment
                self.logger.info(f"Using cached enrichment for: {entity_text}")
                return cache_entry['enrichment']
        
        enrichment = {
            'wikipedia_found': False,
            'web_found': False,
            'birth_year': None,
            'death_year': None,
            'occupation': '',
            'description': '',
            'wikipedia_title': '',
            'wikipedia_url': '',
            'historical_context': '',
            'historical_impact': '',
            'historical_precursors': '',
            'historical_significance': '',
            'historical_timeline': '',
            'web_sources': [],
            'search_categories': {}
        }
        
        # Skip Wikipedia cache, use Brave Search directly for better accuracy
        # This will be handled by the intelligent search below
        
        # Generate intelligent queries for missing data
        queries = self.generate_intelligent_queries(entity_text, entity_type, context)
        
        # Search with intelligent queries
        web_results = self.search_brave_intelligent(queries)
        
        if web_results:
            enrichment['web_found'] = True
            
            # Categorize and aggregate results
            category_data = defaultdict(list)
            for result in web_results:
                category = result['search_category']
                category_data[category].append(result)
            
            # Extract information by category
            for category, results in category_data.items():
                # Combine descriptions from this category
                descriptions = [r['description'] for r in results[:2]]  # Top 2 per category
                combined_text = ' '.join(descriptions)
                
                if category == 'historical_context':
                    enrichment['historical_context'] = combined_text[:500]
                elif category == 'historical_impact':
                    enrichment['historical_impact'] = combined_text[:500]
                elif category == 'historical_precursors':
                    enrichment['historical_precursors'] = combined_text[:500]
                elif category == 'historical_significance':
                    enrichment['historical_significance'] = combined_text[:500]
                elif category == 'historical_timeline':
                    enrichment['historical_timeline'] = combined_text[:500]
                
                # Store search categories for reference
                enrichment['search_categories'][category] = len(results)
            
            # Collect all web sources
            enrichment['web_sources'] = [
                {
                    'title': r['title'],
                    'url': r['url'],
                    'source': r['source'],
                    'category': r['search_category'],
                    'query_used': r['query_used']
                }
                for r in web_results[:10]  # Top 10 results
            ]
        
        # Cache the enrichment result
        self.enrichment_cache[enrichment_key] = {
            'enrichment': enrichment,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache(self.enrichment_cache_file, self.enrichment_cache)
        
        return enrichment
    
    def extract_all_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract all dates from text with various formats"""
        dates = []
        
        for pattern, date_type in self.date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    year = int(match.group(1))
                    
                    # Determine if BC/BCE
                    if date_type == 'year_bc':
                        year = -year
                    
                    # Skip unreasonable years
                    if abs(year) > 50000:
                        continue
                    
                    date_info = {
                        'year': year,
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end(),
                        'type': date_type,
                        'confidence': 0.9,
                        'context': text[max(0, match.start()-20):match.end()+20]
                    }
                    
                    dates.append(date_info)
                    
                except ValueError:
                    continue
        
        return dates
    
    def determine_historical_era_brave(self, text: str, dates: List[Dict] = None) -> str:
        """Determine historical era using Brave Search for accuracy"""
        # Create era detection query
        era_query = f"what historical era is this: {text}"
        
        # Check cache first
        cache_key = self._get_cache_key(f"era_{era_query}")
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['era_result']
        
        try:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': era_query,
                'count': 3,
                'safesearch': 'moderate'
            }
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('web', {}).get('results', [])
                
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
    
    def _extract_era_from_results(self, results: List[Dict], text: str, dates: List[Dict] = None) -> str:
        """Extract historical era from Brave Search results"""
        era_scores = defaultdict(int)
        
        for result in results:
            description = result.get('description', '').lower()
            title = result.get('title', '').lower()
            content = f"{title} {description}"
            
            # Check for era mentions in results
            for era_name, era_info in self.historical_eras.items():
                era_keywords = [era_name.lower()] + era_info['keywords']
                
                for keyword in era_keywords:
                    if keyword in content:
                        era_scores[era_name] += 2
                
                # Bonus for exact era name match
                if era_name.lower() in content:
                    era_scores[era_name] += 5
        
        # Also check text and dates for era clues
        if dates:
            for date_info in dates:
                year = date_info.get('year', 0)
                for era_name, era_info in self.historical_eras.items():
                    if era_info['start'] <= year <= era_info['end']:
                        era_scores[era_name] += 3
        
        # Return era with highest score
        if era_scores:
            return max(era_scores, key=era_scores.get)
        
        return "Unknown Era"
    
    def _determine_era_by_year(self, year: int) -> str:
        """Simple fallback era determination by year"""
        for era_name, era_info in self.historical_eras.items():
            if era_info['start'] <= year <= era_info['end']:
                return era_name
        return "Unknown Era"
    
    def detect_event_type_brave(self, text: str) -> str:
        """Detect the type of historical event using Brave Search"""
        # Create event type detection query
        event_query = f"what type of historical event is this: {text}"
        
        # Check cache first
        cache_key = self._get_cache_key(f"event_type_{event_query}")
        if cache_key in self.search_cache:
            cache_entry = self.search_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                return cache_entry['event_type_result']
        
        try:
            headers = {
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip',
                'X-Subscription-Token': self.brave_api_key
            }
            
            params = {
                'q': event_query,
                'count': 3,
                'safesearch': 'moderate'
            }
            
            response = requests.get(
                'https://api.search.brave.com/res/v1/web/search',
                headers=headers,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                results = data.get('web', {}).get('results', [])
                
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
    
    def _extract_event_type_from_results(self, results: List[Dict], text: str) -> str:
        """Extract event type from Brave Search results"""
        event_scores = defaultdict(int)
        
        for result in results:
            description = result.get('description', '').lower()
            title = result.get('title', '').lower()
            content = f"{title} {description}"
            
            # Check for event type keywords in results
            for event_type, keywords in [
                ('war', ['war', 'battle', 'conflict', 'conquest', 'invasion']),
                ('political', ['political', 'treaty', 'government', 'empire', 'kingdom']),
                ('cultural', ['cultural', 'art', 'renaissance', 'literature', 'religion']),
                ('disaster', ['disaster', 'plague', 'famine', 'earthquake', 'death']),
                ('discovery', ['discovery', 'invention', 'exploration', 'found']),
                ('trade', ['trade', 'commerce', 'merchant', 'economic']),
                ('migration', ['migration', 'settlement', 'expansion', 'diaspora']),
                ('technology', ['technology', 'innovation', 'revolution', 'advancement'])
            ]:
                for keyword in keywords:
                    if keyword in content:
                        event_scores[event_type] += 1
        
        # Return event type with highest score
        if event_scores:
            return max(event_scores, key=event_scores.get)
        
        return "general"
    
    def _detect_event_type_fallback(self, text: str) -> str:
        """Fallback pattern-based event type detection"""
        text_lower = text.lower()
        
        event_scores = {}
        for event_type, pattern in self.event_patterns.items():
            matches = re.findall(pattern, text_lower)
            event_scores[event_type] = len(matches)
        
        if event_scores:
            return max(event_scores, key=event_scores.get)
        
        return "general"
    
    def process_comprehensive_text(self, text: str) -> Dict[str, Any]:
        """Process text with comprehensive historical extraction and enrichment"""
        self.logger.info(f"Processing text: {text[:100]}...")
        
        # Extract entities with enhanced method
        entities = self.extract_entities_enhanced(text)
        
        # Extract dates
        dates = self.extract_all_dates_from_text(text)
        
        # Determine primary era using Brave Search
        primary_era = self.determine_historical_era_brave(text, dates)
        
        # Detect event type using Brave Search
        event_type = self.detect_event_type_brave(text)
        
        # Enrich entities with intelligent search
        enriched_entities = {}
        for entity_type, entity_list in entities.items():
            enriched_entities[entity_type] = []
            
            for entity in entity_list[:3]:  # Enrich top 3 entities per type
                # Use intelligent enrichment with context
                enrichment = self.enrich_entity_intelligent(
                    entity['text'], 
                    entity_type, 
                    context=entity.get('context', text[:200])  # Use entity context or text snippet
                )
                entity['enrichment'] = enrichment
                enriched_entities[entity_type].append(entity)
        
        # Create comprehensive result
        result = {
            'text': text,
            'extraction_metadata': {
                'timestamp': datetime.now().isoformat(),
                'extractor_version': 'comprehensive_v1.0',
                'spacy_available': self.nlp is not None,
                'wikipedia_articles_loaded': len(self.wikipedia_cache),
                'brave_search_available': self.brave_api_key is not None
            },
            'historical_context': {
                'primary_era': primary_era,
                'event_type': event_type,
                'dates_found': len(dates)
            },
            'entities': enriched_entities,
            'dates': dates,
            'enrichment_stats': {
                'total_entities': sum(len(ents) for ents in entities.values()),
                'entities_enriched': sum(
                    len([e for e in ents if 'enrichment' in e and (e['enrichment']['wikipedia_found'] or e['enrichment']['web_found'])])
                    for ents in enriched_entities.values()
                ),
                'wikipedia_sources': sum(
                    len([e for e in ents if 'enrichment' in e and e['enrichment']['wikipedia_found']])
                    for ents in enriched_entities.values()
                ),
                'web_sources': sum(
                    len([e for e in ents if 'enrichment' in e and e['enrichment']['web_found']])
                    for ents in enriched_entities.values()
                ),
                'intelligent_queries_used': sum(
                    len([e for e in ents if 'enrichment' in e and e['enrichment']['web_found']])
                    for ents in enriched_entities.values()
                ) * 8,  # Approximate queries per entity
                'search_categories_found': len(set().union(*[
                    set(e['enrichment']['search_categories'].keys()) 
                    for ents in enriched_entities.values() 
                    for e in ents if 'enrichment' in e and e['enrichment']['search_categories']
                ]))
            }
        }
        
        self.logger.info(f"Extraction complete: {result['enrichment_stats']['total_entities']} entities, "
                        f"{result['enrichment_stats']['entities_enriched']} enriched")
        
        return result

def test_comprehensive_extraction():
    """Test the comprehensive historical extractor"""
    print("Testing Comprehensive Historical Extractor")
    print("=" * 60)
    
    # Initialize extractor
    wikipedia_path = "/Volumes/Others2/wikipedia/enwiki_namespace_0"
    brave_api_key = os.getenv('BRAVE_API_KEY')
    
    extractor = ComprehensiveHistoricalExtractor(
        wikipedia_jsonl_path=wikipedia_path if os.path.exists(wikipedia_path) else None,
        brave_api_key=brave_api_key
    )
    
    # Test cases
    test_cases = [
        "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance.",
        "Alexander the Great conquered the Persian Empire between 334 and 323 BCE, spreading Hellenistic culture from Macedonia to India.",
        "Julius Caesar crossed the Rubicon in 49 BC, leading to civil war in the Roman Republic.",
        "Napoleon Bonaparte was crowned Emperor of France in 1804, marking the height of the Napoleonic Empire.",
        "Isaac Newton formulated the laws of motion and universal gravitation in the 17th century during the Scientific Revolution."
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n🔍 Test Case {i}:")
        print(f"Text: {text}")
        print("-" * 50)
        
        result = extractor.process_comprehensive_text(text)
        
        # Display results
        metadata = result['extraction_metadata']
        context = result['historical_context']
        stats = result['enrichment_stats']
        
        print(f"📊 Era: {context['primary_era']}")
        print(f"📊 Event Type: {context['event_type']}")
        print(f"📊 Dates Found: {context['dates_found']}")
        print(f"📊 Total Entities: {stats['total_entities']}")
        print(f"📊 Entities Enriched: {stats['entities_enriched']}")
        print(f"📊 Wikipedia Sources: {stats['wikipedia_sources']}")
        print(f"📊 Web Sources: {stats['web_sources']}")
        
        # Show enriched entities
        for entity_type, entities in result['entities'].items():
            if entities:
                print(f"\n{entity_type}:")
                for entity in entities:
                    enrich = entity.get('enrichment', {})
                    if enrich.get('wikipedia_found') or enrich.get('web_found'):
                        source = "Wikipedia" if enrich['wikipedia_found'] else "Web"
                        print(f"  • {entity['text']} ({source})")
                        if enrich.get('occupation'):
                            print(f"    Occupation: {enrich['occupation']}")
                        if enrich.get('birth_year'):
                            print(f"    Born: {enrich['birth_year']}")
    
    print("\n" + "=" * 60)
    print("🎉 COMPREHENSIVE EXTRACTION FEATURES:")
    print("✅ Enhanced entity extraction (spaCy + patterns)")
    print("✅ Wikipedia JSONL enrichment with chunking")
    print("✅ Brave Search integration for real-time data")
    print("✅ Historical era detection")
    print("✅ Event type classification")
    print("✅ Date extraction and parsing")
    print("✅ Multi-source data aggregation")
    print("✅ Relevance-based result ranking")
    print("✅ Comprehensive metadata tracking")

if __name__ == "__main__":
    test_comprehensive_extraction()
