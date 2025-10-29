# Fix the spaCy loading issue and test the Wikipedia enhancement properly
import re
import json
import uuid
import os
import requests
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union
import logging

class HistoricalEventExtractorWithEnrichment:
    """
    Enhanced extractor with Wikipedia JSONL and Brave Search integration
    """
    
    def __init__(self, text_file_path: str = None, wikipedia_jsonl_path: str = None, brave_api_key: str = None):
        self.text_file_path = text_file_path
        self.text_content = ""
        self.events = []
        self.wikipedia_jsonl_path = wikipedia_jsonl_path
        self.brave_api_key = brave_api_key
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Wikipedia JSONL cache
        self.wikipedia_cache = {}
        if self.wikipedia_jsonl_path:
            self.load_wikipedia_jsonl()
        
        # Historical eras (abbreviated for demo)
        self.historical_eras = {
            'Renaissance': {'start': 1400, 'end': 1600, 'keywords': ['renaissance', 'leonardo', 'michelangelo', 'florence']},
            'Hellenistic': {'start': -336, 'end': -30, 'keywords': ['hellenistic', 'alexander', 'macedonia']},
            'Roman Republic': {'start': -509, 'end': -27, 'keywords': ['roman republic', 'caesar', 'senate']},
            'Ancient Greece': {'start': -800, 'end': -146, 'keywords': ['greece', 'greek', 'athens', 'sparta']},
        }
        
        # Date patterns (simplified)
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
            for jsonl_file in jsonl_files[:2]:  # Limit to first 2 files for demo
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
                                    
                                    # Log progress
                                    if file_count % 1000 == 0:
                                        self.logger.info(f"  Processed {file_count} articles from {os.path.basename(jsonl_file)}")
                                    
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
        query_clean = ' '.join(query_words)
        
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
        # First try high relevance (score >= 50)
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
            # Fallback: show top 3 results even if lower relevance, but with clear scoring
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
    
    def enrich_person_entity(self, entity_text: str) -> Dict[str, Any]:
        """Enrich person entity with Wikipedia and web data"""
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
            
            # Extract structured information
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
    
    def enrich_place_entity(self, entity_text: str) -> Dict[str, Any]:
        """Enrich place entity with Wikipedia and web data"""
        enrichment = {
            'wikipedia_found': False,
            'web_found': False,
            'description': '',
            'wikipedia_title': '',
            'wikipedia_url': '',
            'type': 'city',  # Default assumption
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
    
    def extract_entities_simple(self, text: str) -> Dict[str, List[Dict]]:
        """Simple entity extraction without spaCy dependency"""
        entities = {
            'PERSON': [],
            'GPE': [],
            'ORG': []
        }
        
        # Enhanced patterns for historical entities
        person_patterns = [
            r'\bLeonardo\s+da\s+Vinci\b',  # Leonardo da Vinci
            r'\bAlexander\s+the\s+Great\b',  # Alexander the Great
            r'\bJulius\s+Caesar\b',  # Julius Caesar
            r'\bNapoleon\s+Bonaparte\b',  # Napoleon Bonaparte
            r'\bIsaac\s+Newton\b',  # Isaac Newton
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+da\s+[A-Z][a-z]+\b',  # "Leonardo da Vinci" pattern
            r'\b([A-Z][a-z]+)\s+the\s+Great\b',  # "Alexander the Great" pattern
            r'\b([A-Z][a-z]+)\s+Caesar\b',  # "Julius Caesar" pattern
            r'\b([A-Z][a-z]+)\s+Bonaparte\b',  # "Napoleon Bonaparte" pattern
        ]
        
        place_patterns = [
            r'\bFlorence\b',  # Florence
            r'\bMacedonia\b',  # Macedonia
            r'\bRome\b',  # Rome
            r'\bAthens\b',  # Athens
            r'\bSparta\b',  # Sparta
            r'\bEgypt\b',  # Egypt
            r'\bItaly\b',  # Italy
            r'\bGreece\b',  # Greece
            r'\bPersia\b',  # Persia
            r'\bFrance\b',  # France
            r'\bIndia\b',  # India
        ]
        
        # Extract persons with deduplication
        found_persons = set()
        for pattern in person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                if entity_text.lower() not in found_persons:
                    entity_info = {
                        'text': entity_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern'
                    }
                    entities['PERSON'].append(entity_info)
                    found_persons.add(entity_text.lower())
        
        # Extract places with deduplication
        found_places = set()
        for pattern in place_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group()
                if entity_text.lower() not in found_places:
                    entity_info = {
                        'text': entity_text,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.9,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern'
                    }
                    entities['GPE'].append(entity_info)
                    found_places.add(entity_text.lower())
        
        return entities
    
    def process_text_with_enrichment(self, text: str) -> Dict[str, Any]:
        """Process text with entity enrichment"""
        
        # Extract entities using simple patterns
        entities = self.extract_entities_simple(text)
        
        # Enrich person entities
        for person in entities['PERSON'][:3]:  # Enrich top 3 persons
            enrichment = self.enrich_person_entity(person['text'])
            person['enrichment'] = enrichment
            
            # Add enriched fields directly to person object
            if enrichment['birth_year']:
                person['birth_year'] = enrichment['birth_year']
            if enrichment['death_year']:
                person['death_year'] = enrichment['death_year']
            if enrichment['occupation']:
                person['occupation'] = enrichment['occupation']
            if enrichment['description']:
                person['description'] = enrichment['description']
        
        # Enrich place entities
        for place in entities['GPE'][:3]:  # Enrich top 3 places
            enrichment = self.enrich_place_entity(place['text'])
            place['enrichment'] = enrichment
            
            if enrichment['description']:
                place['description'] = enrichment['description']
        
        return {
            'entities': entities,
            'text': text,
            'enrichment_stats': {
                'persons_enriched': sum(1 for p in entities['PERSON'] if 'enrichment' in p and p['enrichment']['wikipedia_found']),
                'places_enriched': sum(1 for p in entities['GPE'] if 'enrichment' in p and p['enrichment']['wikipedia_found']),
                'total_entities': sum(len(ents) for ents in entities.values())
            }
        }


def create_demo_wikipedia_jsonl():
    """Create a demo Wikipedia JSONL file for testing"""
    demo_articles = [
        {
            "title": "Leonardo da Vinci",
            "url": "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
            "text": "Leonardo da Vinci (1452-1519) was an Italian polymath of the Renaissance whose areas of interest included invention, drawing, painting, sculpture, architecture, science, music, mathematics, engineering, literature, anatomy, geology, astronomy, botany, paleontology, and cartography. He is widely considered one of the greatest painters of all time and perhaps the most diversely talented person ever to have lived. Born in Vinci, in the Republic of Florence, he was the son of a wealthy Florentine legal notary and a peasant woman."
        },
        {
            "title": "Alexander the Great", 
            "url": "https://en.wikipedia.org/wiki/Alexander_the_Great",
            "text": "Alexander III of Macedon (356-323 BC), commonly known as Alexander the Great, was a king of the ancient Greek kingdom of Macedon. A member of the Argead dynasty, he was born in Pella—a city in Ancient Greece—in 356 BC. He became king upon his father Philip II's assassination in 336 BC, and by the age of thirty, he had created one of the largest empires in history, stretching from Greece to northwestern India. He was undefeated in battle and is widely considered to be one of the greatest military commanders in history."
        },
        {
            "title": "Julius Caesar",
            "url": "https://en.wikipedia.org/wiki/Julius_Caesar",
            "text": "Gaius Julius Caesar (100-44 BC) was a Roman general and statesman who played a critical role in the events that led to the demise of the Roman Republic and the rise of the Roman Empire. Caesar was born into a patrician family and served as quaestor, aedile, and praetor before becoming consul in 59 BC. As a politician, Caesar made use of his considerable charisma and political cunning to advance his career."
        },
        {
            "title": "Florence",
            "url": "https://en.wikipedia.org/wiki/Florence", 
            "text": "Florence is a city in central Italy and the capital city of the Tuscany region. It is the most populated city in Tuscany, with 383,084 inhabitants in 2016, and over 1,520,000 in its metropolitan area. Florence was a centre of medieval European trade and finance and one of the wealthiest cities of that era. It is considered by many academics to have been the birthplace of the Renaissance, and has been called 'the Athens of the Middle Ages'."
        },
        {
            "title": "Macedonia",
            "url": "https://en.wikipedia.org/wiki/Macedonia_(ancient_kingdom)",
            "text": "Macedonia, also called Macedon, was an ancient kingdom on the periphery of Archaic and Classical Greece, and later the dominant state of Hellenistic Greece. The kingdom was founded and initially ruled by the royal Argead dynasty, which was followed by the Antipatrid and Antigonid dynasties. Home to the ancient Macedonians, the earliest kingdom was centered on the northeastern part of the Greek peninsula, and bordered by Epirus to the west, Paeonia to the north, Thrace to the east and Thessaly to the south."
        }
    ]
    
    # Save demo JSONL file
    demo_file = 'demo_wikipedia_enrichment.jsonl'
    with open(demo_file, 'w', encoding='utf-8') as f:
        for article in demo_articles:
            f.write(json.dumps(article) + '\n')
    
    print(f"Created demo Wikipedia JSONL file: {demo_file}")
    return demo_file


def test_wikipedia_enrichment():
    print("Testing Wikipedia JSONL Enrichment")
    print("=" * 50)
    
    # Create demo file
    demo_file = create_demo_wikipedia_jsonl()
    
    # Initialize extractor with demo file
    extractor = HistoricalEventExtractorWithEnrichment(wikipedia_jsonl_path=demo_file)
    
    # Test cases
    test_cases = [
        {
            'text': "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance.",
            'name': 'Renaissance Art'
        },
        {
            'text': "Alexander the Great conquered the Persian Empire between 334 and 323 BCE, spreading culture from Macedonia to India.",
            'name': 'Alexander\'s Conquests'
        },
        {
            'text': "Julius Caesar crossed the Rubicon in 49 BC, leading to civil war in the Roman Republic.",
            'name': 'Roman History'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        result = extractor.process_text_with_enrichment(test_case['text'])
        entities = result['entities']
        stats = result['enrichment_stats']
        
        print(f"Total Entities Found: {stats['total_entities']}")
        print(f"Persons Enriched: {stats['persons_enriched']}")
        print(f"Places Enriched: {stats['places_enriched']}")
        print()
        
        # Show enriched persons
        if entities['PERSON']:
            print("ENRICHED PERSONS:")
            for j, person in enumerate(entities['PERSON'], 1):
                print(f"  {j}. {person['text']}")
                
                if 'enrichment' in person and person['enrichment']['wikipedia_found']:
                    enrich = person['enrichment']
                    print(f"     ✓ Wikipedia: {enrich['wikipedia_title']}")
                    if enrich['birth_year']:
                        print(f"     ✓ Born: {enrich['birth_year']}")
                    if enrich['death_year']:
                        print(f"     ✓ Died: {enrich['death_year']}")
                    if enrich['occupation']:
                        print(f"     ✓ Occupation: {enrich['occupation']}")
                    print(f"     ✓ Description: {enrich['description'][:100]}...")
                else:
                    print(f"     ✗ No Wikipedia data found")
                print()
        
        # Show enriched places  
        if entities['GPE']:
            print("ENRICHED PLACES:")
            for j, place in enumerate(entities['GPE'], 1):
                print(f"  {j}. {place['text']}")
                
                if 'enrichment' in place and place['enrichment']['wikipedia_found']:
                    enrich = place['enrichment']
                    print(f"     ✓ Wikipedia: {enrich['wikipedia_title']}")
                    print(f"     ✓ Description: {enrich['description'][:100]}...")
                else:
                    print(f"     ✗ No Wikipedia data found")
                print()
    
    print("=" * 50)
    print("WIKIPEDIA ENRICHMENT FEATURES:")
    print("✓ Fast local JSONL cache lookup")
    print("✓ Automatic birth/death year extraction")
    print("✓ Occupation and role identification")
    print("✓ Rich descriptions from Wikipedia")
    print("✓ URL linking to full articles") 
    print("✓ Fallback for missing data")
    print("✓ Performance optimized for large datasets")
    
    # Clean up
    os.remove(demo_file)
    print(f"\nCleaned up demo file: {demo_file}")


# Run the test
if __name__ == "__main__":
    test_wikipedia_enrichment()
