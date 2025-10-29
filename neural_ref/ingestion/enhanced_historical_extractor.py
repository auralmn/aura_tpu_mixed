# Enhanced Historical Event Extractor with Era Names and spaCy Entity Extraction
import re
import json
import uuid
import spacy
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union

class HistoricalEventExtractorWithEras:
    """
    Enhanced extractor with era names and entity-based historical figure detection
    """
    
    def __init__(self, text_file_path: str = None, spacy_model: str = "en_core_web_sm"):
        self.text_file_path = text_file_path
        self.text_content = ""
        self.events = []
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            print(f"spaCy model '{spacy_model}' not found. Installing...")
            import subprocess
            try:
                subprocess.run([f"python", "-m", "spacy", "download", spacy_model], check=True)
                self.nlp = spacy.load(spacy_model)
                print(f"Successfully installed and loaded: {spacy_model}")
            except:
                print("Failed to install spaCy model. Using fallback entity extraction.")
                self.nlp = None
        
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
            # Prehistoric eras
            'Paleolithic': {'start': -2500000, 'end': -10000, 'keywords': ['paleolithic', 'old stone age', 'hunter', 'gatherer', 'cave', 'mammoth']},
            'Mesolithic': {'start': -10000, 'end': -8000, 'keywords': ['mesolithic', 'middle stone age', 'foraging', 'fishing']},
            'Neolithic': {'start': -8000, 'end': -3000, 'keywords': ['neolithic', 'new stone age', 'agriculture', 'farming', 'pottery', 'settlement']},
            
            # Metal ages
            'Copper Age': {'start': -5000, 'end': -3200, 'keywords': ['copper', 'chalcolithic', 'metallurgy']},
            'Bronze Age': {'start': -3200, 'end': -1200, 'keywords': ['bronze', 'bronze age', 'metal', 'alloy']},
            'Iron Age': {'start': -1200, 'end': -600, 'keywords': ['iron', 'iron age', 'iron tools', 'ironworking']},
            
            # Ancient civilizations
            'Ancient Near East': {'start': -3500, 'end': -539, 'keywords': ['mesopotamia', 'babylon', 'assyria', 'sumerian', 'akkadian', 'hammurabi']},
            'Ancient Egypt': {'start': -3100, 'end': -30, 'keywords': ['egypt', 'pharaoh', 'pyramid', 'nile', 'hieroglyph', 'mummy']},
            'Ancient Greece': {'start': -800, 'end': -146, 'keywords': ['greece', 'greek', 'athens', 'sparta', 'pericles', 'socrates', 'plato', 'aristotle']},
            'Hellenistic': {'start': -336, 'end': -30, 'keywords': ['hellenistic', 'alexander', 'ptolemy', 'seleucid', 'macedon']},
            'Roman Republic': {'start': -509, 'end': -27, 'keywords': ['roman republic', 'senate', 'consul', 'cicero', 'caesar', 'republic']},
            'Roman Empire': {'start': -27, 'end': 476, 'keywords': ['roman empire', 'emperor', 'augustus', 'rome', 'legion', 'imperial']},
            'Byzantine Empire': {'start': 330, 'end': 1453, 'keywords': ['byzantine', 'constantinople', 'orthodox', 'justinian']},
            
            # Medieval periods
            'Early Medieval': {'start': 476, 'end': 1000, 'keywords': ['early medieval', 'dark ages', 'migration', 'germanic', 'feudal']},
            'High Medieval': {'start': 1000, 'end': 1300, 'keywords': ['high medieval', 'crusades', 'cathedral', 'scholastic', 'chivalry']},
            'Late Medieval': {'start': 1300, 'end': 1500, 'keywords': ['late medieval', 'plague', 'hundred years', 'gothic', 'manuscript']},
            
            # Renaissance and early modern
            'Renaissance': {'start': 1400, 'end': 1600, 'keywords': ['renaissance', 'humanism', 'leonardo', 'michelangelo', 'medici', 'florence']},
            'Age of Exploration': {'start': 1400, 'end': 1650, 'keywords': ['exploration', 'columbus', 'vasco', 'magellan', 'conquistador', 'new world']},
            'Reformation': {'start': 1517, 'end': 1648, 'keywords': ['reformation', 'luther', 'protestant', 'calvin', 'counter-reformation']},
            'Baroque': {'start': 1600, 'end': 1750, 'keywords': ['baroque', 'absolutism', 'louis xiv', 'versailles']},
            
            # Enlightenment and modern
            'Enlightenment': {'start': 1685, 'end': 1815, 'keywords': ['enlightenment', 'voltaire', 'rousseau', 'revolution', 'reason']},
            'Industrial Revolution': {'start': 1760, 'end': 1840, 'keywords': ['industrial', 'factory', 'steam', 'railroad', 'textile']},
            'Modern Era': {'start': 1800, 'end': 1914, 'keywords': ['modern', 'nationalism', 'imperialism', 'colonialism']},
            'World Wars Era': {'start': 1914, 'end': 1945, 'keywords': ['world war', 'great war', 'hitler', 'stalin', 'fascism']},
            'Contemporary': {'start': 1945, 'end': 2100, 'keywords': ['contemporary', 'cold war', 'nuclear', 'digital', 'globalization']}
        }
        
        # Historical figures by era (for entity matching)
        self.historical_figures_by_era = {
            'Ancient Near East': ['Hammurabi', 'Sargon', 'Nebuchadnezzar', 'Cyrus', 'Darius'],
            'Ancient Egypt': ['Cleopatra', 'Ramesses', 'Akhenaten', 'Tutankhamun', 'Hatshepsut'],
            'Ancient Greece': ['Alexander the Great', 'Pericles', 'Socrates', 'Plato', 'Aristotle', 'Homer'],
            'Roman Republic': ['Julius Caesar', 'Cicero', 'Pompey', 'Crassus', 'Brutus'],
            'Roman Empire': ['Augustus', 'Trajan', 'Marcus Aurelius', 'Constantine', 'Diocletian'],
            'Byzantine Empire': ['Justinian', 'Theodora', 'Basil II', 'Constantine VII'],
            'Early Medieval': ['Charlemagne', 'Alfred the Great', 'Clovis', 'Theodoric'],
            'High Medieval': ['William the Conqueror', 'Thomas Aquinas', 'Eleanor of Aquitaine', 'Richard the Lionheart'],
            'Late Medieval': ['Joan of Arc', 'Geoffrey Chaucer', 'Dante', 'Giotto'],
            'Renaissance': ['Leonardo da Vinci', 'Michelangelo', 'Machiavelli', 'Medici', 'Erasmus'],
            'Age of Exploration': ['Christopher Columbus', 'Vasco da Gama', 'Magellan', 'Cortés'],
            'Reformation': ['Martin Luther', 'John Calvin', 'Henry VIII', 'Ignatius Loyola'],
            'Enlightenment': ['Voltaire', 'Rousseau', 'Diderot', 'Kant', 'Locke'],
            'Industrial Revolution': ['James Watt', 'Adam Smith', 'Robert Owen'],
            'Modern Era': ['Napoleon', 'Abraham Lincoln', 'Otto von Bismarck', 'Queen Victoria'],
            'World Wars Era': ['Winston Churchill', 'Franklin Roosevelt', 'Adolf Hitler', 'Stalin'],
            'Contemporary': ['John F. Kennedy', 'Martin Luther King', 'Nelson Mandela', 'Margaret Thatcher']
        }
        
        # Historical places by era
        self.historical_places_by_era = {
            'Ancient Near East': ['Mesopotamia', 'Babylon', 'Assyria', 'Akkad', 'Ur', 'Nineveh'],
            'Ancient Egypt': ['Egypt', 'Memphis', 'Thebes', 'Alexandria', 'Giza', 'Nile'],
            'Ancient Greece': ['Athens', 'Sparta', 'Troy', 'Mycenae', 'Crete', 'Delphi'],
            'Hellenistic': ['Alexandria', 'Pergamon', 'Antioch', 'Byblos'],
            'Roman Republic': ['Rome', 'Carthage', 'Gaul'],
            'Roman Empire': ['Rome', 'Constantinople', 'Britannia', 'Germania'],
            'Byzantine Empire': ['Constantinople', 'Ravenna', 'Thessalonica'],
            'Medieval': ['Paris', 'London', 'Florence', 'Venice', 'Holy Roman Empire'],
            'Renaissance': ['Florence', 'Rome', 'Venice', 'Milan', 'Papal States'],
            'Age of Exploration': ['Spain', 'Portugal', 'Americas', 'New World', 'Indies'],
            'Modern': ['France', 'Britain', 'Germany', 'Austria', 'Prussia', 'Ottoman Empire']
        }
        
        # Enhanced date patterns
        self.date_patterns = [
            # Phase formats
            (r'Phase\s+\d+\s*\((-?\d{1,5})\s*(?:AD|CE)\)', 'phase_date_ce'),
            (r'Phase\s+\d+\s*\((-?\d{1,5})\s*(?:BC|BCE)\)', 'phase_date_bc'),
            
            # Standard BC dates
            (r'\b(\d{1,5})\s*BCE?\b', 'year_bc'),
            (r'\b(\d{1,5})\s*B\.C\.E?\b', 'year_bc'),
            (r'\baround\s*(\d{1,5})\s*BCE?\b', 'year_bc_approx'),
            (r'\bc\.\s*(\d{1,5})\s*BCE?\b', 'year_bc_approx'),
            
            # Century BC
            (r'\b(\d{1,3})(?:st|nd|rd|th)\s*century\s*BCE?\b', 'century_bc'),
            
            # Year ranges BC
            (r'\b(\d{1,5})\s*[-–—]\s*(\d{1,5})\s*BCE?\b', 'year_range_bc'),
            
            # CE/AD dates
            (r'\b(-?\d{1,5})\s*(?:CE|AD)\b', 'year_ce_or_ad'),
            (r'\b(-?\d{1,5})\s*A\.D\.\b', 'year_ce_or_ad'),
            
            # Plain years
            (r'\b([1-9]\d{2,3}|20[0-2][0-9])\b(?!\s*(?:BC|BCE|AD|CE))', 'year_plain'),
            
            # Century CE
            (r'\b(\d{1,2})(?:st|nd|rd|th)\s*century(?:\s*(?:CE|AD))?\b', 'century_ce'),
        ]
        
        # Known historical figures for validation/enhancement
        self.known_historical_figures = {
            'Alexander the Great', 'Julius Caesar', 'Augustus', 'Cleopatra', 'Hammurabi',
            'Sargon', 'Nebuchadnezzar', 'Cyrus', 'Darius', 'Xerxes', 'Pericles',
            'Socrates', 'Plato', 'Aristotle', 'Homer', 'Herodotus', 'Leonardo da Vinci',
            'Michelangelo', 'Machiavelli', 'Christopher Columbus', 'Napoleon', 'Charlemagne',
            'Martin Luther', 'Voltaire', 'Rousseau', 'Shakespeare', 'Dante',
            'Joan of Arc', 'Marco Polo', 'Genghis Khan', 'Saladin', 'Richard the Lionheart'
        }
        
        # Known historical places
        self.known_historical_places = {
            'Rome', 'Athens', 'Sparta', 'Alexandria', 'Constantinople', 'Babylon',
            'Memphis', 'Thebes', 'Florence', 'Venice', 'Paris', 'London',
            'Jerusalem', 'Mecca', 'Damascus', 'Baghdad', 'Cordoba', 'Toledo',
            'Prague', 'Vienna', 'Moscow', 'Novgorod', 'Kiev', 'Cairo',
            'Mesopotamia', 'Persia', 'Anatolia', 'Gaul', 'Britannia', 'Germania'
        }
    
    def extract_entities_with_spacy(self, text: str, era: str = None) -> Dict[str, List[Dict]]:
        """Extract entities using spaCy with historical context enhancement"""
        entities = {
            'PERSON': [],
            'GPE': [],      # Geopolitical entities (countries, cities, states)
            'ORG': [],      # Organizations
            'EVENT': [],    # Events
            'WORK_OF_ART': [],  # Works of art, books, songs, etc.
            'LANGUAGE': [], # Languages
            'NORP': [],     # Nationalities, religious/political groups
            'FAC': [],      # Buildings, airports, highways, bridges, etc.
            'PRODUCT': []   # Objects, vehicles, foods, etc.
        }
        
        if self.nlp:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract entities with confidence and context
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0,  # spaCy doesn't provide confidence scores by default
                    'context': text[max(0, ent.start_char-20):ent.end_char+20]
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
                elif ent.label_ in ['LANGUAGE']:
                    entities['LANGUAGE'].append(entity_info)
                elif ent.label_ in ['NORP']:
                    entities['NORP'].append(entity_info)
                elif ent.label_ in ['FAC']:
                    entities['FAC'].append(entity_info)
                elif ent.label_ in ['PRODUCT']:
                    entities['PRODUCT'].append(entity_info)
        
        # Enhance with known historical entities
        entities = self._enhance_with_historical_knowledge(entities, text, era)
        
        # Clean and deduplicate
        entities = self._clean_entities(entities)
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event
    
    def _enhance_with_historical_knowledge(self, entities: Dict, text: str, era: str = None) -> Dict:
        """Enhance spaCy entities with historical knowledge"""
        text_lower = text.lower()
        
        # Add known historical figures not caught by spaCy
        for figure in self.known_historical_figures:
            if figure.lower() in text_lower:
                # Check if already found by spaCy
                already_found = any(figure.lower() in ent['text'].lower() 
                                  for ent in entities['PERSON'])
                
                if not already_found:
                    # Find position in text
                    start_pos = text_lower.find(figure.lower())
                    if start_pos != -1:
                        entity_info = {
                            'text': figure,
                            'label': 'PERSON',
                            'start': start_pos,
                            'end': start_pos + len(figure),
                            'confidence': 0.9,  # High confidence for known figures
                            'context': text[max(0, start_pos-20):start_pos+len(figure)+20],
                            'source': 'historical_knowledge'
                        }
                        entities['PERSON'].append(entity_info)
        
        # Add known historical places
        for place in self.known_historical_places:
            if place.lower() in text_lower:
                already_found = any(place.lower() in ent['text'].lower() 
                                  for ent in entities['GPE'])
                
                if not already_found:
                    start_pos = text_lower.find(place.lower())
                    if start_pos != -1:
                        entity_info = {
                            'text': place,
                            'label': 'GPE',
                            'start': start_pos,
                            'end': start_pos + len(place),
                            'confidence': 0.9,
                            'context': text[max(0, start_pos-20):start_pos+len(place)+20],
                            'source': 'historical_knowledge'
                        }
                        entities['GPE'].append(entity_info)
        
        # Extract historical titles and positions
        title_patterns = [
            (r'\b(Emperor|Pharaoh|King|Queen|Pope|Bishop|Cardinal|Duke|Earl|Count|Baron|Prince|Princess)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'PERSON'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+the\s+(Great|Terrible|Bold|Wise|Just|Mad|Good|Bad|First|Second|Third)\b', 'PERSON'),
            (r'\b(Saint|St\.)\s+([A-Z][a-z]+)\b', 'PERSON')
        ]
        
        for pattern, entity_type in title_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                full_name = match.group().strip()
                already_found = any(full_name.lower() in ent['text'].lower() 
                                  for ent in entities[entity_type])
                
                if not already_found:
                    entity_info = {
                        'text': full_name,
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.85,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern_matching'
                    }
                    entities[entity_type].append(entity_info)
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event
    
    def _clean_entities(self, entities: Dict) -> Dict:
        """Clean and deduplicate entities"""
        for entity_type in entities:
            # Remove duplicates based on text similarity
            cleaned = []
            seen_texts = set()
            
            for entity in entities[entity_type]:
                text_lower = entity['text'].lower().strip()
                
                # Skip very short entities (likely noise)
                if len(text_lower) < 2:
                    continue
                
                # Skip if we've seen similar text
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    cleaned.append(entity)
            
            # Sort by confidence
            cleaned.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            entities[entity_type] = cleaned
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event


class HistoricalEventExtractorSpacy:
    """
    Enhanced extractor using spaCy for entity extraction with era detection
    """
    
    def __init__(self, text_file_path: str = None, spacy_model: str = "en_core_web_sm"):
        self.text_file_path = text_file_path
        self.text_content = ""
        self.events = []
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load(spacy_model)
            print(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            print(f"spaCy model '{spacy_model}' not found. Installing...")
            import subprocess
            try:
                subprocess.run([f"python", "-m", "spacy", "download", spacy_model], check=True)
                self.nlp = spacy.load(spacy_model)
                print(f"Successfully installed and loaded: {spacy_model}")
            except:
                print("Failed to install spaCy model. Using fallback entity extraction.")
                self.nlp = None
        
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
        
        # Historical eras
        self.historical_eras = {
            'Paleolithic': {'start': -2500000, 'end': -10000, 'keywords': ['paleolithic', 'old stone age', 'hunter', 'gatherer', 'cave', 'mammoth']},
            'Mesolithic': {'start': -10000, 'end': -8000, 'keywords': ['mesolithic', 'middle stone age', 'foraging', 'fishing']},
            'Neolithic': {'start': -8000, 'end': -3000, 'keywords': ['neolithic', 'new stone age', 'agriculture', 'farming', 'pottery', 'settlement']},
            'Copper Age': {'start': -5000, 'end': -3200, 'keywords': ['copper', 'chalcolithic', 'metallurgy']},
            'Bronze Age': {'start': -3200, 'end': -1200, 'keywords': ['bronze', 'bronze age', 'metal', 'alloy']},
            'Iron Age': {'start': -1200, 'end': -600, 'keywords': ['iron', 'iron age', 'iron tools', 'ironworking']},
            'Ancient Near East': {'start': -3500, 'end': -539, 'keywords': ['mesopotamia', 'babylon', 'assyria', 'sumerian', 'akkadian', 'hammurabi']},
            'Ancient Egypt': {'start': -3100, 'end': -30, 'keywords': ['egypt', 'pharaoh', 'pyramid', 'nile', 'hieroglyph', 'mummy']},
            'Ancient Greece': {'start': -800, 'end': -146, 'keywords': ['greece', 'greek', 'athens', 'sparta', 'pericles', 'socrates', 'plato', 'aristotle']},
            'Hellenistic': {'start': -336, 'end': -30, 'keywords': ['hellenistic', 'alexander', 'ptolemy', 'seleucid', 'macedon']},
            'Roman Republic': {'start': -509, 'end': -27, 'keywords': ['roman republic', 'senate', 'consul', 'cicero', 'caesar', 'republic']},
            'Roman Empire': {'start': -27, 'end': 476, 'keywords': ['roman empire', 'emperor', 'augustus', 'rome', 'legion', 'imperial']},
            'Byzantine Empire': {'start': 330, 'end': 1453, 'keywords': ['byzantine', 'constantinople', 'orthodox', 'justinian']},
            'Early Medieval': {'start': 476, 'end': 1000, 'keywords': ['early medieval', 'dark ages', 'migration', 'germanic', 'feudal']},
            'High Medieval': {'start': 1000, 'end': 1300, 'keywords': ['high medieval', 'crusades', 'cathedral', 'scholastic', 'chivalry']},
            'Late Medieval': {'start': 1300, 'end': 1500, 'keywords': ['late medieval', 'plague', 'hundred years', 'gothic', 'manuscript']},
            'Renaissance': {'start': 1400, 'end': 1600, 'keywords': ['renaissance', 'humanism', 'leonardo', 'michelangelo', 'medici', 'florence']},
            'Age of Exploration': {'start': 1400, 'end': 1650, 'keywords': ['exploration', 'columbus', 'vasco', 'magellan', 'conquistador', 'new world']},
            'Reformation': {'start': 1517, 'end': 1648, 'keywords': ['reformation', 'luther', 'protestant', 'calvin', 'counter-reformation']},
            'Baroque': {'start': 1600, 'end': 1750, 'keywords': ['baroque', 'absolutism', 'louis xiv', 'versailles']},
            'Enlightenment': {'start': 1685, 'end': 1815, 'keywords': ['enlightenment', 'voltaire', 'rousseau', 'revolution', 'reason']},
            'Industrial Revolution': {'start': 1760, 'end': 1840, 'keywords': ['industrial', 'factory', 'steam', 'railroad', 'textile']},
            'Modern Era': {'start': 1800, 'end': 1914, 'keywords': ['modern', 'nationalism', 'imperialism', 'colonialism']},
            'World Wars Era': {'start': 1914, 'end': 1945, 'keywords': ['world war', 'great war', 'hitler', 'stalin', 'fascism']},
            'Contemporary': {'start': 1945, 'end': 2100, 'keywords': ['contemporary', 'cold war', 'nuclear', 'digital', 'globalization']}
        }
        
        # Known historical figures for validation/enhancement
        self.known_historical_figures = {
            'Alexander the Great', 'Julius Caesar', 'Augustus', 'Cleopatra', 'Hammurabi',
            'Sargon', 'Nebuchadnezzar', 'Cyrus', 'Darius', 'Xerxes', 'Pericles',
            'Socrates', 'Plato', 'Aristotle', 'Homer', 'Herodotus', 'Leonardo da Vinci',
            'Michelangelo', 'Machiavelli', 'Christopher Columbus', 'Napoleon', 'Charlemagne',
            'Martin Luther', 'Voltaire', 'Rousseau', 'Shakespeare', 'Dante',
            'Joan of Arc', 'Marco Polo', 'Genghis Khan', 'Saladin', 'Richard the Lionheart'
        }
        
        # Known historical places
        self.known_historical_places = {
            'Rome', 'Athens', 'Sparta', 'Alexandria', 'Constantinople', 'Babylon',
            'Memphis', 'Thebes', 'Florence', 'Venice', 'Paris', 'London',
            'Jerusalem', 'Mecca', 'Damascus', 'Baghdad', 'Cordoba', 'Toledo',
            'Prague', 'Vienna', 'Moscow', 'Novgorod', 'Kiev', 'Cairo',
            'Mesopotamia', 'Persia', 'Anatolia', 'Gaul', 'Britannia', 'Germania'
        }
        
        # Date patterns
        self.date_patterns = [
            (r'Phase\s+\d+\s*\((-?\d{1,5})\s*(?:AD|CE)\)', 'phase_date_ce'),
            (r'Phase\s+\d+\s*\((-?\d{1,5})\s*(?:BC|BCE)\)', 'phase_date_bc'),
            (r'\b(\d{1,5})\s*BCE?\b', 'year_bc'),
            (r'\b(\d{1,5})\s*B\.C\.E?\b', 'year_bc'),
            (r'\baround\s*(\d{1,5})\s*BCE?\b', 'year_bc_approx'),
            (r'\bc\.\s*(\d{1,5})\s*BCE?\b', 'year_bc_approx'),
            (r'\b(\d{1,3})(?:st|nd|rd|th)\s*century\s*BCE?\b', 'century_bc'),
            (r'\b(\d{1,5})\s*[-–—]\s*(\d{1,5})\s*BCE?\b', 'year_range_bc'),
            (r'\b(-?\d{1,5})\s*(?:CE|AD)\b', 'year_ce_or_ad'),
            (r'\b(-?\d{1,5})\s*A\.D\.\b', 'year_ce_or_ad'),
            (r'\b([1-9]\d{2,3}|20[0-2][0-9])\b(?!\s*(?:BC|BCE|AD|CE))', 'year_plain'),
            (r'\b(\d{1,2})(?:st|nd|rd|th)\s*century(?:\s*(?:CE|AD))?\b', 'century_ce'),
        ]
    
    def extract_entities_with_spacy(self, text: str, era: str = None) -> Dict[str, List[Dict]]:
        """Extract entities using spaCy with historical context enhancement"""
        entities = {
            'PERSON': [],
            'GPE': [],      # Geopolitical entities (countries, cities, states)
            'ORG': [],      # Organizations
            'EVENT': [],    # Events
            'WORK_OF_ART': [],  # Works of art, books, songs, etc.
            'LANGUAGE': [], # Languages
            'NORP': [],     # Nationalities, religious/political groups
            'FAC': [],      # Buildings, airports, highways, bridges, etc.
            'PRODUCT': []   # Objects, vehicles, foods, etc.
        }
        
        if self.nlp:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract entities with confidence and context
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 1.0,  # spaCy doesn't provide confidence scores by default
                    'context': text[max(0, ent.start_char-20):ent.end_char+20]
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
                elif ent.label_ in ['LANGUAGE']:
                    entities['LANGUAGE'].append(entity_info)
                elif ent.label_ in ['NORP']:
                    entities['NORP'].append(entity_info)
                elif ent.label_ in ['FAC']:
                    entities['FAC'].append(entity_info)
                elif ent.label_ in ['PRODUCT']:
                    entities['PRODUCT'].append(entity_info)
        
        # Enhance with known historical entities
        entities = self._enhance_with_historical_knowledge(entities, text, era)
        
        # Clean and deduplicate
        entities = self._clean_entities(entities)
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event
    
    def _enhance_with_historical_knowledge(self, entities: Dict, text: str, era: str = None) -> Dict:
        """Enhance spaCy entities with historical knowledge"""
        text_lower = text.lower()
        
        # Add known historical figures not caught by spaCy
        for figure in self.known_historical_figures:
            if figure.lower() in text_lower:
                # Check if already found by spaCy
                already_found = any(figure.lower() in ent['text'].lower() 
                                  for ent in entities['PERSON'])
                
                if not already_found:
                    # Find position in text
                    start_pos = text_lower.find(figure.lower())
                    if start_pos != -1:
                        entity_info = {
                            'text': figure,
                            'label': 'PERSON',
                            'start': start_pos,
                            'end': start_pos + len(figure),
                            'confidence': 0.9,  # High confidence for known figures
                            'context': text[max(0, start_pos-20):start_pos+len(figure)+20],
                            'source': 'historical_knowledge'
                        }
                        entities['PERSON'].append(entity_info)
        
        # Add known historical places
        for place in self.known_historical_places:
            if place.lower() in text_lower:
                already_found = any(place.lower() in ent['text'].lower() 
                                  for ent in entities['GPE'])
                
                if not already_found:
                    start_pos = text_lower.find(place.lower())
                    if start_pos != -1:
                        entity_info = {
                            'text': place,
                            'label': 'GPE',
                            'start': start_pos,
                            'end': start_pos + len(place),
                            'confidence': 0.9,
                            'context': text[max(0, start_pos-20):start_pos+len(place)+20],
                            'source': 'historical_knowledge'
                        }
                        entities['GPE'].append(entity_info)
        
        # Extract historical titles and positions
        title_patterns = [
            (r'\b(Emperor|Pharaoh|King|Queen|Pope|Bishop|Cardinal|Duke|Earl|Count|Baron|Prince|Princess)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', 'PERSON'),
            (r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+the\s+(Great|Terrible|Bold|Wise|Just|Mad|Good|Bad|First|Second|Third)\b', 'PERSON'),
            (r'\b(Saint|St\.)\s+([A-Z][a-z]+)\b', 'PERSON')
        ]
        
        for pattern, entity_type in title_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                full_name = match.group().strip()
                already_found = any(full_name.lower() in ent['text'].lower() 
                                  for ent in entities[entity_type])
                
                if not already_found:
                    entity_info = {
                        'text': full_name,
                        'label': entity_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.85,
                        'context': text[max(0, match.start()-20):match.end()+20],
                        'source': 'pattern_matching'
                    }
                    entities[entity_type].append(entity_info)
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event
    
    def _clean_entities(self, entities: Dict) -> Dict:
        """Clean and deduplicate entities"""
        for entity_type in entities:
            # Remove duplicates based on text similarity
            cleaned = []
            seen_texts = set()
            
            for entity in entities[entity_type]:
                text_lower = entity['text'].lower().strip()
                
                # Skip very short entities (likely noise)
                if len(text_lower) < 2:
                    continue
                
                # Skip if we've seen similar text
                if text_lower not in seen_texts:
                    seen_texts.add(text_lower)
                    cleaned.append(entity)
            
            # Sort by confidence
            cleaned.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            entities[entity_type] = cleaned
        
        return entities
    
    def process_complex_text_with_eras(self, text: str) -> Dict[str, Any]:
        """Process complex text with era detection and entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with era information
        event = self.create_historical_event_with_era(sentence_info)
        
        return event
    
    def create_historical_event_with_era(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with era information and entity-based figures"""
        
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with era-specific knowledge
        entities = self.extract_entities_by_era(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military',
                'political': 'political',
                'discovery': 'cultural',
                'disaster': 'natural',
                'cultural': 'cultural',
                'migration': 'social',
                'technology': 'technological',
                'trade': 'economic',
                'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with era information
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                
                # Textual date field
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown",
                "month": "unknown",
                "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0,
                "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # Enhanced entity fields
                "keyFigures": entities['PERSON'][:5],  # Up to 5 key figures
                "involvedParties": entities['ORG'][:3],
                "places": entities['GPE'][:5],
                "cultures": entities['CULTURE'][:3],
                "historicalEvents": entities['EVENT'][:3],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using automated analysis with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "automated_text_analysis_with_eras",
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()])
            }
        }
        
        return historical_event
    
    def determine_historical_era(self, year: int, text: str = "") -> str:
        """Determine historical era"""
        text_lower = text.lower()
        keyword_matches = []
        
        for era_name, era_info in self.historical_eras.items():
            if era_info['start'] <= year <= era_info['end']:
                keyword_score = 0
                for keyword in era_info['keywords']:
                    if keyword in text_lower:
                        keyword_score += 1
                
                if keyword_score > 0:
                    keyword_matches.append((era_name, keyword_score))
        
        if keyword_matches:
            keyword_matches.sort(key=lambda x: x[1], reverse=True)
            return keyword_matches[0][0]
        
        # Fallback classification
        if year < -10000:
            return "Paleolithic"
        elif year < -8000:
            return "Mesolithic" 
        elif year < -3000:
            return "Neolithic"
        elif year < -1200:
            return "Bronze Age"
        elif year < -600:
            return "Iron Age"
        elif year < -336:
            return "Ancient Greece"
        elif year < -27:
            return "Hellenistic"
        elif year < 476:
            return "Roman Empire"
        elif year < 1000:
            return "Early Medieval"
        elif year < 1300:
            return "High Medieval"
        elif year < 1500:
            return "Late Medieval"
        elif year < 1600:
            return "Renaissance"
        elif year < 1750:
            return "Early Modern"
        elif year < 1815:
            return "Enlightenment"
        elif year < 1914:
            return "Modern Era"
        elif year < 1945:
            return "World Wars Era"
        else:
            return "Contemporary"
    
    def extract_all_dates_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract all dates"""
        all_dates = []
        
        for pattern, pattern_type in self.date_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            
            for match in matches:
                date_info = {
                    'pattern_type': pattern_type,
                    'matched_text': match.group(),
                    'full_context': text[max(0, match.start()-50):match.end()+50],
                    'year': None,
                    'year_start': None,
                    'year_end': None,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'era': 'CE',
                    'confidence': 1.0
                }
                
                try:
                    if pattern_type == 'phase_date_ce':
                        year = int(match.group(1))
                        date_info['year'] = year
                        date_info['era'] = 'CE'
                    elif pattern_type == 'phase_date_bc':
                        year = int(match.group(1))
                        if year < 0:
                            date_info['year'] = year
                            date_info['era'] = 'BC'
                        else:
                            date_info['year'] = -year
                            date_info['era'] = 'BC'
                    elif pattern_type in ['year_bc', 'year_bc_approx']:
                        year = int(match.group(1))
                        date_info['year'] = -year
                        date_info['era'] = 'BC'
                    elif pattern_type == 'century_bc':
                        century = int(match.group(1))
                        date_info['year_start'] = -century * 100
                        date_info['year_end'] = -(century - 1) * 100 - 1
                        date_info['year'] = date_info['year_start'] + 50
                        date_info['era'] = 'BC'
                    elif pattern_type == 'year_range_bc':
                        year1, year2 = int(match.group(1)), int(match.group(2))
                        date_info['year_start'] = -max(year1, year2)
                        date_info['year_end'] = -min(year1, year2)
                        date_info['year'] = (date_info['year_start'] + date_info['year_end']) / 2
                        date_info['era'] = 'BC'
                    elif pattern_type == 'year_ce_or_ad':
                        year = int(match.group(1))
                        if year < 0:
                            date_info['year'] = year
                            date_info['era'] = 'BC'
                            date_info['confidence'] = 0.8
                        else:
                            date_info['year'] = year
                            date_info['era'] = 'CE'
                    elif pattern_type == 'year_plain':
                        year = int(match.group(1))
                        date_info['year'] = year
                        date_info['era'] = 'CE'
                    elif pattern_type == 'century_ce':
                        century = int(match.group(1))
                        date_info['year_start'] = (century - 1) * 100 + 1
                        date_info['year_end'] = century * 100
                        date_info['year'] = date_info['year_start'] + 50
                        date_info['era'] = 'CE'
                    
                    all_dates.append(date_info)
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing date '{match.group()}': {e}")
                    continue
        
        # Remove overlaps
        all_dates.sort(key=lambda x: (x['start_pos'], -x['confidence'], -len(x['matched_text'])))
        filtered_dates = []
        
        for current in all_dates:
            overlap = False
            for existing in filtered_dates:
                if (current['start_pos'] < existing['end_pos'] and 
                    current['end_pos'] > existing['start_pos']):
                    overlap = True
                    break
            if not overlap:
                filtered_dates.append(current)
        
        return filtered_dates
    
    def format_date_for_json(self, year: float) -> str:
        """Format year as ISO date string"""
        if year < 0:
            abs_year = abs(int(year))
            return f"-{abs_year:04d}-01-01T00:00:00Z"
        else:
            int_year = max(1, int(year))
            return f"{int_year:04d}-01-01T00:00:00Z"
    
    def create_historical_event_with_spacy(self, sentence_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured event with spaCy entity extraction"""
        event_id = str(uuid.uuid4())
        
        # Extract primary year and all dates
        primary_year = 1500
        primary_era = 'CE'
        all_dates = []
        date_text_descriptions = []
        
        if sentence_info['temporal_info']:
            primary_temporal = sentence_info['temporal_info'][0]
            if primary_temporal['year'] is not None:
                primary_year = primary_temporal['year']
                primary_era = primary_temporal['era']
            
            for temporal in sentence_info['temporal_info']:
                if temporal['year'] is not None:
                    date_entry = {
                        'year': temporal['year'],
                        'era': temporal['era'],
                        'confidence': temporal.get('confidence', 1.0),
                        'text': temporal['matched_text'],
                        'context': temporal.get('full_context', '')
                    }
                    all_dates.append(date_entry)
                    
                    if temporal['year'] < 0:
                        date_text_descriptions.append(f"{abs(int(temporal['year']))} BC")
                    else:
                        date_text_descriptions.append(f"{int(temporal['year'])} CE")
        
        all_dates.sort(key=lambda x: x['year'])
        
        # Determine historical era
        historical_era = self.determine_historical_era(primary_year, sentence_info['text'])
        
        # Extract entities with spaCy
        entities = self.extract_entities_with_spacy(sentence_info['text'], historical_era)
        
        # Determine event type
        event_type = 'cultural'
        if sentence_info['event_types']:
            type_mapping = {
                'war': 'military', 'political': 'political', 'discovery': 'cultural',
                'disaster': 'natural', 'cultural': 'cultural', 'migration': 'social',
                'technology': 'technological', 'trade': 'economic', 'religious': 'religious'
            }
            event_type = type_mapping.get(sentence_info['event_types'][0], 'cultural')
        
        # Create event name
        event_name = sentence_info['text'][:100].strip()
        if len(sentence_info['text']) > 100:
            event_name += "..."
        
        # Extract location from GPE entities
        location = "Unknown"
        if entities['GPE']:
            location = entities['GPE'][0]['text']
        
        # Create temporal pattern with era
        if len(all_dates) > 1:
            date_range = f"{abs(int(all_dates[0]['year']))} {all_dates[0]['era']} to {abs(int(all_dates[-1]['year']))} {all_dates[-1]['era']}"
            temporal_pattern = f"{historical_era} period - Multi-phase event spanning {date_range}"
        else:
            temporal_pattern = f"{historical_era} period - Event occurring around {abs(int(primary_year))} {primary_era}"
        
        # Create precursor events
        earliest_year = min([d['year'] for d in all_dates]) if all_dates else primary_year
        precursor_year = earliest_year - (100 if earliest_year < 0 else 50)
        precursor_events = [{
            "description": f"{historical_era} period developments leading to events around {abs(int(earliest_year))} {primary_era}",
            "month_parsed": None,
            "season_parsed": None,
            "year_parsed": float(precursor_year)
        }]
        
        # Create structured event with spaCy entities
        historical_event = {
            "event_id": event_id,
            "source_text": sentence_info['text'],
            "summary": f"{historical_era} period event: {event_name}",
            "precursor_events": precursor_events,
            "historian_annotation": {
                "eventId": str(uuid.uuid4()),
                "eventName": event_name,
                "eventType": event_type,
                
                # Numeric date fields
                "eventYear": int(primary_year),
                "eventYearStart": int(all_dates[0]['year']) if all_dates else int(primary_year),
                "eventYearEnd": int(all_dates[-1]['year']) if len(all_dates) > 1 else int(primary_year),
                
                # Era information
                "historicalEra": historical_era,
                "eraDescription": f"{historical_era} period ({self.historical_eras.get(historical_era, {}).get('start', 'Unknown')} - {self.historical_eras.get(historical_era, {}).get('end', 'Unknown')})",
                
                # Multiple dates array
                "dates": all_dates,
                "dateText": "; ".join(date_text_descriptions) if date_text_descriptions else f"{abs(int(primary_year))} {primary_era}",
                
                # Legacy fields
                "eventDate": self.format_date_for_json(primary_year),
                "season": "unknown", "month": "unknown", "dayOfWeek": "unknown",
                
                "temporalPattern": temporal_pattern,
                "eventLocation": location,
                "latitude": 0.0, "longitude": 0.0,
                "geopoliticalContext": f"{historical_era} period context of {location}",
                "precursors": [f"{historical_era} period events leading to {event_name}"],
                
                # spaCy-enhanced entity fields
                "keyFigures": [ent['text'] for ent in entities['PERSON'][:5]],
                "places": [ent['text'] for ent in entities['GPE'][:5]],
                "organizations": [ent['text'] for ent in entities['ORG'][:3]],
                "events": [ent['text'] for ent in entities['EVENT'][:3]],
                "worksOfArt": [ent['text'] for ent in entities['WORK_OF_ART'][:3]],
                "languages": [ent['text'] for ent in entities['LANGUAGE'][:3]],
                "nationalities": [ent['text'] for ent in entities['NORP'][:3]],
                "facilities": [ent['text'] for ent in entities['FAC'][:3]],
                "products": [ent['text'] for ent in entities['PRODUCT'][:3]],
                
                # Detailed entity information with confidence scores
                "entitiesDetailed": {
                    "persons": entities['PERSON'][:5],
                    "places": entities['GPE'][:5],
                    "organizations": entities['ORG'][:3],
                    "events": entities['EVENT'][:3],
                    "worksOfArt": entities['WORK_OF_ART'][:3],
                    "languages": entities['LANGUAGE'][:3],
                    "nationalities": entities['NORP'][:3],
                    "facilities": entities['FAC'][:3],
                    "products": entities['PRODUCT'][:3]
                },
                
                # Legacy fields for compatibility
                "involvedParties": [ent['text'] for ent in entities['ORG'][:3]],
                
                "description": sentence_info['text'],
                "outcomes": [f"Immediate results of this {historical_era} period event"],
                "consequences": [f"Long-term {historical_era} period impact"],
                "significance": f"Significant {event_type} event during the {historical_era} period in {location}",
                "culturalImpact": f"{historical_era} period cultural impact",
                "economicImpact": f"{historical_era} period economic impact",
                "socialImpact": f"{historical_era} period social impact",
                "environmentalImpact": "Environmental impact to be assessed",
                "modernEquivalent": "Modern equivalent to be determined",
                "historicalPattern": f"Part of {event_type} patterns during the {historical_era}",
                "goldsteinScale": 5.0,
                "sources": ["Extracted from historical text using spaCy NLP"],
                "historiographicalDebates": f"Scholarly debates about {historical_era} period events",
                "methodologicalChallenges": "Extracted using spaCy NLP with era detection",
                "futurePredictions": ["Future relevance to be assessed"],
                "confidenceScoreFuturePredictions": 50,
                "lessonsFuture": f"Historical lessons from {historical_era} period events"
            },
            "extraction_metadata": {
                "extraction_method": "spacy_nlp_with_eras",
                "spacy_model": self.nlp.meta['name'] if self.nlp else 'none',
                "confidence_score": sentence_info['event_score'],
                "sentence_id": sentence_info.get('sentence_id', 0),
                "word_count": sentence_info['word_count'],
                "entities_found": {k: len(v) for k, v in entities.items()},
                "temporal_references": len(sentence_info['temporal_info']),
                "primary_era": primary_era,
                "primary_year": primary_year,
                "historical_era": historical_era,
                "total_dates_found": len(all_dates),
                "era_keywords_matched": len([kw for kw in self.historical_eras.get(historical_era, {}).get('keywords', []) if kw in sentence_info['text'].lower()]),
                "spacy_entities_found": sum(len(v) for v in entities.values()),
                "enhanced_entities_found": sum(1 for entity_list in entities.values() for entity in entity_list if entity.get('source') in ['historical_knowledge', 'pattern_matching'])
            }
        }
        
        return historical_event
    
    def process_complex_text_with_spacy(self, text: str) -> Dict[str, Any]:
        """Process complex text with spaCy entity extraction"""
        
        # Extract all dates
        temporal_info = self.extract_all_dates_from_text(text)
        
        # Calculate event score
        event_score = 0
        event_types = []
        
        for event_type, pattern in self.event_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                event_score += 1
                event_types.append(event_type)
        
        event_score += len(temporal_info) * 0.7
        
        # Create sentence info
        sentence_info = {
            'sentence_id': 0,
            'text': text,
            'event_score': event_score,
            'event_types': event_types,
            'temporal_info': temporal_info,
            'word_count': len(text.split())
        }
        
        # Create structured event with spaCy
        event = self.create_historical_event_with_spacy(sentence_info)
        
        return event


# Test spaCy entity extraction
def test_spacy_extraction():
    print("Testing spaCy Entity Extraction with Historical Events")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            'text': "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance, revolutionizing artistic techniques with oil painting and sfumato.",
            'name': 'Renaissance Art'
        },
        {
            'text': "Alexander the Great conquered the Persian Empire between 334 and 323 BCE, spreading Hellenistic culture from Macedonia to India and founding Alexandria in Egypt.",
            'name': 'Hellenistic Conquest'
        },
        {
            'text': "The Roman Senate assassinated Julius Caesar on the Ides of March 44 BC in the Theater of Pompey, leading to civil wars and the end of the Roman Republic.",
            'name': 'Roman Political Crisis'
        }
    ]
    
    extractor = HistoricalEventExtractorSpacy()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 50)
        
        event = extractor.process_complex_text_with_spacy(test_case['text'])
        annotation = event['historian_annotation']
        metadata = event['extraction_metadata']
        
        print(f"Historical Era: {annotation['historicalEra']}")
        print(f"Primary Year: {annotation['eventYear']}")
        print(f"spaCy Model: {metadata['spacy_model']}")
        print()
        
        print("SPACY ENTITIES EXTRACTED:")
        print(f"  Persons: {annotation['keyFigures']}")
        print(f"  Places: {annotation['places']}")
        print(f"  Organizations: {annotation['organizations']}")
        print(f"  Works of Art: {annotation['worksOfArt']}")
        print(f"  Languages: {annotation['languages']}")
        print(f"  Nationalities: {annotation['nationalities']}")
        print()
        
        print("DETAILED ENTITY INFO:")
        for j, person in enumerate(annotation['entitiesDetailed']['persons'][:3]):
            print(f"  Person {j+1}: '{person['text']}' (confidence: {person['confidence']}, source: {person.get('source', 'spacy')})")
        
        for j, place in enumerate(annotation['entitiesDetailed']['places'][:3]):
            print(f"  Place {j+1}: '{place['text']}' (confidence: {place['confidence']}, source: {place.get('source', 'spacy')})")
        
        print(f"\nTotal spaCy Entities: {metadata['spacy_entities_found']}")
        print(f"Enhanced Entities: {metadata['enhanced_entities_found']}")
        
        # Save individual test case
        filename = f"spacy_test_{i}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(event, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {filename}")
    
    print(f"\nAll {len(test_cases)} spaCy test cases processed successfully!")


# Run the spaCy tests
if __name__ == "__main__":
    spacy_results = test_spacy_extraction()
