#!/usr/bin/env python3

import json
from typing import Dict, List, Any

class BraveSearchQueryTemplates:
    """Comprehensive query templates for gathering historian annotation fields"""
    
    def __init__(self):
        self.queries = self.generate_brave_search_queries()
        self.template = self.create_search_query_template()
    
    def generate_brave_search_queries(self) -> Dict[str, List[str]]:
        """Generate comprehensive Brave Search queries to populate all historian_annotation fields"""
        
        search_queries = {}
        
        # 1. EVENT IDENTIFICATION QUERIES
        search_queries['event_identification'] = [
            "{event_name} historical event date year",
            "{event_name} what happened when where",
            "{event_name} historical significance timeline",
            "{event_name} event type political cultural military economic",
            "{event_name} chronology sequence of events",
            "{event_name} historical pattern recurring theme",
            "{event_name} cyclical historical trends",
            "{event_name} temporal context period era"
        ]
        
        # 2. LOCATION AND GEOGRAPHY QUERIES  
        search_queries['location_data'] = [
            "{event_name} location where happened geographic",
            "{event_name} city country region coordinates",
            "{location} latitude longitude coordinates GPS",
            "{location} geographic location map position",
            "{event_name} {location} geopolitical context political situation",
            "{location} {era} political boundaries territories",
            "{location} {year} government political system",
            "{event_name} international relations diplomatic context"
        ]
        
        # 3. HISTORICAL CONTEXT QUERIES
        search_queries['historical_context'] = [
            "{event_name} causes precursors background leading to",
            "{event_name} historical antecedents previous events",
            "{event_name} underlying factors conditions",
            "events leading to {event_name} chronological sequence",
            "{event_name} key figures important people leaders",
            "{event_name} main participants historical figures",
            "{key_figure} biography role in {event_name}",
            "{key_figure} birth death dates occupation nationality",
            "{event_name} involved parties organizations institutions",
            "{event_name} stakeholders participants groups",
            "{event_name} political parties factions sides",
            "{event_name} historical pattern similar events",
            "{event_name} recurring themes historical cycles",
            "historical precedents for {event_name}"
        ]
        
        # 4. EVENT ANALYSIS QUERIES
        search_queries['event_analysis'] = [
            "{event_name} detailed description what happened",
            "{event_name} step by step sequence chronology",
            "{event_name} primary sources contemporary accounts",
            "{event_name} historical narrative story",
            "{event_name} immediate results outcomes effects",
            "{event_name} short term consequences direct impact",
            "{event_name} what happened after immediate aftermath",
            "{event_name} long term consequences effects legacy",
            "{event_name} historical impact significance importance",
            "{event_name} lasting effects modern relevance",
            "{event_name} why important historical significance",
            "{event_name} impact on history watershed moment",
            "{event_name} turning point historical importance"
        ]
        
        # 5. IMPACT ANALYSIS QUERIES
        search_queries['impact_analysis'] = [
            "{event_name} cultural impact society culture",
            "{event_name} effect on art literature religion",
            "{event_name} cultural changes social transformation",
            "{event_name} influence on customs traditions values",
            "{event_name} economic impact financial consequences",
            "{event_name} effect on trade commerce economy",
            "{event_name} economic changes costs benefits",
            "{event_name} financial implications monetary effects",
            "{event_name} social impact society people",
            "{event_name} effect on social structure classes",
            "{event_name} demographic changes population",
            "{event_name} social transformation community impact",
            "{event_name} environmental impact ecological effects",
            "{event_name} effect on environment nature",
            "{event_name} environmental consequences damage",
            "{event_name} ecological changes natural impact",
            "{event_name} modern equivalent contemporary parallel",
            "{event_name} relevance today current events",
            "{event_name} modern comparison similar events",
            "{event_name} lessons for today modern application"
        ]
        
        # 6. RESEARCH METADATA QUERIES
        search_queries['research_metadata'] = [
            "{event_name} primary sources historical documents",
            "{event_name} secondary sources scholarly articles",
            "{event_name} academic research bibliography",
            "{event_name} historical records archives manuscripts",
            "{event_name} historiographical debate scholarly controversy",
            "{event_name} different interpretations historical debate",
            "{event_name} academic disagreement scholarly dispute",
            "{event_name} historical revisionism new interpretations",
            "{event_name} research challenges methodological issues",
            "{event_name} historical evidence problems limitations",
            "{event_name} source reliability authenticity questions",
            "{event_name} historical methodology research methods",
            "{event_name} lessons learned future implications",
            "{event_name} predictions future trends patterns",
            "{event_name} contemporary relevance modern lessons",
            "{event_name} historical wisdom insights guidance"
        ]
        
        return search_queries
    
    def create_search_query_template(self) -> Dict[str, Any]:
        """Create a practical template for using Brave Search queries"""
        
        template = {
            "instructions": "Replace placeholders with actual event details from your text",
            "placeholders": {
                "{event_name}": "The specific name of the historical event",
                "{location}": "Geographic location where event occurred", 
                "{era}": "Historical era/period (e.g., Renaissance, Roman Empire)",
                "{year}": "Specific year of the event",
                "{key_figure}": "Name of important historical figure"
            },
            
            "priority_queries": {
                "essential_first": [
                    "{event_name} historical event date year location",
                    "{event_name} what happened when where why",
                    "{event_name} key figures important people",
                    "{event_name} causes precursors background",
                    "{event_name} outcomes consequences effects"
                ],
                
                "detailed_second": [
                    "{event_name} cultural impact society",
                    "{event_name} economic impact financial effects", 
                    "{event_name} political impact government",
                    "{event_name} social impact people community",
                    "{event_name} historical significance importance"
                ],
                
                "research_third": [
                    "{event_name} primary sources historical documents",
                    "{event_name} scholarly debate interpretation",
                    "{event_name} modern relevance contemporary parallel",
                    "{event_name} long term legacy lasting effects"
                ]
            },
            
            "search_strategy": {
                "step_1": "Start with essential queries to get basic event information",
                "step_2": "Use detailed queries to gather impact and significance data",
                "step_3": "Research queries for academic depth and source material",
                "step_4": "Location-specific queries for geographic and coordinate data",
                "step_5": "Biography queries for each key figure identified"
            }
        }
        
        return template
    
    def get_priority_queries(self, event_name: str, location: str = "", era: str = "", year: str = "", key_figure: str = "") -> List[str]:
        """Get priority queries with placeholders replaced"""
        queries = []
        
        # Essential queries
        for query_template in self.template['priority_queries']['essential_first']:
            query = query_template.format(
                event_name=event_name,
                location=location,
                era=era,
                year=year,
                key_figure=key_figure
            )
            queries.append(query)
        
        # Detailed queries
        for query_template in self.template['priority_queries']['detailed_second']:
            query = query_template.format(
                event_name=event_name,
                location=location,
                era=era,
                year=year,
                key_figure=key_figure
            )
            queries.append(query)
        
        return queries
    
    def get_queries_for_category(self, category: str, event_name: str, location: str = "", era: str = "", year: str = "", key_figure: str = "") -> List[str]:
        """Get queries for a specific category with placeholders replaced"""
        if category not in self.queries:
            return []
        
        queries = []
        for query_template in self.queries[category]:
            query = query_template.format(
                event_name=event_name,
                location=location,
                era=era,
                year=year,
                key_figure=key_figure
            )
            queries.append(query)
        
        return queries
    
    def get_location_queries(self, location: str, era: str = "", year: str = "") -> List[str]:
        """Get location-specific queries"""
        location_queries = [
            "{location} latitude longitude coordinates",
            "{location} geographic location map",
            "{location} {era} political boundaries", 
            "{location} historical significance",
            "{location} demographics population {year}",
            "{location} economy trade {era}",
            "{location} cultural center {era}",
            "{location} modern name current status"
        ]
        
        queries = []
        for query_template in location_queries:
            query = query_template.format(
                location=location,
                era=era,
                year=year
            )
            queries.append(query)
        
        return queries
    
    def get_biography_queries(self, person: str, event_name: str = "") -> List[str]:
        """Get biography queries for key figures"""
        biography_queries = [
            "{person} birth death dates biography",
            "{person} occupation role profession",
            "{person} nationality origin background", 
            "{person} achievements accomplishments",
            "{person} role in {event_name}",
            "{person} historical significance impact",
            "{person} contemporary accounts descriptions",
            "{person} modern assessment historical importance"
        ]
        
        queries = []
        for query_template in biography_queries:
            query = query_template.format(
                person=person,
                event_name=event_name
            )
            queries.append(query)
        
        return queries
