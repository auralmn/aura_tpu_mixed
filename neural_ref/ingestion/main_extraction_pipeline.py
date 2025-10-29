#!/usr/bin/env python3

import sys
import os
import json
from datetime import datetime
from typing import List, Dict
sys.path.append('/Users/nick/PycharmProjects/AURASNN')

from comprehensive_historical_extractor import ComprehensiveHistoricalExtractor

class MainExtractionPipeline:
    """
    Main data extraction pipeline using comprehensive historical extractor
    """
    
    def __init__(self, 
                 wikipedia_jsonl_path: str = None,
                 brave_api_key: str = None,
                 output_dir: str = "extracted_data"):
        
        self.output_dir = output_dir
        self.extractor = ComprehensiveHistoricalExtractor(
            wikipedia_jsonl_path=wikipedia_jsonl_path,
            brave_api_key=brave_api_key
        )
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("🚀 Main Extraction Pipeline Initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📚 Wikipedia articles loaded: {len(self.extractor.wikipedia_cache)}")
        print(f"🌐 Brave Search available: {brave_api_key is not None}")
    
    def process_text_file(self, file_path: str) -> Dict:
        """Process a text file containing historical content"""
        print(f"\n📖 Processing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into sentences or paragraphs for processing
            sentences = self._split_into_sentences(content)
            
            results = []
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 20:  # Skip very short sentences
                    print(f"  Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}...")
                    
                    result = self.extractor.process_comprehensive_text(sentence)
                    results.append(result)
            
            # Save results
            output_file = os.path.join(self.output_dir, f"extracted_{os.path.basename(file_path)}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved results to: {output_file}")
            return {
                'file': file_path,
                'sentences_processed': len(results),
                'output_file': output_file,
                'results': results
            }
            
        except Exception as e:
            print(f"❌ Error processing file {file_path}: {e}")
            return {'error': str(e)}
    
    def process_text_string(self, text: str, name: str = "text") -> Dict:
        """Process a text string directly"""
        print(f"\n📝 Processing text: {name}")
        print(f"Text: {text[:100]}...")
        
        result = self.extractor.process_comprehensive_text(text)
        
        # Save result
        output_file = os.path.join(self.output_dir, f"extracted_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved result to: {output_file}")
        return result
    
    def process_batch_texts(self, texts: List[str], names: List[str] = None) -> List[Dict]:
        """Process multiple texts in batch"""
        if names is None:
            names = [f"text_{i+1}" for i in range(len(texts))]
        
        print(f"\n📦 Processing batch of {len(texts)} texts")
        
        results = []
        for i, (text, name) in enumerate(zip(texts, names)):
            print(f"  Processing {i+1}/{len(texts)}: {name}")
            result = self.process_text_string(text, name)
            results.append(result)
        
        return results
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def generate_summary_report(self, results: List[Dict]) -> Dict:
        """Generate a summary report of all extractions"""
        total_entities = 0
        total_enriched = 0
        total_wikipedia_sources = 0
        total_web_sources = 0
        eras_found = set()
        event_types_found = set()
        
        for result in results:
            if 'enrichment_stats' in result:
                stats = result['enrichment_stats']
                total_entities += stats['total_entities']
                total_enriched += stats['entities_enriched']
                total_wikipedia_sources += stats['wikipedia_sources']
                total_web_sources += stats['web_sources']
            
            if 'historical_context' in result:
                context = result['historical_context']
                eras_found.add(context['primary_era'])
                event_types_found.add(context['event_type'])
        
        summary = {
            'extraction_summary': {
                'total_texts_processed': len(results),
                'total_entities_found': total_entities,
                'total_entities_enriched': total_enriched,
                'wikipedia_sources_used': total_wikipedia_sources,
                'web_sources_used': total_web_sources,
                'enrichment_rate': (total_enriched / total_entities * 100) if total_entities > 0 else 0
            },
            'historical_analysis': {
                'eras_detected': list(eras_found),
                'event_types_found': list(event_types_found),
                'era_coverage': len(eras_found),
                'event_type_coverage': len(event_types_found)
            },
            'system_info': {
                'wikipedia_articles_available': len(self.extractor.wikipedia_cache),
                'brave_search_available': self.extractor.brave_api_key is not None,
                'spacy_available': self.extractor.nlp is not None,
                'extraction_timestamp': datetime.now().isoformat()
            }
        }
        
        return summary

def demo_extraction_pipeline():
    """Demonstrate the main extraction pipeline"""
    print("🎯 Main Extraction Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    wikipedia_path = "/Volumes/Others2/wikipedia/enwiki_namespace_0"
    brave_api_key = os.getenv('BRAVE_API_KEY')
    
    pipeline = MainExtractionPipeline(
        wikipedia_jsonl_path=wikipedia_path if os.path.exists(wikipedia_path) else None,
        brave_api_key=brave_api_key
    )
    
    # Demo texts
    demo_texts = [
        "Leonardo da Vinci painted the Mona Lisa around 1503 in Florence during the Italian Renaissance.",
        "Alexander the Great conquered the Persian Empire between 334 and 323 BCE, spreading Hellenistic culture from Macedonia to India.",
        "Julius Caesar crossed the Rubicon in 49 BC, leading to civil war in the Roman Republic.",
        "Napoleon Bonaparte was crowned Emperor of France in 1804, marking the height of the Napoleonic Empire.",
        "Isaac Newton formulated the laws of motion and universal gravitation in the 17th century during the Scientific Revolution."
    ]
    
    # Process texts
    results = pipeline.process_batch_texts(demo_texts)
    
    # Generate summary report
    summary = pipeline.generate_summary_report(results)
    
    # Save summary
    summary_file = os.path.join(pipeline.output_dir, "extraction_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 EXTRACTION SUMMARY:")
    print(f"   Texts Processed: {summary['extraction_summary']['total_texts_processed']}")
    print(f"   Total Entities: {summary['extraction_summary']['total_entities_found']}")
    print(f"   Entities Enriched: {summary['extraction_summary']['total_entities_enriched']}")
    print(f"   Enrichment Rate: {summary['extraction_summary']['enrichment_rate']:.1f}%")
    print(f"   Wikipedia Sources: {summary['extraction_summary']['wikipedia_sources_used']}")
    print(f"   Web Sources: {summary['extraction_summary']['web_sources_used']}")
    print(f"   Intelligent Queries: {summary['extraction_summary'].get('intelligent_queries_used', 0)}")
    print(f"   Search Categories: {summary['extraction_summary'].get('search_categories_found', 0)}")
    print(f"   Eras Detected: {', '.join(summary['historical_analysis']['eras_detected'])}")
    print(f"   Event Types: {', '.join(summary['historical_analysis']['event_types_found'])}")
    print(f"\n📁 Summary saved to: {summary_file}")
    print(f"📁 All results in: {pipeline.output_dir}")

if __name__ == "__main__":
    demo_extraction_pipeline()
