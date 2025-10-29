# SPDX-License-Identifier: Apache-2.0
"""
AURA Dataset Integration: Unified Multi-Modal Learning System
- Historical timeline engine integration
- Grammar pattern integration
- Instruction dataset integration
- Self-awareness engine coordination
- Cross-dataset learning and adaptation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from collections import defaultdict

from aura.neural.self_awareness import SelfAwarenessEngine
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from historical_timeline_engine import HistoricalTimelineEngine
from aura.datasets.grammar_integration import GrammarIntegrationEngine
from aura.datasets.instruction_integration import InstructionFollowingEngine

class MultiModalDatasetIntegrator:
    """
    Unified integrator for all AURA dataset types
    """
    
    def __init__(self, self_awareness_engine: SelfAwarenessEngine, device: str = 'mps'):
        self.engine = self_awareness_engine
        self.device = device
        
        # Initialize all dataset engines
        self.historical_engine = HistoricalTimelineEngine(self_awareness_engine, device)
        self.grammar_engine = GrammarIntegrationEngine(self_awareness_engine, device)
        self.instruction_engine = InstructionFollowingEngine(self_awareness_engine, device)
        
        # Integration statistics
        self.integration_stats = {
            'datasets_loaded': 0,
            'total_samples_processed': 0,
            'cross_dataset_learning_episodes': 0,
            'integration_quality_scores': [],
            'domain_transfer_success_rate': 0.0
        }
        
        # Cross-dataset learning state
        self.cross_dataset_memory = {
            'historical_patterns': [],
            'grammar_patterns': [],
            'instruction_patterns': [],
            'domain_mappings': {},
            'quality_correlations': {}
        }
    
    def load_all_datasets(self, 
                         historical_data_path: str = None,
                         grammar_data_path: str = None,
                         instruction_data_path: str = None,
                         max_samples_per_dataset: int = 10000) -> Dict[str, bool]:
        """Load all available datasets"""
        
        results = {}
        
        # Load historical dataset
        if historical_data_path and Path(historical_data_path).exists():
            try:
                success = self.historical_engine.register_historical_dataset(
                    historical_data_path, 
                    max_samples=max_samples_per_dataset
                )
                results['historical'] = success
                if success:
                    self.integration_stats['datasets_loaded'] += 1
                    print(f"✓ Historical dataset loaded: {historical_data_path}")
            except Exception as e:
                logging.error(f"Failed to load historical dataset: {e}")
                results['historical'] = False
        else:
            results['historical'] = False
        
        # Load grammar patterns
        if grammar_data_path and Path(grammar_data_path).exists():
            try:
                success = self.grammar_engine.load_grammar_patterns(
                    grammar_data_path,
                    max_patterns_per_domain=max_samples_per_dataset // 8  # 8 domains
                )
                results['grammar'] = success
                if success:
                    self.integration_stats['datasets_loaded'] += 1
                    print(f"✓ Grammar patterns loaded: {grammar_data_path}")
            except Exception as e:
                logging.error(f"Failed to load grammar patterns: {e}")
                results['grammar'] = False
        else:
            results['grammar'] = False
        
        # Load instruction dataset
        if instruction_data_path and Path(instruction_data_path).exists():
            try:
                success = self.instruction_engine.load_instruction_dataset(
                    instruction_data_path,
                    max_samples=max_samples_per_dataset
                )
                results['instruction'] = success
                if success:
                    self.integration_stats['datasets_loaded'] += 1
                    print(f"✓ Instruction dataset loaded: {instruction_data_path}")
            except Exception as e:
                logging.error(f"Failed to load instruction dataset: {e}")
                results['instruction'] = False
        else:
            results['instruction'] = False
        
        # Update total samples processed
        self._update_sample_counts()
        
        return results
    
    def _update_sample_counts(self):
        """Update total sample counts from all engines"""
        total_samples = 0
        
        if hasattr(self.historical_engine, 'datasets'):
            for dataset in self.historical_engine.datasets.values():
                total_samples += len(dataset)
        
        if hasattr(self.grammar_engine, 'pattern_dataset') and self.grammar_engine.pattern_dataset:
            total_samples += len(self.grammar_engine.pattern_dataset)
        
        if hasattr(self.instruction_engine, 'instruction_dataset') and self.instruction_engine.instruction_dataset:
            total_samples += len(self.instruction_engine.instruction_dataset)
        
        self.integration_stats['total_samples_processed'] = total_samples
    
    def analyze_multi_modal_input(self, 
                                 text: str,
                                 historical_context: Optional[Dict] = None,
                                 grammar_analysis: bool = True,
                                 instruction_analysis: bool = True) -> Dict[str, Any]:
        """Analyze input across all modalities"""
        
        results = {
            'text': text,
            'historical_analysis': None,
            'grammar_analysis': None,
            'instruction_analysis': None,
            'cross_modal_insights': None,
            'integrated_awareness': None
        }
        
        # Historical analysis
        if historical_context:
            try:
                hist_analysis = self.historical_engine.analyze_historical_event(historical_context)
                results['historical_analysis'] = hist_analysis
            except Exception as e:
                logging.warning(f"Historical analysis failed: {e}")
        
        # Grammar analysis
        if grammar_analysis:
            try:
                grammar_analysis = self.grammar_engine.analyze_text_grammar(text)
                results['grammar_analysis'] = grammar_analysis
            except Exception as e:
                logging.warning(f"Grammar analysis failed: {e}")
        
        # Instruction analysis
        if instruction_analysis:
            try:
                instruction_analysis = self.instruction_engine.analyze_instruction(text)
                results['instruction_analysis'] = instruction_analysis
            except Exception as e:
                logging.warning(f"Instruction analysis failed: {e}")
        
        # Cross-modal insights
        results['cross_modal_insights'] = self._generate_cross_modal_insights(results)
        
        # Integrated awareness
        results['integrated_awareness'] = self._integrate_awareness_results(results)
        
        return results
    
    def _generate_cross_modal_insights(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from cross-modal analysis"""
        
        insights = {
            'domain_consistency': None,
            'complexity_alignment': None,
            'quality_correlations': None,
            'temporal_context': None,
            'linguistic_patterns': None
        }
        
        # Domain consistency across modalities
        domains = []
        if analysis_results['historical_analysis']:
            domains.append(analysis_results['historical_analysis'].get('expert_analysis', {}).get('expert_type'))
        if analysis_results['grammar_analysis']:
            domains.append(analysis_results['grammar_analysis'].get('text_analysis', {}).get('predicted_domain'))
        if analysis_results['instruction_analysis']:
            domains.append(analysis_results['instruction_analysis'].get('instruction_analysis', {}).get('predicted_domain'))
        
        if domains:
            domain_counts = {}
            for domain in domains:
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
            insights['domain_consistency'] = {
                'domains': domain_counts,
                'consistency_score': max(domain_counts.values()) / len(domains) if domains else 0.0
            }
        
        # Complexity alignment
        complexities = []
        if analysis_results['grammar_analysis']:
            complexities.append(analysis_results['grammar_analysis'].get('text_analysis', {}).get('predicted_complexity'))
        if analysis_results['instruction_analysis']:
            complexities.append(analysis_results['instruction_analysis'].get('instruction_analysis', {}).get('predicted_complexity'))
        
        if complexities:
            complexity_counts = {}
            for complexity in complexities:
                if complexity:
                    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
            insights['complexity_alignment'] = {
                'complexities': complexity_counts,
                'alignment_score': max(complexity_counts.values()) / len(complexities) if complexities else 0.0
            }
        
        # Quality correlations
        quality_scores = []
        if analysis_results['grammar_analysis']:
            quality_scores.append(analysis_results['grammar_analysis'].get('text_analysis', {}).get('quality_score'))
        if analysis_results['instruction_analysis']:
            quality_scores.append(analysis_results['instruction_analysis'].get('instruction_analysis', {}).get('quality_score'))
        
        if quality_scores:
            insights['quality_correlations'] = {
                'scores': quality_scores,
                'average_quality': np.mean([q for q in quality_scores if q is not None]),
                'quality_variance': np.var([q for q in quality_scores if q is not None])
            }
        
        return insights
    
    def _integrate_awareness_results(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate awareness results from all modalities"""
        
        awareness_results = []
        
        # Collect awareness results from each modality
        if analysis_results['historical_analysis']:
            hist_awareness = analysis_results['historical_analysis'].get('awareness_result', {})
            if hist_awareness:
                awareness_results.append(hist_awareness)
        
        if analysis_results['grammar_analysis']:
            grammar_awareness = analysis_results['grammar_analysis'].get('awareness_result', {})
            if grammar_awareness:
                awareness_results.append(grammar_awareness)
        
        if analysis_results['instruction_analysis']:
            inst_awareness = analysis_results['instruction_analysis'].get('awareness_result', {})
            if inst_awareness:
                awareness_results.append(inst_awareness)
        
        if not awareness_results:
            return {'error': 'No awareness results available'}
        
        # Integrate awareness levels
        awareness_levels = [ar.get('awareness_level') for ar in awareness_results if ar.get('awareness_level')]
        confidence_scores = [ar.get('confidence', 0.0) for ar in awareness_results if ar.get('confidence') is not None]
        consciousness_gates = [ar.get('consciousness_gate', 0.0) for ar in awareness_results if ar.get('consciousness_gate') is not None]
        
        # Determine integrated awareness level
        if awareness_levels:
            # Use highest awareness level
            level_hierarchy = ['REACTIVE', 'CONSCIOUS', 'SELF_AWARE', 'META_COGNITIVE']
            max_level_idx = max([level_hierarchy.index(level) for level in awareness_levels if level in level_hierarchy])
            integrated_level = level_hierarchy[max_level_idx]
        else:
            integrated_level = 'REACTIVE'
        
        # Calculate integrated confidence
        integrated_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        # Calculate integrated consciousness gate
        integrated_consciousness = np.mean(consciousness_gates) if consciousness_gates else 0.0
        
        return {
            'integrated_awareness_level': integrated_level,
            'integrated_confidence': float(integrated_confidence),
            'integrated_consciousness_gate': float(integrated_consciousness),
            'modality_count': len(awareness_results),
            'individual_results': awareness_results
        }
    
    def generate_multi_modal_response(self, 
                                    prompt: str,
                                    target_domain: str = None,
                                    target_complexity: str = None,
                                    historical_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate response using all available modalities"""
        
        # Analyze input across modalities
        analysis = self.analyze_multi_modal_input(
            prompt, 
            historical_context=historical_context,
            grammar_analysis=True,
            instruction_analysis=True
        )
        
        # Generate responses from each modality
        responses = {}
        
        # Grammar-aware generation
        if hasattr(self.grammar_engine, 'pattern_dataset') and self.grammar_engine.pattern_dataset:
            try:
                grammar_response = self.grammar_engine.generate_grammar_aware_text(
                    prompt, target_domain or 'general', target_complexity or 'simple'
                )
                responses['grammar'] = grammar_response
            except Exception as e:
                logging.warning(f"Grammar generation failed: {e}")
        
        # Instruction-based generation
        if hasattr(self.instruction_engine, 'instruction_dataset') and self.instruction_engine.instruction_dataset:
            try:
                instruction_response = self.instruction_engine.generate_response(
                    prompt, target_domain, target_complexity
                )
                responses['instruction'] = instruction_response
            except Exception as e:
                logging.warning(f"Instruction generation failed: {e}")
        
        # Historical context generation
        if historical_context and hasattr(self.historical_engine, 'sub_experts'):
            try:
                historical_response = self.historical_engine.generate_alternate_timeline(
                    historical_context, f"Generate response for: {prompt}", num_steps=3
                )
                responses['historical'] = historical_response
            except Exception as e:
                logging.warning(f"Historical generation failed: {e}")
        
        # Integrate responses
        integrated_response = self._integrate_responses(responses, analysis)
        
        return {
            'prompt': prompt,
            'analysis': analysis,
            'individual_responses': responses,
            'integrated_response': integrated_response,
            'generation_metadata': {
                'modalities_used': list(responses.keys()),
                'target_domain': target_domain,
                'target_complexity': target_complexity,
                'historical_context_provided': historical_context is not None
            }
        }
    
    def _integrate_responses(self, responses: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate responses from multiple modalities"""
        
        # Extract text responses
        text_responses = []
        for modality, response in responses.items():
            if isinstance(response, dict):
                if 'generated_text' in response:
                    text_responses.append(response['generated_text'])
                elif 'generated_response' in response:
                    text_responses.append(response['generated_response'])
                elif 'alternate_timeline' in response:
                    # Convert timeline to text
                    timeline = response['alternate_timeline']
                    if isinstance(timeline, torch.Tensor):
                        timeline_text = f"Based on historical analysis: {timeline.shape}"
                    else:
                        timeline_text = str(timeline)
                    text_responses.append(timeline_text)
        
        # Simple integration: combine all responses
        if text_responses:
            integrated_text = "\n\n".join(text_responses)
        else:
            integrated_text = "No responses generated from available modalities."
        
        # Determine quality score
        quality_scores = []
        for modality, response in responses.items():
            if isinstance(response, dict):
                if 'analysis' in response and 'instruction_analysis' in response['analysis']:
                    quality_scores.append(response['analysis']['instruction_analysis'].get('quality_score', 0.0))
                elif 'text_analysis' in response:
                    quality_scores.append(response['text_analysis'].get('quality_score', 0.0))
        
        integrated_quality = np.mean(quality_scores) if quality_scores else 0.0
        
        return {
            'integrated_text': integrated_text,
            'quality_score': float(integrated_quality),
            'source_modalities': list(responses.keys()),
            'response_count': len(text_responses)
        }
    
    def get_comprehensive_insights_report(self) -> Dict[str, Any]:
        """Generate comprehensive insights from all integrated datasets"""
        
        report = {
            'integration_statistics': self.integration_stats,
            'dataset_reports': {},
            'cross_modal_capabilities': {
                'multi_modal_analysis': True,
                'cross_dataset_learning': True,
                'domain_transfer': True,
                'quality_assessment': True,
                'response_generation': True,
                'historical_context_integration': True
            },
            'self_awareness_report': self.engine.get_self_report()
        }
        
        # Collect reports from each engine
        if hasattr(self.historical_engine, 'get_historical_insights_report'):
            try:
                report['dataset_reports']['historical'] = self.historical_engine.get_historical_insights_report()
            except Exception as e:
                logging.warning(f"Failed to get historical report: {e}")
        
        if hasattr(self.grammar_engine, 'get_grammar_insights_report'):
            try:
                report['dataset_reports']['grammar'] = self.grammar_engine.get_grammar_insights_report()
            except Exception as e:
                logging.warning(f"Failed to get grammar report: {e}")
        
        if hasattr(self.instruction_engine, 'get_instruction_insights_report'):
            try:
                report['dataset_reports']['instruction'] = self.instruction_engine.get_instruction_insights_report()
            except Exception as e:
                logging.warning(f"Failed to get instruction report: {e}")
        
        return report

# Example usage
def create_multi_modal_integration_example():
    """Create example for multi-modal dataset integration"""
    example_code = """
# Example: Multi-Modal Dataset Integration

from aura.neural.self_awareness import SelfAwarenessEngine
from aura.datasets.integration import MultiModalDatasetIntegrator

# 1. Initialize self-awareness engine
engine_config = {
    'state_dim': 512,
    'thought_dim': 128,
    'awareness_threshold': 0.7,
    'learning_config': {
        'batch_size': 4,
        'meta_learning': True,
        'cross_dataset_learning': True
    }
}

engine = SelfAwarenessEngine('multi_modal_consciousness', engine_config)
engine.initialize()

# 2. Create multi-modal integrator
integrator = MultiModalDatasetIntegrator(engine, device='mps')

# 3. Load all datasets
dataset_paths = {
    'historical_data_path': 'pretrain_tests/historical_high_confidence_annotated.jsonl',
    'grammar_data_path': 'pretrain_tests/spacy_svc_patterns_by_domain.json',
    'instruction_data_path': 'pretrain_tests/instruct_55k_clean.jsonl'
}

load_results = integrator.load_all_datasets(
    **dataset_paths,
    max_samples_per_dataset=5000
)

print(f"Dataset loading results: {load_results}")

# 4. Multi-modal analysis
text = "Explain the historical significance of the printing press and its impact on education"
historical_context = {
    'event_id': 'printing-press',
    'historian_annotation': {
        'eventName': 'Invention of the Printing Press',
        'eventType': 'cultural',
        'eventDate': '1440-01-01T00:00:00Z'
    }
}

analysis = integrator.analyze_multi_modal_input(
    text, 
    historical_context=historical_context
)

print(f"Multi-modal analysis: {analysis}")

# 5. Generate integrated response
response = integrator.generate_multi_modal_response(
    prompt="How did the printing press change society?",
    target_domain="cultural",
    target_complexity="complex",
    historical_context=historical_context
)

print(f"Integrated response: {response}")

# 6. Get comprehensive insights
insights = integrator.get_comprehensive_insights_report()
print(f"Comprehensive insights: {insights}")
"""
    return example_code

example_multi_modal = create_multi_modal_integration_example()
with open('multi_modal_integration_example.py', 'w') as f:
    f.write(example_multi_modal)
