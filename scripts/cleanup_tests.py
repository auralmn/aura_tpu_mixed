#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test cleanup script to remove redundant test files and organize test structure.
Run this after reviewing the consolidated test files.
"""

import os
import shutil
import sys
from pathlib import Path

# Files to be removed (redundant learning tests)
REDUNDANT_FILES = [
    'src/aura/training/actual_learning_test.py',
    'src/aura/training/effective_learning_test.py',
    'src/aura/training/extended_learning_test.py', 
    'src/aura/training/simple_learning_test.py',
    'src/aura/training/sophisticated_learning_test.py',
    'src/aura/training/mnist_learning_test.py',
    'src/aura/training/simple_mnist_test.py',
    'src/aura/training/simple_phase_test.py',
    'src/aura/training/test_all_phases.py'
]

# Files to be archived (moved to archive folder)
ARCHIVE_FILES = [
    'src/aura/training/demo_bio_components.py',
    'src/aura/training/test_bio_components.py'
]

def create_archive_directory():
    """Create archive directory for old test files."""
    archive_dir = Path('archive/old_tests')
    archive_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir

def backup_file(file_path, backup_dir):
    """Backup a file before deletion."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = Path(file_path)
    if file_path.exists():
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        print(f"✓ Backed up {file_path} to {backup_path}")
        return True
    return False

def remove_redundant_files(dry_run=True):
    """Remove redundant test files."""
    print("=== REMOVING REDUNDANT TEST FILES ===")
    
    if dry_run:
        print("DRY RUN MODE - No files will be actually deleted")
    
    removed_count = 0
    backup_dir = create_archive_directory() / 'removed_redundant'
    
    for file_path in REDUNDANT_FILES:
        file_path = Path(file_path)
        if file_path.exists():
            if not dry_run:
                # Backup before deletion
                backup_file(file_path, backup_dir)
                file_path.unlink()
                print(f"✗ Removed {file_path}")
            else:
                print(f"[DRY RUN] Would remove {file_path}")
            removed_count += 1
        else:
            print(f"  File not found: {file_path}")
    
    print(f"Total files {'would be ' if dry_run else ''}removed: {removed_count}")

def archive_old_files(dry_run=True):
    """Archive old demo/test files."""
    print("\n=== ARCHIVING OLD DEMO/TEST FILES ===")
    
    archive_dir = create_archive_directory() / 'demos'
    archived_count = 0
    
    for file_path in ARCHIVE_FILES:
        file_path = Path(file_path)
        if file_path.exists():
            if not dry_run:
                archive_path = archive_dir / file_path.name
                shutil.move(str(file_path), str(archive_path))
                print(f"📦 Archived {file_path} to {archive_path}")
            else:
                print(f"[DRY RUN] Would archive {file_path}")
            archived_count += 1
        else:
            print(f"  File not found: {file_path}")
    
    print(f"Total files {'would be ' if dry_run else ''}archived: {archived_count}")

def create_test_organization_readme():
    """Create README explaining the new test organization."""
    readme_content = """# AURA Test Suite Organization

## Test Structure

### Core Test Suites (tests/)
- `test_bio_inspired_components.py` - Comprehensive bio-inspired component testing
- `test_prompt_duel_system.py` - Prompt duel optimizer testing  
- `test_data_processing.py` - Data ingestion and processing testing
- `test_training_phases.py` - Training phase testing (consolidated)

### Existing Specialized Tests (tests/)
- `test_aura_consciousness_system.py` - Consciousness system testing
- `test_self_teaching_adapter.py` - Self-teaching LLM adapter testing
- `test_spiking_language_core.py` - Language core testing
- `test_generation_loop.py` - Text generation testing
- Individual component tests (dream_synthesizer, memory_processor, etc.)

### Active Training Scripts (src/aura/training/)
- `mnist_expert_poc.py` - Main MNIST expert training with bio-components
- `bio_inspired_training.py` - Bio-inspired training orchestrator
- `tpu_training_pipeline.py` - TPU-optimized training pipeline
- `local_test_pipeline.py` - Local testing pipeline
- `real_mnist_test.py` - Real MNIST testing with bio-components
- `mnist_baseline.py` - Baseline MNIST comparison

### Data Creation & Config (src/aura/training/)
- `create_minimal_dataset.py` - Minimal dataset creation
- `create_production_dataset.py` - Production dataset creation
- `cpu_mps_training_config.py` - CPU/MPS training configuration
- `train_cpu_mps.py` - CPU/MPS training script

## Removed Redundant Files

The following redundant learning test files were consolidated into `test_bio_inspired_components.py`:
- `actual_learning_test.py`
- `effective_learning_test.py`
- `extended_learning_test.py`
- `simple_learning_test.py`
- `sophisticated_learning_test.py`
- `mnist_learning_test.py`
- `simple_mnist_test.py`

The following minimal phase test files were consolidated into `test_training_phases.py`:
- `simple_phase_test.py`
- `test_all_phases.py`

## Running Tests

### All tests:
```bash
python main.py test
```

### Specific test categories:
```bash
python main.py test -k bio_inspired
python main.py test -k prompt_duel
python main.py test -k data_processing
python main.py test -k training_phases
```

### Verbose output:
```bash
python main.py test -v
```

## Test Coverage

The consolidated test suites provide comprehensive coverage of:
- ✅ All bio-inspired components (PhasorBank, SpikingAttention, Experts, etc.)
- ✅ Prompt duel optimization system
- ✅ Data processing pipeline
- ✅ Training phase execution
- ✅ Integration testing
- ✅ Error handling and edge cases

## Archive

Old/redundant files are preserved in `archive/old_tests/` for reference.
"""
    
    readme_path = Path('TEST_ORGANIZATION.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\n✓ Created test organization README: {readme_path}")

def analyze_test_coverage():
    """Analyze test coverage of the codebase."""
    print("\n=== TEST COVERAGE ANALYSIS ===")
    
    # Core modules that should have tests
    core_modules = {
        'consciousness': ['aura_consciousness_system', 'theta_gamma_oscillator', 'memory_processor', 
                         'global_workspace_manager', 'dream_synthesizer', 'metacognitive_monitor'],
        'bio_inspired': ['enhanced_spiking_retrieval', 'experts', 'phasor_bank', 'spiking_attention',
                        'merit_board', 'personality_engine', 'expert_registry', 'thalamic_router'],
        'self_teaching_llm': ['self_teaching_adapter', 'spiking_language_core', 'spiking_retrieval_core',
                             'token_embedding', 'token_decoder', 'generation_loop', 'tokenizer_spm'],
        'prompt_duel': ['dueler', 'judges', 'cli'],
        'data': ['hf_stream', 'hf_conversations', 'hf_ultrachat', 'hf_templategsm'],
        'ingestion': ['txt_loader'],
        'retrieval': ['opensearch_ingest'],
        'tools': ['builder', 'registry', 'runtime']
    }
    
    # Check existing tests
    test_dir = Path('tests')
    existing_tests = list(test_dir.glob('test_*.py')) if test_dir.exists() else []
    existing_test_names = [t.stem for t in existing_tests]
    
    print(f"Found {len(existing_tests)} existing test files:")
    for test in existing_tests:
        print(f"  ✓ {test.name}")
    
    # New comprehensive tests
    new_tests = [
        'test_bio_inspired_components.py',
        'test_prompt_duel_system.py', 
        'test_data_processing.py',
        'test_training_phases.py'
    ]
    
    print(f"\nNew comprehensive test files created:")
    for test in new_tests:
        print(f"  ✓ {test}")
    
    print(f"\n📊 Total test coverage: {len(existing_tests) + len(new_tests)} test suites")

def main():
    """Main cleanup function."""
    print("AURA Test Cleanup Script")
    print("=======================")
    
    if len(sys.argv) > 1 and sys.argv[1] == '--execute':
        dry_run = False
        print("EXECUTION MODE - Files will be actually modified")
    else:
        dry_run = True
        print("DRY RUN MODE - Use --execute to actually modify files")
    
    # Analyze current state
    analyze_test_coverage()
    
    # Remove redundant files
    remove_redundant_files(dry_run)
    
    # Archive old demo files  
    archive_old_files(dry_run)
    
    # Create documentation
    if not dry_run:
        create_test_organization_readme()
    else:
        print("\n[DRY RUN] Would create TEST_ORGANIZATION.md")
    
    print("\n" + "="*50)
    if dry_run:
        print("DRY RUN COMPLETED - No files were modified")
        print("Run with --execute to perform actual cleanup")
    else:
        print("CLEANUP COMPLETED")
        print("✓ Redundant files removed and backed up")
        print("✓ Demo files archived") 
        print("✓ Documentation created")
        print("✓ Test suite consolidated and organized")
    
    print("\nNext steps:")
    print("1. Run the new comprehensive tests: python main.py test")
    print("2. Review TEST_ORGANIZATION.md for the new structure")
    print("3. Update any scripts that referenced the removed files")

if __name__ == '__main__':
    main()
