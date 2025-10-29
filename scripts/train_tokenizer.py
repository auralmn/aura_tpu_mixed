#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Enhanced SentencePiece Tokenizer Training Script
Integrates with AURA data processing and optimization pipeline
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aura.self_teaching_llm.tokenizer_spm import SPMTokenizer
from aura.ingestion.txt_loader import load_text_corpus, load_json_corpus, load_text_corpus_all


def prepare_training_data(
    txt_dirs: List[str],
    json_dirs: List[str],
    output_dir: str,
    min_length: int = 10,
    max_files: Optional[int] = None
) -> str:
    """
    Prepare training data from multiple sources.
    
    Args:
        txt_dirs: List of directories containing .txt files
        json_dirs: List of directories containing .json/.jsonl files
        output_dir: Where to write combined corpus
        min_length: Minimum text length to include
        max_files: Maximum number of files to process (for testing)
        
    Returns:
        Path to prepared training corpus
    """
    print("🔍 Gathering training data...")
    
    os.makedirs(output_dir, exist_ok=True)
    corpus_path = os.path.join(output_dir, "tokenizer_training_corpus.txt")
    
    total_items = 0
    total_chars = 0
    
    with open(corpus_path, 'w', encoding='utf-8') as out_file:
        # Process text directories
        for txt_dir in txt_dirs:
            if not os.path.exists(txt_dir):
                print(f"⚠️  Directory not found: {txt_dir}")
                continue
            
            print(f"📁 Processing text directory: {txt_dir}")
            items = load_text_corpus(txt_dir)
            
            for text, affect_vector in items:
                if len(text) >= min_length:
                    out_file.write(text + "\n")
                    total_items += 1
                    total_chars += len(text)
                    
                    if max_files and total_items >= max_files:
                        break
            
            if max_files and total_items >= max_files:
                break
        
        # Process JSON directories
        for json_dir in json_dirs:
            if not os.path.exists(json_dir):
                print(f"⚠️  Directory not found: {json_dir}")
                continue
            
            print(f"📁 Processing JSON directory: {json_dir}")
            items = load_json_corpus(json_dir)
            
            for text, affect_vector in items:
                if len(text) >= min_length:
                    out_file.write(text + "\n")
                    total_items += 1
                    total_chars += len(text)
                    
                    if max_files and total_items >= max_files:
                        break
            
            if max_files and total_items >= max_files:
                break
    
    print(f"✅ Prepared corpus:")
    print(f"   Items: {total_items:,}")
    print(f"   Characters: {total_chars:,}")
    print(f"   Average length: {total_chars // total_items if total_items > 0 else 0}")
    print(f"   Saved to: {corpus_path}")
    
    return corpus_path


def estimate_vocab_size(corpus_path: str) -> int:
    """Estimate appropriate vocabulary size based on corpus."""
    print("\n📊 Analyzing corpus for vocab size estimation...")
    
    char_count = 0
    unique_words = set()
    line_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            char_count += len(line)
            words = line.strip().split()
            unique_words.update(words)
            
            if line_count >= 10000:  # Sample first 10k lines
                break
    
    # Heuristic: vocab size ~= sqrt(unique_words) * 100
    estimated = int((len(unique_words) ** 0.5) * 100)
    
    # Clamp to reasonable range
    estimated = max(2000, min(estimated, 50000))
    
    # Round to nearest 1000
    estimated = round(estimated / 1000) * 1000
    
    print(f"   Unique words (sampled): {len(unique_words):,}")
    print(f"   Estimated vocab size: {estimated:,}")
    
    return estimated


def train_tokenizer_enhanced(
    corpus_path: str,
    output_dir: str,
    vocab_size: Optional[int] = None,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    user_symbols: Optional[List[str]] = None,
    special_tokens: bool = True
) -> str:
    """
    Train SentencePiece tokenizer with enhanced features.
    
    Args:
        corpus_path: Path to training corpus
        output_dir: Output directory for model
        vocab_size: Vocabulary size (auto-estimated if None)
        model_type: "unigram" or "bpe"
        character_coverage: Coverage for character sampling
        user_symbols: Additional user-defined symbols
        special_tokens: Add common special tokens
        
    Returns:
        Path to trained model
    """
    print("\n🚀 Training SentencePiece tokenizer...")
    
    # Auto-estimate vocab size if not provided
    if vocab_size is None:
        vocab_size = estimate_vocab_size(corpus_path)
    
    # Add common special tokens
    # Note: <pad>, <bos>, <eos>, <unk> are built-in SentencePiece control tokens
    # and should NOT be included in user_defined_symbols
    if special_tokens:
        default_symbols = [
            "<INST>", "<INPUT>", "<RESPONSE>", "<SEP>",
            "<SYSTEM>", "<USER>", "<ASSISTANT>",
            "<THOUGHT>", "<ACTION>", "<OBSERVATION>"
        ]
        
        if user_symbols:
            # Combine and deduplicate
            all_symbols = list(dict.fromkeys(default_symbols + user_symbols))
        else:
            all_symbols = default_symbols
    else:
        all_symbols = user_symbols or []
    
    print(f"   Vocab size: {vocab_size:,}")
    print(f"   Model type: {model_type}")
    print(f"   Character coverage: {character_coverage}")
    print(f"   Special tokens: {len(all_symbols)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_prefix = os.path.join(output_dir, "spiece")
    
    # Train the model
    try:
        model_path = SPMTokenizer.train_from_dir(
            input_dir=os.path.dirname(corpus_path),  # Will find corpus file
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            user_defined_symbols=all_symbols,
            max_sentence_length=1000000,
            hard_vocab_limit=False,
            byte_fallback=True,
            train_extremely_large_corpus=True,  # Required for large corpora (>10M sentences)
            clean_controls=True,
            normalize_spaces=True,
            use_iterator=False,
            ascii_only=False
        )
        
        print(f"✅ Tokenizer trained successfully!")
        print(f"   Model: {model_path}")
        print(f"   Vocab: {model_prefix}.vocab")
        
        return model_path
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def validate_tokenizer(model_path: str, test_texts: Optional[List[str]] = None):
    """Validate trained tokenizer with test samples."""
    print("\n🔍 Validating tokenizer...")
    
    tokenizer = SPMTokenizer(model_path)
    
    if test_texts is None:
        test_texts = [
            "Hello, world! This is a test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning and artificial intelligence are transforming technology.",
            "SentencePiece is a tokenization library for neural text processing.",
            "Testing special characters: @#$%^&*() and emojis 🚀🧠🔬"
        ]
    
    print("\n📝 Test Encodings:")
    for i, text in enumerate(test_texts, 1):
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        decoded = tokenizer.decode(tokens)
        
        print(f"\n{i}. Input: {text[:60]}...")
        print(f"   Tokens: {len(tokens)} tokens")
        print(f"   Sample: {tokens[:10]}...")
        print(f"   Decoded: {decoded[:60]}...")
        print(f"   Match: {'✅' if text.strip() == decoded.strip() else '⚠️'}")
    
    # Vocabulary statistics
    vocab_size = tokenizer.proc.get_piece_size()
    print(f"\n📊 Vocabulary Statistics:")
    print(f"   Total vocab size: {vocab_size:,}")
    print(f"   BOS token: {tokenizer.proc.bos_id()} = '{tokenizer.proc.id_to_piece(tokenizer.proc.bos_id())}'")
    print(f"   EOS token: {tokenizer.proc.eos_id()} = '{tokenizer.proc.id_to_piece(tokenizer.proc.eos_id())}'")
    print(f"   PAD token: {tokenizer.proc.pad_id()} = '{tokenizer.proc.id_to_piece(tokenizer.proc.pad_id())}'")
    print(f"   UNK token: {tokenizer.proc.unk_id()} = '{tokenizer.proc.id_to_piece(tokenizer.proc.unk_id())}'")
    
    print("\n✅ Validation complete!")


def create_training_config(output_path: str, **kwargs):
    """Save training configuration for reproducibility."""
    config = {
        "vocab_size": kwargs.get("vocab_size"),
        "model_type": kwargs.get("model_type", "unigram"),
        "character_coverage": kwargs.get("character_coverage", 0.9995),
        "special_tokens": kwargs.get("special_tokens", True),
        "txt_dirs": kwargs.get("txt_dirs", []),
        "json_dirs": kwargs.get("json_dirs", []),
        "timestamp": str(Path(output_path).stat().st_mtime)
    }
    
    config_path = output_path.replace(".model", "_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Training config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced SentencePiece Tokenizer Training for AURA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from text files
  python scripts/train_tokenizer.py --txt_dirs data/txt --output models/spm
  
  # Train from multiple sources
  python scripts/train_tokenizer.py \\
    --txt_dirs data/txt data/corpus \\
    --json_dirs data/json \\
    --output models/spm \\
    --vocab_size 32000
  
  # Quick test with small vocab
  python scripts/train_tokenizer.py \\
    --txt_dirs data/txt \\
    --output models/spm_test \\
    --vocab_size 2000 \\
    --max_files 1000
        """
    )
    
    # Input sources
    parser.add_argument("--txt_dirs", nargs="+", default=["data/txt"],
                       help="Directories containing .txt files")
    parser.add_argument("--json_dirs", nargs="+", default=[],
                       help="Directories containing .json/.jsonl files")
    
    # Output configuration
    parser.add_argument("--output", type=str, default="models/spm",
                       help="Output directory for tokenizer model")
    parser.add_argument("--work_dir", type=str, default="data/tokenizer_training",
                       help="Working directory for corpus preparation")
    
    # Training parameters
    parser.add_argument("--vocab_size", type=int, default=None,
                       help="Vocabulary size (auto-estimated if not specified)")
    parser.add_argument("--model_type", type=str, default="unigram",
                       choices=["unigram", "bpe"],
                       help="Tokenizer model type")
    parser.add_argument("--character_coverage", type=float, default=0.9995,
                       help="Character coverage for tokenization")
    
    # Special tokens
    parser.add_argument("--special_tokens", action="store_true", default=True,
                       help="Add common special tokens")
    parser.add_argument("--user_symbols", type=str, default="",
                       help="Additional user symbols (comma-separated)")
    
    # Processing options
    parser.add_argument("--min_length", type=int, default=10,
                       help="Minimum text length to include")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum files to process (for testing)")
    
    # Validation
    parser.add_argument("--skip_validation", action="store_true",
                       help="Skip tokenizer validation")
    
    args = parser.parse_args()
    
    print("🚀 AURA SentencePiece Tokenizer Training")
    print("=" * 60)
    
    # Step 1: Prepare training data
    corpus_path = prepare_training_data(
        txt_dirs=args.txt_dirs,
        json_dirs=args.json_dirs,
        output_dir=args.work_dir,
        min_length=args.min_length,
        max_files=args.max_files
    )
    
    # Step 2: Train tokenizer
    user_symbols = [s.strip() for s in args.user_symbols.split(",") if s.strip()]
    
    model_path = train_tokenizer_enhanced(
        corpus_path=corpus_path,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        user_symbols=user_symbols if user_symbols else None,
        special_tokens=args.special_tokens
    )
    
    # Step 3: Save configuration
    create_training_config(
        model_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        special_tokens=args.special_tokens,
        txt_dirs=args.txt_dirs,
        json_dirs=args.json_dirs
    )
    
    # Step 4: Validate tokenizer
    if not args.skip_validation:
        validate_tokenizer(model_path)
    
    print("\n" + "=" * 60)
    print("🎉 Tokenizer Training Complete!")
    print("=" * 60)
    print(f"\n✅ Model: {model_path}")
    print(f"✅ Vocab: {model_path.replace('.model', '.vocab')}")
    print(f"\nNext steps:")
    print(f"  1. Use in training: --spm_model {model_path}")
    print(f"  2. Load in code: SPMTokenizer('{model_path}')")
    print(f"  3. Integrate with AURA training pipeline")


if __name__ == "__main__":
    main()
