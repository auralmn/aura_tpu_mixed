# AURA Tokenizer Training Guide

## 🎯 **Why Tokenizer Training is Critical**

The SentencePiece tokenizer is **essential** for AURA because:
- 🔤 Converts text to numerical tokens for neural processing
- 📊 Vocabulary size directly impacts model performance
- 🎯 Domain-specific tokenization improves accuracy
- 💾 Efficient encoding reduces memory usage
- 🌍 Supports multilingual and special tokens

**Without a trained tokenizer, you cannot:**
- Train the language model
- Run inference on text
- Use the self-teaching LLM adapter
- Deploy on TPU

---

## 🚀 **Quick Start**

### Basic Training
```bash
# Train tokenizer from text files
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm \\
  --vocab_size 32000
```

### Advanced Training
```bash
# Train from multiple sources with custom config
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt data/kb \\
  --json_dirs data/json data/pretrain \\
  --output models/spm \\
  --vocab_size 32000 \\
  --model_type unigram \\
  --special_tokens \\
  --user_symbols "<INST>,<RESP>,<THINK>"
```

---

## 📋 **Complete Training Workflow**

### Step 1: Prepare Training Data

#### Option A: Use Existing Text Files
```bash
# Organize your text data
mkdir -p data/txt
# Add .txt files to data/txt/
```

#### Option B: Download Training Corpus
```bash
# Download C4 or other corpus
python src/aura/data/hf_stream.py \\
  --dataset allenai/c4 \\
  --config en \\
  --split train \\
  --output data/c4_sample \\
  --max_samples 100000
```

#### Option C: Use AURA Data Processing
```bash
# Process with affect vectors
python -c "
from aura.ingestion.txt_loader import load_text_corpus
items = load_text_corpus('data/txt')
print(f'Loaded {len(items)} text items with affect vectors')
"
```

### Step 2: Train Tokenizer

#### Small Vocabulary (Fast, for testing)
```bash
python scripts/train_tokenizer.py \
  --txt_dirs data/txt \
  --output models/spm_small \\
  --vocab_size 2000 \\
  --max_files 1000
```
**Use case**: Quick experiments, testing pipeline

#### Medium Vocabulary (Recommended)
```bash
python3 scripts/train_tokenizer.py \
  --txt_dirs data/txt data/kb \
  --json_dirs data/json data/pretrain \
  --output models/spm_medium \
  --vocab_size 16000
```
**Use case**: Most AURA deployments, good balance

#### Large Vocabulary (Best quality)
```bash
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt data/kb \\
  --json_dirs data/json data/pretrain \\
  --output models/spm_large \\
  --vocab_size 32000 \\
  --character_coverage 0.9999
```
**Use case**: Production deployments, multilingual support

### Step 3: Validate Tokenizer
```bash
# Validation runs automatically unless skipped
# Manual validation:
python -c "
from aura.self_teaching_llm.tokenizer_spm import SPMTokenizer
tokenizer = SPMTokenizer('models/spm/spiece.model')
text = 'Hello, world!'
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
print(f'Original: {text}')
print(f'Tokens: {tokens}')
print(f'Decoded: {decoded}')
"
```

### Step 4: Use in Training
```bash
# Local training
python src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \\
  --spm_model models/spm/spiece.model \\
  --dataset allenai/c4 \\
  --steps 1000

# TPU training
export SPM_MODEL=models/spm/spiece.model
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

---

## ⚙️ **Configuration Options**

### Vocabulary Size Guidelines

| Size | Tokens | Use Case | Training Time | Memory |
|------|--------|----------|---------------|---------|
| Tiny | 1,000-2,000 | Testing only | 1-2 min | Low |
| Small | 4,000-8,000 | Experiments | 5-10 min | Low |
| Medium | 16,000 | Production | 15-30 min | Medium |
| Large | 32,000 | High quality | 30-60 min | Medium |
| XL | 50,000+ | Multilingual | 1-2 hours | High |

### Model Types

#### Unigram (Recommended)
```bash
--model_type unigram
```
- **Pros**: Better subword segmentation, flexible
- **Cons**: Slightly slower training
- **Use**: General purpose, recommended default

#### BPE (Byte-Pair Encoding)
```bash
--model_type bpe
```
- **Pros**: Faster training, deterministic
- **Cons**: Less flexible segmentation
- **Use**: When determinism is critical

### Special Tokens

#### Default Special Tokens (Recommended)
```bash
--special_tokens  # Enabled by default
```
Adds:
- `<pad>`, `<bos>`, `<eos>`, `<unk>` - Standard tokens
- `<INST>`, `<INPUT>`, `<RESPONSE>`, `<SEP>` - Instruction tuning
- `<SYSTEM>`, `<USER>`, `<ASSISTANT>` - Chat format
- `<THOUGHT>`, `<ACTION>`, `<OBSERVATION>` - Reasoning

#### Custom Special Tokens
```bash
--user_symbols "<EXPERT_0>,<EXPERT_1>,<NEUROPLASTIC>,<CAUSAL>"
```

### Character Coverage

```bash
# English only (default)
--character_coverage 0.9995

# Multilingual
--character_coverage 0.9999

# Include rare characters
--character_coverage 1.0
```

---

## 🔧 **Integration with AURA**

### 1. Integration with Data Processing
```python
# Use with affect-aware text loading
from aura.ingestion.txt_loader import load_text_corpus_all
from aura.self_teaching_llm.tokenizer_spm import SPMTokenizer

# Load data with affect vectors
items = load_text_corpus_all('data/txt', 'data/json')

# Train tokenizer
SPMTokenizer.train_from_dir(
    input_dir='data/txt',
    model_prefix='models/spm/spiece',
    vocab_size=16000
)

# Use tokenizer
tokenizer = SPMTokenizer('models/spm/spiece.model')
for text, affect in items:
    tokens = tokenizer.encode(text)
    # Process tokens with affect awareness...
```

### 2. Integration with Optimization Pipeline
```python
# Tokenizer training with optimization config
from scripts.train_tokenizer import train_tokenizer_enhanced

model_path = train_tokenizer_enhanced(
    corpus_path='data/corpus.txt',
    output_dir='models/spm',
    vocab_size=32000,
    model_type='unigram'
)

# Use with optimized training
# Automatically integrated in deploy_optimizations.py
```

### 3. Integration with TPU Training
```bash
# Train tokenizer first
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm \\
  --vocab_size 32000

# Upload to TPU
gcloud compute tpus tpu-vm scp \\
  models/spm/spiece.model \\
  $NAME:~/aura_tpu/models/spm/ \\
  --worker all --recurse

# Use in TPU training
export SPM_MODEL=models/spm/spiece.model
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

---

## 📊 **Training Data Requirements**

### Minimum Requirements
- **Files**: At least 100 text files
- **Total size**: 10MB+ of text
- **Vocab coverage**: Diverse vocabulary

### Recommended
- **Files**: 10,000+ text files or samples
- **Total size**: 100MB-1GB of text  
- **Domains**: Multiple domains/styles
- **Quality**: Clean, well-formatted text

### Optimal
- **Files**: 100,000+ samples
- **Total size**: 1GB-10GB of text
- **Domains**: Broad coverage
- **Languages**: Multiple if needed
- **Quality**: High-quality, curated corpus

---

## 🔍 **Troubleshooting**

### Issue: "No .txt files found"
```bash
# Check your data directory
ls -la data/txt/

# Solution: Add text files or specify correct path
python scripts/train_tokenizer.py --txt_dirs /path/to/your/data
```

### Issue: "sentencepiece not installed"
```bash
# Install sentencepiece
pip install sentencepiece

# Or use requirements
pip install -r requirements.txt
```

### Issue: Vocab size too large/small
```bash
# Auto-estimate appropriate size
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm
  # Omit --vocab_size for auto-estimation
```

### Issue: Training takes too long
```bash
# Test with smaller dataset first
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm_test \\
  --max_files 1000 \\
  --vocab_size 2000
```

### Issue: Tokenizer gives poor results
```bash
# Increase vocab size
--vocab_size 32000

# Use unigram model
--model_type unigram

# Increase character coverage
--character_coverage 0.9999

# Add more training data
--txt_dirs data/txt data/more_data
```

---

## 📈 **Performance Tips**

### For Fastest Training
1. Use `--model_type bpe`
2. Limit `--max_files 10000`
3. Use smaller `--vocab_size 8000`
4. Use default `--character_coverage`

### For Best Quality
1. Use `--model_type unigram`
2. Use large corpus (no --max_files limit)
3. Use larger `--vocab_size 32000`
4. Increase `--character_coverage 0.9999`

### For Multilingual
1. Collect diverse language data
2. Use `--character_coverage 0.9999` or `1.0`
3. Use larger `--vocab_size 50000`
4. Add language-specific `--user_symbols`

---

## 🎯 **Best Practices**

### 1. Train Once, Use Everywhere
- Train on representative data
- Save model in version control (Git LFS)
- Reuse across experiments
- Only retrain if domain changes

### 2. Version Your Tokenizers
```bash
# Include date/version in output
python scripts/train_tokenizer.py \\
  --output models/spm_v1_20251029 \\
  --vocab_size 32000
```

### 3. Keep Training Config
- Config automatically saved as `spiece_config.json`
- Track vocab size, coverage, special tokens
- Reproducible training

### 4. Validate Before Production
```bash
# Always validate
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm
  # Validation runs automatically
```

### 5. Match Training Data to Use Case
- English only → English corpus
- Code generation → Include code
- Chat/instruction → Include dialogues
- Domain-specific → Domain data

---

## 🚀 **Complete Example Workflow**

```bash
# 1. Setup directories
mkdir -p data/txt data/json models/spm

# 2. Prepare training data (example with C4)
python -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
with open('data/txt/c4_sample.txt', 'w') as f:
    for i, item in enumerate(ds):
        if i >= 100000: break
        f.write(item['text'] + '\\n\\n')
print('Downloaded 100k samples')
"

# 3. Train tokenizer
python scripts/train_tokenizer.py \\
  --txt_dirs data/txt \\
  --output models/spm \\
  --vocab_size 32000 \\
  --model_type unigram \\
  --special_tokens

# 4. Validate
python -c "
from aura.self_teaching_llm.tokenizer_spm import SPMTokenizer
tok = SPMTokenizer('models/spm/spiece.model')
print(f'Vocab size: {tok.proc.get_piece_size()}')
test = 'The quick brown fox jumps over the lazy dog.'
tokens = tok.encode(test)
print(f'Tokens: {tokens}')
print(f'Decoded: {tok.decode(tokens)}')
"

# 5. Use in training
python src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \\
  --spm_model models/spm/spiece.model \\
  --dataset allenai/c4 \\
  --steps 1000 \\
  --lr 8e-4

# 6. Deploy on TPU (if using TPU)
./scripts/gcloud_tpu_multihost_optimized.sh setup
export SPM_MODEL=models/spm/spiece.model
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

---

## ✅ **Checklist Before Training**

- [ ] Training data prepared (at least 10MB text)
- [ ] Output directory created (`mkdir -p models/spm`)
- [ ] SentencePiece installed (`pip install sentencepiece`)
- [ ] Vocab size decided (default: 32000)
- [ ] Special tokens defined (if custom)
- [ ] Training time allocated (15-60 minutes)

## ✅ **Checklist After Training**

- [ ] Model file exists (`models/spm/spiece.model`)
- [ ] Vocab file exists (`models/spm/spiece.vocab`)
- [ ] Config saved (`models/spm/spiece_config.json`)
- [ ] Validation passed
- [ ] Model committed to version control (Git LFS)
- [ ] Ready to use in training!

---

## 🎉 **You're Ready!**

Your tokenizer is the **foundation** of your AURA language model. With a well-trained tokenizer:
- ✅ Efficient text encoding
- ✅ Better model performance  
- ✅ Faster training
- ✅ Production-ready pipeline

**Next Steps**: Use your tokenizer in AURA training and enjoy the optimized performance! 🚀
