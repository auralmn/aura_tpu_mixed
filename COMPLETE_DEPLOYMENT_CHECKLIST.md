# Complete AURA Deployment Checklist

## 🎯 **Full Production Deployment Guide**

This checklist covers **everything** needed to deploy AURA with all optimizations.

---

## 📋 **Pre-Deployment Checklist**

### 1. ✅ **Environment Setup**
- [ ] Python 3.10+ installed
- [ ] JAX installed (`pip install jax`)
- [ ] Flax installed (`pip install flax`)
- [ ] Optax installed (`pip install optax`)
- [ ] SentencePiece installed (`pip install sentencepiece`)
- [ ] Git repository cloned
- [ ] All dependencies installed (`pip install -r requirements.txt`)

### 2. ✅ **Tokenizer Training** (CRITICAL!)
```bash
# Train tokenizer first!
python scripts/train_tokenizer.py \
  --txt_dirs data/txt \
  --output models/spm \
  --vocab_size 32000
```

- [ ] Training data prepared (at least 10MB text)
- [ ] Tokenizer trained (`models/spm/spiece.model` exists)
- [ ] Tokenizer validated (test encoding/decoding works)
- [ ] Vocab file created (`models/spm/spiece.vocab`)
- [ ] Config saved (`models/spm/spiece_config.json`)

**Why Critical**: Without a tokenizer, you cannot train or run the model!

### 3. ✅ **Optimization Validation**
```bash
# Run validation tests
./scripts/validate_optimizations.sh
```

- [ ] All optimization modules present
- [ ] Syntax checks passed
- [ ] Unit tests passed (86%+ pass rate)
- [ ] Import checks successful
- [ ] Smoke tests passed

### 4. ✅ **TPU Setup** (If using TPU)
```bash
# Setup TPU pod
export PROJECT=your-project
export ZONE=us-central2-b
export NAME=aura-v4-32
export VERSION=tpu-ubuntu2204-base

./scripts/gcloud_tpu_multihost_optimized.sh create
```

- [ ] GCP project configured
- [ ] TPU zone selected
- [ ] TPU pod created
- [ ] Workers accessible via SSH
- [ ] Repository cloned on all workers
- [ ] Tokenizer uploaded to TPU workers

---

## 🚀 **Deployment Steps**

### Step 1: Train Tokenizer
```bash
# MUST DO FIRST!
python scripts/train_tokenizer.py \
  --txt_dirs data/txt data/corpus \
  --json_dirs data/json \
  --output models/spm \
  --vocab_size 32000 \
  --model_type unigram \
  --special_tokens
```

**Expected Output**:
```
✅ Model: models/spm/spiece.model
✅ Vocab: models/spm/spiece.vocab
📊 Vocabulary Statistics:
   Total vocab size: 32,000
```

### Step 2: Deploy Optimizations (Local)
```bash
# Deploy all optimizations
python scripts/deploy_optimizations.py \
  --model-size medium \
  --sequence-length 512 \
  --num-experts 16
```

**Expected Output**:
```
🚀 TPU Optimization Setup Complete
🧠 Neuroplasticity engine deployed
🔬 Causal reasoning deployed
🧬 Evolutionary experts deployed
🎯 Meta-learning deployed
🎉 AURA OPTIMIZATION DEPLOYMENT COMPLETE!
```

### Step 3: Validate Deployment
```bash
# Check deployment report
cat optimization_deployment_report.json

# Verify optimizations
python -c "
from aura.optimization.tpu_optimizer import create_optimized_training_setup
config = create_optimized_training_setup('medium', 512, 16)
print(f'✅ Optimizations ready!')
print(f'Batch size: {config.get_training_config()[\"batch_size\"]}')
"
```

### Step 4: Deploy to TPU (If using TPU)
```bash
# Setup with optimizations
./scripts/gcloud_tpu_multihost_optimized.sh setup

# Upload tokenizer
gcloud compute tpus tpu-vm scp \
  models/spm/spiece.model \
  $NAME:~/aura_tpu/models/spm/ \
  --worker all --recurse

# Validate on TPU
./scripts/gcloud_tpu_multihost_optimized.sh validate
```

### Step 5: Launch Training

#### Local Training
```bash
python src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \
  --spm_model models/spm/spiece.model \
  --dataset allenai/c4 \
  --config en \
  --split train \
  --steps 1000 \
  --lr 8e-4 \
  --seq_len 512 \
  --batch_size 128 \
  --dtype bf16 \
  --use_optimizations true
```

#### TPU Training
```bash
# Configure
export MODEL_SIZE=medium
export SPM_MODEL=models/spm/spiece.model
export SEQ_LEN=512
export BATCH_SIZE=128
export STEPS=10000

# Launch optimized training
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

### Step 6: Monitor Training
```bash
# Check optimization status
./scripts/gcloud_tpu_multihost_optimized.sh status

# View performance metrics
./scripts/gcloud_tpu_multihost_optimized.sh metrics

# Monitor logs (TPU)
gcloud compute tpus tpu-vm ssh $NAME --worker 0 \
  --command 'tail -f ~/aura_tpu/logs/train_optimized_worker_0.out'
```

---

## 📊 **Verification Checklist**

### After Tokenizer Training
- [ ] Model file size > 100KB
- [ ] Vocab file readable
- [ ] Test encoding works: `tokenizer.encode("Hello world")`
- [ ] Test decoding works: `tokenizer.decode(tokens)`
- [ ] Roundtrip test passes: `decode(encode(text)) == text`

### After Optimization Deployment
- [ ] Deployment report created
- [ ] All components show "deployed" status
- [ ] No error messages in logs
- [ ] Import tests pass
- [ ] Smoke tests complete

### After Training Launch
- [ ] Training process started (check PIDs)
- [ ] Logs being written
- [ ] Loss decreasing
- [ ] No out-of-memory errors
- [ ] Performance metrics within expected range

---

## 🎯 **Expected Performance**

### With Optimizations (vs Baseline)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Training Speed | 1000 steps/hr | 2000-3000 steps/hr | **2-3x faster** |
| Memory Usage | 28GB/core | 14-16GB/core | **50% reduction** |
| Batch Size | Fixed 64 | Dynamic 128+ | **2x larger** |
| Convergence | Standard | Faster | **15-20% fewer steps** |

### Tokenizer Impact

| Vocab Size | Encoding Speed | Model Quality | Memory |
|------------|----------------|---------------|--------|
| 2,000 | Very Fast | Low | Minimal |
| 8,000 | Fast | Good | Low |
| 16,000 | Medium | Very Good | Medium |
| 32,000 | Medium | Excellent | Medium |
| 50,000+ | Slower | Best | Higher |

---

## 🔧 **Troubleshooting Guide**

### Issue 1: No Tokenizer Model
```
Error: FileNotFoundError: models/spm/spiece.model
```

**Solution**:
```bash
# Train tokenizer first!
python scripts/train_tokenizer.py \
  --txt_dirs data/txt \
  --output models/spm \
  --vocab_size 32000
```

### Issue 2: Tokenizer Training Fails
```
Error: No .txt files found in data/txt
```

**Solution**:
```bash
# Add training data
mkdir -p data/txt
# Download or add text files
python -c "
from datasets import load_dataset
ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)
with open('data/txt/sample.txt', 'w') as f:
    for i, item in enumerate(ds):
        if i >= 10000: break
        f.write(item['text'] + '\\n')
"
```

### Issue 3: Optimization Deployment Fails
```
Error: Import failed for optimization modules
```

**Solution**:
```bash
# Verify files exist
ls -la src/aura/optimization/

# Run validation
./scripts/validate_optimizations.sh

# Check dependencies
pip install -U jax flax optax
```

### Issue 4: TPU Training Hangs
```
Training process started but no logs
```

**Solution**:
```bash
# Check process status
./scripts/gcloud_tpu_multihost_optimized.sh status

# View logs
gcloud compute tpus tpu-vm ssh $NAME --worker 0 \
  --command 'cat ~/aura_tpu/logs/train_optimized_worker_0.out'

# Restart if needed
./scripts/gcloud_tpu_multihost_optimized.sh stop
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

### Issue 5: Out of Memory
```
Error: Out of memory on TPU
```

**Solution**:
```bash
# Reduce batch size
export BATCH_SIZE=64
export PER_DEVICE_BATCH=4

# Enable gradient checkpointing (automatic in optimized script)
# Or reduce sequence length
export SEQ_LEN=256
```

---

## 📝 **Configuration Templates**

### Small/Test Configuration
```bash
# Quick test setup
export MODEL_SIZE=small
export VOCAB_SIZE=2000
export SEQ_LEN=256
export BATCH_SIZE=32
export STEPS=100
```

### Medium/Development Configuration
```bash
# Development setup
export MODEL_SIZE=medium
export VOCAB_SIZE=16000
export SEQ_LEN=512
export BATCH_SIZE=128
export STEPS=1000
```

### Large/Production Configuration
```bash
# Production setup
export MODEL_SIZE=large
export VOCAB_SIZE=32000
export SEQ_LEN=1024
export BATCH_SIZE=256
export STEPS=100000
```

---

## 🎯 **Critical Path**

**Must complete in order:**

1. **Train Tokenizer** ← Without this, nothing works!
   ```bash
   python scripts/train_tokenizer.py --txt_dirs data/txt --output models/spm
   ```

2. **Deploy Optimizations**
   ```bash
   python scripts/deploy_optimizations.py
   ```

3. **Validate Everything**
   ```bash
   ./scripts/validate_optimizations.sh
   ```

4. **Launch Training**
   ```bash
   # Local or TPU
   python src/aura/self_teaching_llm/build_aura_model.py pretrain_hf ...
   ```

---

## ✅ **Final Checklist**

### Before Training
- [x] Tokenizer trained and validated
- [x] Optimizations deployed
- [x] Validation tests passed
- [x] Training data accessible
- [x] Config parameters set
- [ ] Monitoring setup ready

### During Training
- [ ] Training started successfully
- [ ] Logs being generated
- [ ] Loss decreasing
- [ ] No errors in logs
- [ ] Performance metrics good

### After Training
- [ ] Model checkpoint saved
- [ ] Training completed without errors
- [ ] Performance meets expectations
- [ ] Model can load and run inference
- [ ] Documentation updated

---

## 🎉 **Success Criteria**

Your deployment is successful when:

✅ **Tokenizer**:
- Model file exists and is validated
- Can encode/decode text correctly
- Vocab size matches configuration

✅ **Optimizations**:
- All components deployed
- Validation tests passing
- Performance gains observed

✅ **Training**:
- Training runs without errors
- Loss decreases over time
- 2-3x speedup vs baseline
- Model checkpoints saved

✅ **Production**:
- Model can load and run
- Inference works correctly
- Performance is acceptable
- System is stable

---

## 📚 **Additional Resources**

- **Tokenizer Guide**: `TOKENIZER_TRAINING_GUIDE.md`
- **Optimization Details**: `OPTIMIZATION_IMPLEMENTATION_SUMMARY.md`
- **TPU Upgrade**: `TPU_OPTIMIZATION_UPGRADE_GUIDE.md`
- **Testing Guide**: `TESTING_SUMMARY.md`
- **Pre-Deployment**: `PRE_DEPLOYMENT_CHECKLIST.md`

---

## 🚀 **Quick Start Commands**

```bash
# Complete deployment in 4 commands:

# 1. Train tokenizer (CRITICAL!)
python scripts/train_tokenizer.py --txt_dirs data/txt --output models/spm --vocab_size 32000

# 2. Deploy optimizations
python scripts/deploy_optimizations.py --model-size medium

# 3. Validate
./scripts/validate_optimizations.sh

# 4. Train (local)
python src/aura/self_teaching_llm/build_aura_model.py pretrain_hf \
  --spm_model models/spm/spiece.model \
  --dataset allenai/c4 \
  --steps 1000

# Or train (TPU)
./scripts/gcloud_tpu_multihost_optimized.sh create
./scripts/gcloud_tpu_multihost_optimized.sh setup
./scripts/gcloud_tpu_multihost_optimized.sh launch_optimized
```

---

**Remember**: The tokenizer is **mandatory** - train it first! 🎯
