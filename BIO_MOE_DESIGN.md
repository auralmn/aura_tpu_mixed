# Bio-MoE: Brain-Inspired Modular Experts with Continuous Learning

## Goals
- Achieve robust performance with bio-inspired modules on real tasks (e.g., MNIST, LLM routing).
- Enable dynamic expert growth, cloning, and specialization across training.
- Support zone-based organization (hippocampus, amygdala, hypothalamus, thalamus, language) with interoperable signals.

## Core Concepts
- Bio-signal fusion: PhasorBankJAX (temporal resonance) + SpikingAttentionJAX (k-WTA attention) modulate feature space.
- Neural-signal bus (concept): share phasor/attention/zone stats; integrate with ThalamicGradientBroadcasterJAX for gradient routing.
- Zone-centric design: Each “brain zone” maintains its own expert registry, growth schedule, and checkpoints.

## Architecture
- EnhancedSpikingRetrievalCore (Liquid-MoE):
  - Experts: list of submodules (MLPExpert, Conv1DExpert; extensible to CNN/Transformer).
  - Gating: softmax over experts using bio-signal features.
  - Controls:
    - `active_experts`: restricts selection domain to enable staged growth.
    - `freeze_mask`: per-expert gradient control (others stop_gradient).
  - Utilities:
    - `compute_gate_weights(x)`: routing-only pass.
    - `expert_outputs(x)`: stacked expert outputs for distillation.
    - `distill_loss(x, teacher_idx, student_idx)`: MSE teacher→student.

- Expert Registry (`expert_registry.py`):
  - Zone presets: hippocampus, amygdala, hypothalamus, thalamus, language.
  - `build_core_kwargs_for_zone(zone, hidden_dim, freeze_experts)`: create core args.
  - `expert_ckpt_path(zone, idx)`: checkpoint path helper.

- MNIST Expert POC (`mnist_expert_poc.py`):
  - Pipeline: input → proj → phasor map → attention gains → core (growth + routing) → classifier.
  - Growth schedule (example): 1 → 3 → 6 experts.
  - Routed learning: only routed expert updates per batch.
  - Cloning/Teaching: at growth boundaries, new expert is distilled from a teacher (alpha=1.0 bootstrap, then alpha=0.5).
  - Metrics: per-epoch utilization distribution, accuracy.

## Features
- Dynamic capacity growth without reinitializing the model.
- Expert cloning & distillation for fast specialization of new capacity.
- Per-batch routed gradient flow (local specialization).
- Zone-based expert registries (plug-and-play across tasks/zones).
- JAX/Flax-native implementation with vectorized masking and jit-safe controls.

## Innovations
- Bio-signal-driven gating (phasor + spiking attention) aligned to hidden feature space.
- Distillation-driven capacity scaling: experts bootstrap from stronger peers at activation time.
- Fine-grained freeze control via `freeze_mask` enabling targeted continuous learning.
- Growth schedules per zone (e.g., hippocampus grows for memory tasks; thalamus routes across zones).

## Intended Extensions
- Per-expert checkpointing: periodic save/load of individual expert params from registry paths.
- Routing analytics: gate entropy, routed accuracy per expert/zone, lifetime utilization.
- Multi-zone integration: thalamus-mediated inter-zone routing, shared signal bus, gradient broadcasting.
- LLM path: language experts (transformer heads/decoders), integrated with SelfTeachingAdapter.

## TPU/Production Notes
- Pretrain experts per zone on TPU; freeze or LoRA-adapt in routing pipelines.
- Lazy-loading experts from registry to manage memory.
- Periodic expert checkpointing for online continual learning.

## Status
- MNIST real test fixed (CSV) with bio-signal attention alignment.
- Expert registry added; growth + cloning + routed learning validated (MNIST ≥95%).
- Next: add per-expert checkpoint IO + multi-zone LLM integration.

## New Additions (Oct 28, 2025)
- SRWKV backend option for SpikingLanguageCore (config `lang_backend: 'lif'|'srwkv'`).
- PersonalityEngineJAX providing per-batch controls:
  - bias_logits (routing bias), temperature (gating sharpness), distill_alpha, merit_momentum.
- MeritBoard extended for live-RAG with dynamic momentum; integrated into MNIST POC training.
- LLM demo wired to ingest local text corpus (data/txt) with affect vectors.
- Retrieval core softmax supports temperature and merit/personality bias.
- Hierarchical group gating added to EnhancedSpikingRetrievalCore (`group_count`), implemented with METAL-friendly group bias.
- Neurogenesis triggers in MNIST POC: entropy-based spawn + short clone/teach phase to specialize new experts.
- RoPE (rotary positional embeddings) integrated in LLM retrieval inputs and inside SpikingLanguageCore (per internal time step) when enabled via `SelfTeachingAdapter(use_rope=True, rope_max_len=2048, rope_base=10000.0)`.
- EMC-style regularization (soft freeze): output-space penalty on non-selected experts to reduce interference (`--emc-lambda`).
- Thalamic feedback: top-down global bias to gating via `thalamic_head` (`--thalamic-scale`).
- Predictive gating head: optional head to bias gate logits based on predicted per-expert utility (`predictive_weight`).
- Soft mixture routing (top-k): route to k experts (default k=2) and normalize weights.
- Committee distillation: ensemble teacher from top-k experts for more robust student specialization.
- Pruning: low-utilization experts masked from gating; inactive mask supported in routing paths.
- Routed analytics: per-expert routed accuracy, gate entropy, thalamic magnitude logged each epoch.

### CLI (MNIST POC)
- `--emc-lambda` (default 1e-4): strength of soft freeze penalty.
- `--predictive-weight` (default 0.0): strength of predictive gating bias.
- `--thalamic-scale` (default 1.0): scale of thalamic bias added to gate logits.
- Existing: bandit policy, growth/neurogenesis thresholds, batch sizes.

### Tokenizer & Corpus (SentencePiece)
- Trainer script: `src/aura/self_teaching_llm/tokenizer_spm.py`
- Cleaning: removes null/control chars and normalizes spaces before training.
- Streaming: optional iterator path avoids writing a temporary corpus file.
- Recommended command:

```
python3 src/aura/self_teaching_llm/tokenizer_spm.py \
  --input_dir data/txt \
  --out_dir models/spm \
  --vocab_size 1200 \
  --model_type unigram \
  --max_sentence_length 1000000 \
  --hard_vocab_limit 0 \
  --byte_fallback 1 \
  --clean_controls 1 \
  --normalize_spaces 1 \
  --use_iterator 1
```

- Verify piece size:

```
python3 - <<'PY'
import sentencepiece as sp
p = sp.SentencePieceProcessor(); p.load('models/spm/spiece.model')
print('pieces:', p.get_piece_size())
PY
```

- Adapter alignment: set `SelfTeachingAdapter(spm_model_path='models/spm/spiece.model')` and ensure vocab sizes match the tokenizer `piece_size` when constructing embeddings.

## How to run (tuning)
- Default (METAL): `python3 src/aura/training/mnist_expert_poc.py`
- With added controls:
  - `--emc-lambda 2e-4 --thalamic-scale 0.5 --predictive-weight 0.2`
- Inspect logs for: utilization, H (entropy), thal (thalamic magnitude), pruned mask, routed_acc.

## TPU v4 Pretraining Quickstart (Instruction Tuning)
- Install TPU-enabled JAX on TPU VMs:

```
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -U flax optax sentencepiece
```

- Recommended dtype on TPU: `bf16`
- Single-host run on TPU (compile + per-step logs):

```
JAX_PLATFORMS=tpu \
python3 src/aura/self_teaching_llm/train_instruct.py \
  --jsonl data/json/instruct_55k.jsonl \
  --spm-model models/spm/spiece.model \
  --steps 1000 \
  --batch-size 128 \
  --accumulate-steps 4 \
  --max-len 256 --pad-to 256 \
  --dtype bf16 \
  --pmap --per-device-batch 8
```

- Multihost (v4-32) run: launch the same command on each host with distributed init.
  - Choose a coordinator: `<COORD_IP:PORT>` on host 0.
  - Set process count and process index for each host.

```
JAX_PLATFORMS=tpu \
python3 src/aura/self_teaching_llm/train_instruct.py \
  --jsonl data/json/instruct_55k.jsonl \
  --spm-model models/spm/spiece.model \
  --steps 2000 \
  --batch-size 128 --accumulate-steps 4 \
  --max-len 256 --pad-to 256 \
  --dtype bf16 \
  --pmap --per-device-batch 8 \
  --init-distributed \
  --coordinator <COORD_IP:PORT> \
  --process-count <N_PROCS> \
  --process-index <THIS_PROC_INDEX>
```

Notes:
- Logging and checkpointing should be gated to process 0. This code already prints progress from process 0 only.
- Shapes must be static for best compile speed: keep `--batch-size`, `--per-device-batch`, `--max-len`, and `--pad-to` fixed.
- If compile appears stalled, set `JAX_LOG_COMPILES=1`, or temporarily `JAX_PLATFORMS=cpu` to isolate logic.

## Planned Next
- Hierarchical MoE (group→expert) and neurogenesis hooks for dynamic expert creation.
- Bandit policies in MeritBoard (UCB/Exp3) informed by ExpertEvaluator concepts.
- Optional neuromorphic tokenizer/embedding in SelfTeachingAdapter.

## Results (Oct 28, 2025, METAL GPU)
- Device: METAL AMD Radeon RX Vega 64
- MNIST Expert POC (hierarchical gating + bandit merit + personality engine)
  - Epoch 0: experts_active=1, util=[1.0, 0, 0, 0, 0, 0], loss=0.4177, acc=0.9345
  - Epoch 1: experts_active=1, util=[1.0, 0, 0, 0, 0, 0], loss=0.1962, acc=0.9496
  - Epoch 2: experts_active=3, util≈[0.988, 0.009, 0.003, 0, 0, 0], loss=0.8527, acc=0.9472
  - Epoch 3: experts_active=3, util≈[0.998, 0.001, 0.001, 0, 0, 0], loss=1.1352, acc=0.9517
  - Target 95% reached at epoch 3

## Build entrypoint (train, compose, chat)
- Script: `src/aura/self_teaching_llm/build_aura_model.py`
- Subcommands:
  - `tokenizer` – train SentencePiece from data/txt (iterator, cleaning, ASCII)
  - `pretrain` – instruction tuning (masked loss) on CPU/GPU/TPU (bf16, minibatch, accumulation, optional pmap)
  - `pretrain_zones` – orchestrate zone-specific pretraining and save heads/experts
  - `chat` – load adapter + zone heads and generate

Examples:

```
python3 src/aura/self_teaching_llm/build_aura_model.py tokenizer \
  --input_dir data/txt --out_dir models/spm --vocab_size 2000 \
  --use_iterator 1 --ascii_only 1 --user_symbols "<INST>,<INP>,<RESP>,<SEP>"

JAX_PLATFORMS=tpu \
python3 src/aura/self_teaching_llm/build_aura_model.py pretrain \
  --jsonl data/json/instruct_55k.jsonl --spm_model models/spm/spiece.model \
  --steps 1000 --batch_size 128 --accumulate_steps 4 \
  --max_len 256 --pad_to 256 --dtype bf16 --pmap --per_device_batch 8 \
  --ckpt_out models/aura/adapter_ckpt.pkl
```

## Zone pretraining (hippocampus, language, amygdala, thalamus, hypothalamus)
- Orchestrator:

```
python3 src/aura/self_teaching_llm/build_aura_model.py pretrain_zones \
  --zones hippocampus,language,amygdala,thalamus,hypothalamus \
  --txt_dir data/txt --json_dir data/json \
  --spm_model models/spm/spiece.model \
  --max_len 256 --pad_to 256 \
  --batch_size 128 --accumulate_steps 4 --dtype bf16 \
  --pmap --per_device_batch 8 \
  --ckpt_root checkpoints \
  --ckpt_out models/aura/adapter_ckpt.pkl
```

- Outputs saved under `checkpoints/`:
  - `hippocampus/` expert_i.msgpack (MNIST POC)
  - `amygdala/bias_head.msgpack` (affect -> bias head)
  - `thalamus/gate_head.msgpack` (routing centroids)
  - `hypothalamus/control_head.msgpack` (temperature, momentum)

## Live composition (inference)
- Load adapter checkpoint + zone heads and chat:

```
python3 src/aura/self_teaching_llm/build_aura_model.py chat \
  --spm_model models/spm/spiece.model \
  --ckpt models/aura/adapter_ckpt.pkl \
  --ckpt_root checkpoints \
  --prompt "Hello!" --max_len 256 --gen_len 64 --temperature 1.0
```

Notes:
- Thalamus centroids provide a gate bias from negative squared distance in embedding space.
- Hypothalamus control head sets generation temperature; amygdala head is prepared for routing bias.
