#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os, json, argparse, time
import numpy as np

# Verify NumPy version (JAX 0.4.31 requires NumPy >= 2.0)
if int(np.__version__.split('.')[0]) < 2:
    raise RuntimeError(f"NumPy 2.0+ required (found {np.__version__}). Install with: python3.12 -m pip install -U 'numpy>=2.0.0' --user")

import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from flax.training import train_state
from flax import serialization
import optax
from jax import distributed as jdist
from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

PLUTCHIK_LABELS = ['joy','trust','fear','surprise','sadness','disgust','anger','anticipation']
COMPASS_INTENTS = ['inform','negotiate','question','clarify','social','express','command','request']
INTENT_MAPPING = {
    'share_news': 'inform', 'ask_help': 'request', 'clarify': 'clarify',
    'complain': 'express', 'thank': 'social', 'propose': 'negotiate'
}
TONE_TO_PROSODY = {
    'ecstatic': {'energy': 0.95, 'pitch_var': 0.9, 'tempo': 1.3},
    'urgent': {'energy': 0.9, 'pitch_var': 0.8, 'tempo': 1.4},
    'neutral': {'energy': 0.5, 'pitch_var': 0.4, 'tempo': 1.0},
}

class ConsciousnessAwareSNN(nn.Module):
    num_experts: int = 5
    hidden_dim: int = 256
    sbert_dim: int = 384
    sp_vocab_size: int = 32000

    @nn.compact
    def __call__(self, sbert_embeddings, pos_tags, syntax_features,
                 sp_token_ids, sp_wb, sp_punct, sp_sublen,
                 training=True):
        # SentencePiece token embeddings
        sp_embed = nn.Embed(num_embeddings=self.sp_vocab_size, features=128, name='sp_token_embeddings')(sp_token_ids)
        # Normalize lengths
        max_len = jnp.maximum(jnp.max(sp_sublen, axis=1, keepdims=True), 1.0)
        len_norm = sp_sublen / max_len
        # Boundary/punctuation features context
        wb_prev = jnp.roll(sp_wb, shift=1, axis=1)
        wb_next = jnp.roll(sp_wb, shift=-1, axis=1)
        pn_prev = jnp.roll(sp_punct, shift=1, axis=1)
        pn_next = jnp.roll(sp_punct, shift=-1, axis=1)
        ling_feats = jnp.stack([sp_wb, wb_prev, wb_next, sp_punct, pn_prev, pn_next, len_norm], axis=-1)  # [B,128,7]
        # Pause prediction
        pause_input = jnp.concatenate([ling_feats, jnp.mean(sp_embed, axis=-1, keepdims=True)], axis=-1)
        pause_h = nn.gelu(nn.Dense(32, name='pause_dense1')(pause_input))
        pause_logits = nn.Dense(1, name='pause_predictor')(pause_h)
        pause_probs = nn.sigmoid(pause_logits).squeeze(-1)  # [B,128]
        # Stress prediction
        stress_input = jnp.concatenate([ling_feats, sp_embed], axis=-1)
        stress_h = nn.gelu(nn.Dense(32, name='stress_dense1')(stress_input))
        stress_logits = nn.Dense(1, name='stress_predictor')(stress_h)
        stress_probs = nn.sigmoid(stress_logits).squeeze(-1)  # [B,128]
        # Aggregate sentence-level features
        stress_var = jnp.std(stress_probs, axis=1, keepdims=True)
        stress_mean = jnp.mean(stress_probs, axis=1, keepdims=True)
        wb_density = jnp.mean(sp_wb, axis=1, keepdims=True)
        punct_count = jnp.sum(sp_punct, axis=1, keepdims=True)
        total_pauses = jnp.sum(pause_probs, axis=1, keepdims=True)
        total_stress = jnp.sum(stress_probs, axis=1, keepdims=True)
        pitch_input = jnp.concatenate([stress_var, stress_mean, wb_density], axis=-1)
        pitch = nn.gelu(nn.Dense(64, name='pitch_encoder')(pitch_input))
        energy_input = jnp.concatenate([total_stress, total_pauses, punct_count], axis=-1)
        energy = nn.gelu(nn.Dense(64, name='energy_encoder')(energy_input))
        # Legacy spaCy branches (optional signals)
        pauses_legacy = nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(32)(syntax_features)))).squeeze(-1)
        stress_legacy = nn.sigmoid(nn.Dense(1)(nn.relu(nn.Dense(32)(pos_tags)))).squeeze(-1)
        # Emotion and intent heads
        emotion_h = nn.relu(nn.Dense(128)(jnp.concatenate([sbert_embeddings, pitch, energy], axis=-1)))
        plutchik_probs = nn.softmax(nn.Dense(8)(emotion_h))
        intent_h = nn.relu(nn.Dense(128)(jnp.concatenate([sbert_embeddings, emotion_h, pitch], axis=-1)))
        primary_intent = nn.softmax(nn.Dense(8)(intent_h))
        urgency = nn.sigmoid(nn.Dense(1)(intent_h))
        certainty = nn.sigmoid(nn.Dense(1)(intent_h))
        formality = nn.sigmoid(nn.Dense(1)(intent_h))
        politeness = nn.sigmoid(nn.Dense(1)(intent_h))
        # Gating and output
        composite = jnp.concatenate([sbert_embeddings, emotion_h, intent_h, pitch, energy], axis=-1)
        gate_weights = nn.softmax(nn.Dense(self.num_experts)(composite))
        output = nn.Dense(self.hidden_dim)(composite)
        return {
            'output': output,
            'emotions': {'plutchik': plutchik_probs},
            'intent': {
                'primary_intent': primary_intent,
                'modifiers': {
                    'urgency': urgency,
                    'certainty': certainty,
                    'formality': formality,
                    'politeness': politeness,
                }
            },
            'gate_weights': gate_weights,
            'prosody': {
                'pause_probs': pause_probs,
                'stress_probs': stress_probs,
                'pitch': pitch,
                'energy': energy,
            }
        }

def load_emotion_dataset(jsonl_path: str):
    recs = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs

def preprocess(records, sbert_model, sp_model_path: str | None = None):
    sp = None
    if sp_model_path:
        try:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(sp_model_path)
            print(f"Loaded SentencePiece model: {sp_model_path} (vocab={sp.get_piece_size()})")
        except Exception as e:
            print(f"Warning: Failed to load SentencePiece model: {e}. Falling back to spaCy.")
            sp = None
    nlp = None
    if sp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    processed = []
    for idx, r in enumerate(records):
        text = r.get('text', '')
        emb = sbert_model.encode(text, convert_to_tensor=False)
        pos = np.zeros((128, 10), dtype=np.float32)
        syn = np.zeros((128, 3), dtype=np.float32)
        if sp is not None:
            ids = sp.encode(text, out_type=int)
            pieces = sp.encode(text, out_type=str)
            # special tokens not strictly needed here; trim/pad to 128
            max_len = 128
            ids = ids[:max_len]
            pieces = pieces[:max_len]
            orig_len = len(ids)
            if orig_len < max_len:
                ids += [sp.pad_id()] * (max_len - orig_len)
                pieces += ['<pad>'] * (max_len - orig_len)
            # word boundaries (▁) and punctuation
            wb = np.zeros((max_len,), dtype=np.float32)
            pn = np.zeros((max_len,), dtype=np.float32)
            sl = np.zeros((max_len,), dtype=np.float32)
            punct_set = {'.','!','? ',',',';',';',':','...','!!','??'}
            for i in range(max_len):
                p = pieces[i]
                wb[i] = 1.0 if p.startswith('▁') else 0.0
                pn[i] = 1.0 if p in punct_set else 0.0
                sl[i] = float(len(p.replace('▁','')))
            # pack into existing shapes: pos_tags (128,10), syntax_features (128,3)
            # pos_tags: first 3 dims = [wb, pn, normalized_length]
            if sl.max() > 0:
                sl_norm = sl / sl.max()
            else:
                sl_norm = sl
            pos[:,0] = wb; pos[:,1] = pn; pos[:,2] = sl_norm
            # syntax_features: replicate core features
            syn[:,0] = sl_norm
            syn[:,1] = pn
            syn[:,2] = wb
        else:
            doc = nlp(text)
            pm = {'NOUN':0,'VERB':1,'ADJ':2,'ADV':3,'PRON':4,'DET':5,'ADP':6,'CONJ':7,'NUM':8,'PUNCT':9}
            for i, tok in enumerate(list(doc)[:128]):
                if tok.pos_ in pm: pos[i, pm[tok.pos_]] = 1.0
                syn[i,0] = min(abs(tok.head.i - tok.i),10)/10.0
                syn[i,1] = 1.0 if tok.is_punct else 0.0
                syn[i,2] = 1.0 if tok.is_stop else 0.0
        p = np.zeros(8, dtype=np.float32)
        prim = r.get('plutchik',{}).get('primary','joy')
        inten = float(r.get('plutchik',{}).get('intensity',0.5))
        if prim in PLUTCHIK_LABELS: p[PLUTCHIK_LABELS.index(prim)] = inten
        sec = r.get('plutchik',{}).get('secondary')
        sec_map = {'optimism':'anticipation','admiration':'trust','anxiety':'fear','hope':'anticipation','excitement':'joy','contentment':'joy','grief':'sadness','despair':'sadness','contempt':'disgust','outrage':'anger','fury':'anger','resentment':'anger'}
        if sec in sec_map: p[PLUTCHIK_LABELS.index(sec_map[sec])] += 0.25
        p = p / (np.sum(p)+1e-6)
        mapped = INTENT_MAPPING.get(r.get('intent','inform'),'inform')
        intent_idx = COMPASS_INTENTS.index(mapped)
        intent_oh = np.zeros(8, dtype=np.float32); intent_oh[intent_idx]=1.0
        style = r.get('style',{})
        beta = float(style.get('beta',0.5)); phi=float(style.get('phi',0.5))
        urgency = inten if inten>0.6 else inten*0.7; certainty = phi if phi>0 else 0.5
        sp_token_ids = np.zeros((128,), dtype=np.int32)
        sp_wb = np.zeros((128,), dtype=np.float32)
        sp_punct = np.zeros((128,), dtype=np.float32)
        sp_sublen = np.zeros((128,), dtype=np.float32)
        if sp is not None:
            sp_token_ids = np.array(ids[:128], dtype=np.int32)
            sp_wb = wb.astype(np.float32)
            sp_punct = pn.astype(np.float32)
            sp_sublen = sl.astype(np.float32)
        processed.append({
            'sbert_embedding': emb.astype(np.float32),
            'pos_tags': pos,
            'syntax_features': syn,
            'sp_token_ids': sp_token_ids,
            'sp_wb': sp_wb,
            'sp_punct': sp_punct,
            'sp_sublen': sp_sublen,
            'plutchik_probs': p,
            'intent_label': intent_oh,
            'urgency': urgency,
            'certainty': certainty,
            'formality': beta,
            'politeness': phi,
        })
    return processed

@jit
def train_step(state, batch, num_classes_emotion: int = 8, num_classes_intent: int = 8, label_smoothing: float = 0.0, diversity_coef: float = 0.02):
    def smooth_labels(y, n_classes):
        return (1.0 - label_smoothing) * y + label_smoothing / n_classes
    def loss_fn(p):
        out = state.apply_fn(
            {'params': p},
            batch['sbert_embedding'],
            batch['pos_tags'],
            batch['syntax_features'],
            batch['sp_token_ids'],
            batch['sp_wb'],
            batch['sp_punct'],
            batch['sp_sublen'],
            training=True
        )
        # Label smoothing
        emo_targets = smooth_labels(batch['plutchik_probs'], num_classes_emotion)
        intent_targets = smooth_labels(batch['intent_label'], num_classes_intent)
        el = optax.softmax_cross_entropy(out['emotions']['plutchik'], emo_targets).mean()
        il = optax.softmax_cross_entropy(out['intent']['primary_intent'], intent_targets).mean()
        m = out['intent']['modifiers']
        ml = ((m['urgency']-batch['urgency'])**2 + (m['certainty']-batch['certainty'])**2 + (m['formality']-batch['formality'])**2 + (m['politeness']-batch['politeness'])**2).mean()
        gw = out['gate_weights']; div = -jnp.mean(jnp.sum(gw * jnp.log(gw + 1e-8), axis=-1))
        total = 1.0*el + 1.0*il + 0.5*ml + diversity_coef*div
        return total, {'loss': total, 'emotion': el, 'intent': il, 'modifiers': ml, 'diversity': -div}
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    return state.apply_gradients(grads=grads), metrics

def batches(data, bs=128, shuffle=True):
    idx = np.arange(len(data));
    if shuffle: np.random.shuffle(idx)
    for s in range(0, len(idx), bs):
        sel = idx[s:s+bs]; d=[data[i] for i in sel]
        yield {
            'sbert_embedding': jnp.array([x['sbert_embedding'] for x in d]),
            'pos_tags': jnp.array([x['pos_tags'] for x in d]),
            'syntax_features': jnp.array([x['syntax_features'] for x in d]),
            'sp_token_ids': jnp.array([x.get('sp_token_ids', np.zeros((128,), np.int32)) for x in d], dtype=jnp.int32),
            'sp_wb': jnp.array([x.get('sp_wb', np.zeros((128,), np.float32)) for x in d]),
            'sp_punct': jnp.array([x.get('sp_punct', np.zeros((128,), np.float32)) for x in d]),
            'sp_sublen': jnp.array([x.get('sp_sublen', np.zeros((128,), np.float32)) for x in d]),
            'plutchik_probs': jnp.array([x['plutchik_probs'] for x in d]),
            'intent_label': jnp.array([x['intent_label'] for x in d]),
            'urgency': jnp.array([x['urgency'] for x in d]).reshape(-1,1),
            'certainty': jnp.array([x['certainty'] for x in d]).reshape(-1,1),
            'formality': jnp.array([x['formality'] for x in d]).reshape(-1,1),
            'politeness': jnp.array([x['politeness'] for x in d]).reshape(-1,1),
        }

def main():
    parser = argparse.ArgumentParser(description='TPU v4-32 training for Emotion+Intent (SBERT-based)')
    parser.add_argument('--data', required=True, help='Path to emotions.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--sp-model', default=os.environ.get('SP_MODEL', ''), help='Optional SentencePiece .model path')
    # Multi-host TPU flags (or via env: COORDINATOR_ADDRESS, NUM_PROCESSES, PROCESS_ID)
    parser.add_argument('--coordinator-address', default=os.environ.get('COORDINATOR_ADDRESS', 'localhost:12355'))
    parser.add_argument('--num-processes', type=int, default=int(os.environ.get('NUM_PROCESSES', '1')))
    parser.add_argument('--process-id', type=int, default=int(os.environ.get('PROCESS_ID', '0')))
    parser.add_argument('--ckpt-dir', default=os.environ.get('CKPT_DIR', ''), help='Checkpoint directory (optional)')
    parser.add_argument('--ckpt-every', type=int, default=0, help='Save checkpoint every N epochs (0=disable)')
    parser.add_argument('--num-experts', type=int, default=int(os.environ.get('NUM_EXPERTS', '8')))
    parser.add_argument('--diversity-coef', type=float, default=float(os.environ.get('DIVERSITY_COEF', '0.05')))
    parser.add_argument('--label-smoothing', type=float, default=float(os.environ.get('LABEL_SMOOTHING', '0.05')))
    parser.add_argument('--final-lr', type=float, default=float(os.environ.get('FINAL_LR', '1e-4')))
    args = parser.parse_args()

    # Initialize JAX distributed for TPU pods (run on ALL hosts with unique process_id)
    if args.num_processes > 1:
        print(f"Initializing JAX distributed: coord={args.coordinator_address}, num_processes={args.num_processes}, process_id={args.process_id}")
        jdist.initialize(coordinator_address=args.coordinator_address,
                         num_processes=args.num_processes,
                         process_id=args.process_id)
    print(f"Devices (pid {args.process_id}/{args.num_processes}): {jax.devices()}")
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    sbert_model = SentenceTransformer(args.model)

    print(f"Loading dataset: {args.data}")
    records = load_emotion_dataset(args.data)
    print(f"Records: {len(records)}")

    # Split 80/10/10
    from sklearn.model_selection import train_test_split
    train_records, temp_records = train_test_split(records, test_size=0.2, random_state=42)
    val_records, test_records = train_test_split(temp_records, test_size=0.5, random_state=42)

    print("Preprocessing...")
    sp_path = args.sp_model if args.sp_model else None
    train_processed = preprocess(train_records, sbert_model, sp_model_path=sp_path)
    val_processed   = preprocess(val_records, sbert_model, sp_model_path=sp_path)

    rng = random.PRNGKey(42)
    model = ConsciousnessAwareSNN(num_experts=args.num_experts)
    params = model.init(
        {'params': rng},
        jnp.ones((2,384)),
        jnp.ones((2,128,10)),
        jnp.ones((2,128,3)),
        jnp.zeros((2,128), dtype=jnp.int32),
        jnp.zeros((2,128)),
        jnp.zeros((2,128)),
        jnp.zeros((2,128)),
        training=False
    )['params']

    steps = (max(1, len(train_processed)//args.batch_size))*args.epochs
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(10, steps//20), steps, args.final_lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    def save_checkpoint(epoch_idx: int, final: bool=False):
        if args.process_id != 0:
            return
        if not args.ckpt_dir:
            return
        ckpt_dir = Path(args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = 'final' if final else f'epoch_{epoch_idx:04d}'
        out_path = ckpt_dir / f'ckpt_{tag}.msgpack'
        data = serialization.to_bytes(state.params)
        with open(out_path, 'wb') as f:
            f.write(data)
        print(f"Saved checkpoint: {out_path}")

    @jit
    def eval_step(params, batch):
        out = model.apply({'params': params},
                          batch['sbert_embedding'], batch['pos_tags'], batch['syntax_features'],
                          batch['sp_token_ids'], batch['sp_wb'], batch['sp_punct'], batch['sp_sublen'],
                          training=False)
        emo_targets = (1.0 - args.label_smoothing) * batch['plutchik_probs'] + args.label_smoothing / 8
        intent_targets = (1.0 - args.label_smoothing) * batch['intent_label'] + args.label_smoothing / 8
        el = optax.softmax_cross_entropy(out['emotions']['plutchik'], emo_targets).mean()
        il = optax.softmax_cross_entropy(out['intent']['primary_intent'], intent_targets).mean()
        m = out['intent']['modifiers']
        ml = ((m['urgency']-batch['urgency'])**2 + (m['certainty']-batch['certainty'])**2 + (m['formality']-batch['formality'])**2 + (m['politeness']-batch['politeness'])**2).mean()
        gw = out['gate_weights']; div = -jnp.mean(jnp.sum(gw * jnp.log(gw + 1e-8), axis=-1))
        total = 1.0*el + 1.0*il + 0.5*ml + args.diversity_coef*div
        return total

    print("Starting training...")
    t0 = time.time()
    best_val = None
    for epoch in range(args.epochs):
        metrics_buf = []
        for step, batch in enumerate(batches(train_processed, bs=args.batch_size, shuffle=True)):
            state, metrics = train_step(state, batch, label_smoothing=args.label_smoothing, diversity_coef=args.diversity_coef)
            metrics_buf.append(metrics)
            if (step+1) % 10 == 0 and (args.process_id == 0):
                avg = jnp.mean(jnp.array([m['loss'] for m in metrics_buf[-10:]]))
                print(f"  epoch {epoch+1} step {step+1}: loss={float(avg):.4f}")
        if args.process_id == 0:
            avg_epoch = jnp.mean(jnp.array([m['loss'] for m in metrics_buf]))
            # Validation
            val_losses = []
            for vb in batches(val_processed, bs=args.batch_size, shuffle=False):
                val_losses.append(eval_step(state.params, vb))
            val_loss = float(jnp.mean(jnp.array(val_losses))) if val_losses else float(avg_epoch)
            print(f"Epoch {epoch+1}: train_loss={float(avg_epoch):.4f} val_loss={val_loss:.4f}")
            if args.ckpt_dir:
                if best_val is None or val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(epoch + 1, final=False)
        if args.ckpt_dir and args.ckpt_every > 0 and ((epoch + 1) % args.ckpt_every == 0):
            save_checkpoint(epoch + 1, final=False)
    # final checkpoint
    if args.ckpt_dir:
        save_checkpoint(args.epochs, final=True)
    dt = time.time()-t0
    if args.process_id == 0:
        print(f"Done. Elapsed {dt/60:.2f} min")

if __name__ == '__main__':
    main()
