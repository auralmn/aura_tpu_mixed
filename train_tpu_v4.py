#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os, json, argparse, time
import numpy as np

# Verify NumPy version (JAX 0.4.31 requires NumPy >= 2.0)
if int(np.__version__.split('.')[0]) < 2:
    raise RuntimeError(f"NumPy 2.0+ required (found {np.__version__}). Install with: python3.12 -m pip install -U 'numpy>=2.0.0' --user")

import sys
# Avoid JAX TPU init warnings when running Torch/XLA path
if ('--torch-xla' in sys.argv) and (os.environ.get('JAX_PLATFORMS') is None):
    os.environ['JAX_PLATFORMS'] = 'cpu'
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
    from transformers import AutoTokenizer, FlaxAutoModel, AutoModel
except Exception:
    AutoTokenizer = None
    FlaxAutoModel = None
    AutoModel = None
try:
    import torch
    import torch.nn as tnn
    import torch.optim as topt
    import torch.utils.data as tdata
    import torch_xla as txla
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
except Exception:
    torch = None
    tnn = None
    topt = None
    tdata = None
    xm = None
    MpDeviceLoader = None

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
    sbert_adapter_dim: int = 0
    sbert_dropout: float = 0.0

    @nn.compact
    def __call__(self, sbert_embeddings, pos_tags, syntax_features,
                 sp_token_ids, sp_wb, sp_punct, sp_sublen,
                 training=True):
        # Optional SBERT adapter (trainable projection + dropout)
        if self.sbert_adapter_dim and self.sbert_adapter_dim > 0:
            x = nn.Dense(self.sbert_adapter_dim, name='sbert_adapter_dense')(sbert_embeddings)
            if self.sbert_dropout and self.sbert_dropout > 0:
                x = nn.Dropout(rate=self.sbert_dropout, deterministic=not training)(x)
            sbert_feat = nn.gelu(x)
        else:
            sbert_feat = sbert_embeddings
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
        emotion_h = nn.relu(nn.Dense(128)(jnp.concatenate([sbert_feat, pitch, energy], axis=-1)))
        plutchik_probs = nn.softmax(nn.Dense(8)(emotion_h))
        intent_h = nn.relu(nn.Dense(128)(jnp.concatenate([sbert_feat, emotion_h, pitch], axis=-1)))
        primary_intent = nn.softmax(nn.Dense(8)(intent_h))
        urgency = nn.sigmoid(nn.Dense(1)(intent_h))
        certainty = nn.sigmoid(nn.Dense(1)(intent_h))
        formality = nn.sigmoid(nn.Dense(1)(intent_h))
        politeness = nn.sigmoid(nn.Dense(1)(intent_h))
        # Gating and output
        composite = jnp.concatenate([sbert_feat, emotion_h, intent_h, pitch, energy], axis=-1)
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

def preprocess(records, sbert_model, sp_model_path: str | None = None,
               hf_tokenizer=None, sbert_max_len: int = 128):
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
        if hf_tokenizer is not None:
            tok = hf_tokenizer(text, max_length=sbert_max_len, truncation=True, padding='max_length', return_tensors=None)
            input_ids_np = np.array(tok['input_ids'], dtype=np.int32)
            attn_mask_np = np.array(tok['attention_mask'], dtype=np.int32)
            emb = None
        else:
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
            'sbert_embedding': (emb.astype(np.float32) if emb is not None else None),
            'sbert_input_ids': (input_ids_np if hf_tokenizer is not None else None),
            'sbert_attention_mask': (attn_mask_np if hf_tokenizer is not None else None),
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

def compute_grads(model, sbert_flax_module, use_flax_sbert, params_all, batch,
                  num_classes_emotion: int = 8, num_classes_intent: int = 8,
                  label_smoothing: float = 0.0, diversity_coef: float = 0.02):
    def smooth_labels(y, n_classes):
        return (1.0 - label_smoothing) * y + label_smoothing / n_classes
    def loss_fn(pa):
        # Compute SBERT embeddings
        if use_flax_sbert:
            outputs = sbert_flax_module(
                input_ids=batch['sbert_input_ids'],
                attention_mask=batch['sbert_attention_mask'],
                params=pa['sbert'],
                train=False
            )
            hidden = outputs.last_hidden_state  # [B,L,H] (bf16)
            mask = batch['sbert_attention_mask'].astype(jnp.float32)
            denom = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), a_min=1.0)
            sbert_emb = (jnp.sum(hidden * mask[..., None], axis=1) / denom).astype(jnp.float32)
        else:
            sbert_emb = batch['sbert_embedding']
        # Forward SNN
        out = model.apply(
            {'params': pa['snn']},
            sbert_emb,
            batch['pos_tags'],
            batch['syntax_features'],
            batch['sp_token_ids'],
            batch['sp_wb'],
            batch['sp_punct'],
            batch['sp_sublen'],
            training=True
        )
        # Losses
        emo_targets = smooth_labels(batch['plutchik_probs'], num_classes_emotion)
        intent_targets = smooth_labels(batch['intent_label'], num_classes_intent)
        el = optax.softmax_cross_entropy(out['emotions']['plutchik'], emo_targets).mean()
        il = optax.softmax_cross_entropy(out['intent']['primary_intent'], intent_targets).mean()
        m = out['intent']['modifiers']
        ml = ((m['urgency']-batch['urgency'])**2 + (m['certainty']-batch['certainty'])**2 + (m['formality']-batch['formality'])**2 + (m['politeness']-batch['politeness'])**2).mean()
        gw = out['gate_weights']; div = -jnp.mean(jnp.sum(gw * jnp.log(gw + 1e-8), axis=-1))
        total = 1.0*el + 1.0*il + 0.5*ml + diversity_coef*div
        return total, {'loss': total, 'emotion': el, 'intent': il, 'modifiers': ml, 'diversity': -div}
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params_all)
    return grads, metrics

def batches(data, bs=128, shuffle=True):
    idx = np.arange(len(data));
    if shuffle: np.random.shuffle(idx)
    for s in range(0, len(idx), bs):
        sel = idx[s:s+bs]; d=[data[i] for i in sel]
        # Build SBERT inputs: prefer HF ids/masks if present; else use embeddings
        has_hf = d[0].get('sbert_input_ids', None) is not None
        batch_dict = {
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
        if has_hf:
            batch_dict['sbert_input_ids'] = jnp.array([x['sbert_input_ids'] for x in d], dtype=jnp.int32)
            batch_dict['sbert_attention_mask'] = jnp.array([x['sbert_attention_mask'] for x in d], dtype=jnp.int32)
            batch_dict['sbert_embedding'] = None
        else:
            batch_dict['sbert_embedding'] = jnp.array([x['sbert_embedding'] for x in d])
            batch_dict['sbert_input_ids'] = None
            batch_dict['sbert_attention_mask'] = None
        yield batch_dict

def run_torch_xla(args):
    if torch is None or xm is None or AutoTokenizer is None or AutoModel is None:
        raise RuntimeError('torch/torch_xla/transformers not installed')
    import os
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    device = txla.device()
    try:
        xm.master_print(f"[XLA] Using device: {device}")
        xm.master_print(f"[XLA] Supported TPU devices: {xm.get_xla_supported_devices('TPU')}")
    except Exception:
        pass
    tokenizer = AutoTokenizer.from_pretrained(args.sbert_model_name, use_fast=True)
    sbert = AutoModel.from_pretrained(args.sbert_model_name).to(device)
    # Enable gradient checkpointing to save memory if supported
    if hasattr(sbert, 'gradient_checkpointing_enable'):
        sbert.gradient_checkpointing_enable()
    sbert.train()

    import sentencepiece as spm
    sp = None
    if args.sp_model:
        sp = spm.SentencePieceProcessor(); sp.load(args.sp_model)
        punct_set = set(['.', '!', '?', ',', ';', ':', '...', '!!', '??'])

    class JsonDataset(tdata.Dataset):
        def __init__(self, records):
            self.recs = records
        def __len__(self): return len(self.recs)
        def __getitem__(self, idx):
            r = self.recs[idx]; text = r.get('text','') or ''
            text = text.replace('\u00A0', ' ')
            tok = tokenizer(text, max_length=args.sbert_max_len, truncation=True, padding='max_length', return_tensors='pt')
            sample = {
                'input_ids': tok['input_ids'].squeeze(0),
                'attention_mask': tok['attention_mask'].squeeze(0)
            }
            if sp is not None:
                ids = sp.encode(text, out_type=int)[:128]
                pieces = sp.encode(text, out_type=str)[:128]
                pad = 128 - len(ids)
                if pad>0:
                    ids += [sp.pad_id()]*pad; pieces += ['<pad>']*pad
                wb = torch.tensor([1.0 if p.startswith('▁') else 0.0 for p in pieces], dtype=torch.float)
                pn = torch.tensor([1.0 if p in punct_set else 0.0 for p in pieces], dtype=torch.float)
                sl = torch.tensor([float(len(p.replace('▁',''))) for p in pieces], dtype=torch.float)
                sample.update({
                    'sp_token_ids': torch.tensor(ids, dtype=torch.long),
                    'sp_wb': wb, 'sp_punct': pn, 'sp_sublen': sl
                })
            # labels
            import numpy as np
            p = np.zeros(8, dtype=np.float32)
            prim = r.get('plutchik',{}).get('primary','joy')
            inten = float(r.get('plutchik',{}).get('intensity',0.5))
            labels = ['joy','trust','fear','surprise','sadness','disgust','anger','anticipation']
            if prim in labels: p[labels.index(prim)] = inten
            sec = r.get('plutchik',{}).get('secondary')
            sec_map = {'optimism':'anticipation','admiration':'trust','anxiety':'fear','hope':'anticipation','excitement':'joy','contentment':'joy','grief':'sadness','despair':'sadness','contempt':'disgust','outrage':'anger','fury':'anger','resentment':'anger'}
            if sec in sec_map: p[labels.index(sec_map[sec])] += 0.25
            p = p / (p.sum()+1e-6)
            intents = ['inform','negotiate','question','clarify','social','express','command','request']
            mapped = {'share_news':'inform','ask_help':'request','clarify':'clarify','complain':'express','thank':'social','propose':'negotiate'}.get(r.get('intent','inform'),'inform')
            oh = np.zeros(8, dtype=np.float32); oh[intents.index(mapped)] = 1.0
            style = r.get('style',{}); beta=float(style.get('beta',0.5)); phi=float(style.get('phi',0.5))
            sample.update({
                'plutchik': torch.tensor(p, dtype=torch.float),
                'intent': torch.tensor(oh, dtype=torch.float),
                'urgency': torch.tensor([inten], dtype=torch.float),
                'certainty': torch.tensor([phi if phi>0 else 0.5], dtype=torch.float),
                'formality': torch.tensor([beta], dtype=torch.float),
                'politeness': torch.tensor([phi], dtype=torch.float),
            })
            return sample

    records = load_emotion_dataset(args.data)
    from sklearn.model_selection import train_test_split
    train_records, temp_records = train_test_split(records, test_size=0.2, random_state=42)
    val_records, _ = train_test_split(temp_records, test_size=0.5, random_state=42)
    train_loader = tdata.DataLoader(JsonDataset(train_records), batch_size=args.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    val_loader = tdata.DataLoader(JsonDataset(val_records), batch_size=args.batch_size, shuffle=False, num_workers=0, persistent_workers=False)
    # Wrap with MpDeviceLoader for XLA stability
    train_loader = MpDeviceLoader(train_loader, device)
    val_loader = MpDeviceLoader(val_loader, device)

    class TorchSNN(tnn.Module):
        def __init__(self, sp_vocab=32000, sbert_dim=768, num_experts=4):
            super().__init__()
            self.sp_embed = tnn.Embedding(sp_vocab, 128)
            self.pitch = tnn.Sequential(tnn.Linear(3,64), tnn.GELU())
            self.energy = tnn.Sequential(tnn.Linear(3,64), tnn.GELU())
            self.emotion = tnn.Sequential(tnn.Linear(sbert_dim+64+64,128), tnn.ReLU(), tnn.Linear(128,8))
            self.intent_h = tnn.Sequential(tnn.Linear(sbert_dim+128+64,128), tnn.ReLU())
            self.intent = tnn.Linear(128,8)
            self.mod = tnn.Linear(128,4)
            self.gate = tnn.Linear(sbert_dim+128+128+64+64, num_experts)
            self.out = tnn.Linear(sbert_dim+128+128+64+64, 256)
        def forward(self, sbert_emb, batch):
            # SP features
            if 'sp_token_ids' in batch:
                sp_e = self.sp_embed(batch['sp_token_ids'])  # [B,128,128]
                wb = batch['sp_wb']; pn = batch['sp_punct']; sl = batch['sp_sublen']
                sln = sl / (sl.max(dim=1, keepdim=True).values.clamp(min=1.0))
                pitch = self.pitch(torch.stack([
                    torch.std((sp_e.mean(-1)), dim=1), torch.mean((sp_e.mean(-1)), dim=1), wb.mean(1)
                ], dim=1))
                energy = self.energy(torch.stack([
                    torch.sum(sln, dim=1), torch.sum(pn, dim=1), torch.sum(wb, dim=1)
                ], dim=1))
            else:
                pitch = torch.zeros((sbert_emb.size(0),64), device=sbert_emb.device)
                energy = torch.zeros((sbert_emb.size(0),64), device=sbert_emb.device)
            emo_h = torch.relu(self.emotion[0](torch.cat([sbert_emb, pitch, energy], dim=-1)))
            emo_logits = self.emotion[2](torch.relu(self.emotion[1](emo_h))) if len(self.emotion)==3 else self.emotion[-1](emo_h)
            intent_h = self.intent_h(torch.cat([sbert_emb, emo_h, pitch], dim=-1))
            intent_logits = self.intent(intent_h)
            mods = torch.sigmoid(self.mod(intent_h))
            comp = torch.cat([sbert_emb, emo_h, intent_h, pitch, energy], dim=-1)
            gate = torch.softmax(self.gate(comp), dim=-1)
            out = self.out(comp)
            return emo_logits, intent_logits, mods, gate, out

    model = TorchSNN(sp_vocab=32000, sbert_dim=sbert.config.hidden_size if hasattr(sbert,'config') else 768, num_experts=args.num_experts).to(device)
    optim = topt.AdamW(list(model.parameters())+list(sbert.parameters()), lr=args.lr, weight_decay=0.02)

    def smooth(y, n, eps): return (1-eps)*y + eps/n
    for epoch in range(args.epochs):
        model.train(); sbert.train()
        total=0.0; cnt=0
        optim.zero_grad()
        k=0
        first_step = True
        print(f"[XLA] Epoch {epoch+1} start. First step may compile (1–3 min)...", flush=True)
        for batch in train_loader:
            if first_step:
                print("[XLA] Compiling first step...", flush=True)
                first_step = False
            batch = {k:(v.to(device)) for k,v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = sbert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            hidden = outputs.last_hidden_state
            mask = batch['attention_mask'].float(); denom = mask.sum(1, keepdim=True).clamp(min=1.0)
            sbert_emb = (hidden * mask.unsqueeze(-1)).sum(1)/denom
            emo_logits, intent_logits, mods, gate, out = model(sbert_emb, batch)
            el = tnn.functional.cross_entropy(emo_logits, smooth(batch['plutchik'],8,args.label_smoothing))
            il = tnn.functional.cross_entropy(intent_logits, smooth(batch['intent'],8,args.label_smoothing))
            ml = tnn.functional.mse_loss(mods[:,0:1], batch['urgency']) + \
                 tnn.functional.mse_loss(mods[:,1:2], batch['certainty']) + \
                 tnn.functional.mse_loss(mods[:,2:3], batch['formality']) + \
                 tnn.functional.mse_loss(mods[:,3:4], batch['politeness'])
            div = -(gate * (gate.clamp(min=1e-8)).log()).sum(-1).mean()
            loss = el + il + 0.5*ml + args.diversity_coef*div
            loss.backward()
            k+=1
            if k % int(os.environ.get('GRAD_ACCUM_STEPS','1')) == 0:
                xm.optimizer_step(optim, barrier=True)
                xm.mark_step()
                if k == 1:
                    print("[XLA] First step compiled and executed.", flush=True)
                optim.zero_grad()
            total += loss.item(); cnt += 1
        if cnt>0: print(f"Epoch {epoch+1}: train_loss={total/cnt:.4f}")
        # val
        model.eval(); sbert.eval(); vloss=0.0; vcnt=0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k:(v.to(device)) for k,v in batch.items() if isinstance(v, torch.Tensor)}
                outputs = sbert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                hidden = outputs.last_hidden_state
                mask = batch['attention_mask'].float(); denom = mask.sum(1, keepdim=True).clamp(min=1.0)
                sbert_emb = (hidden * mask.unsqueeze(-1)).sum(1)/denom
                emo_logits, intent_logits, mods, gate, out = model(sbert_emb, batch)
                el = tnn.functional.cross_entropy(emo_logits, smooth(batch['plutchik'],8,args.label_smoothing))
                il = tnn.functional.cross_entropy(intent_logits, smooth(batch['intent'],8,args.label_smoothing))
                ml = tnn.functional.mse_loss(mods[:,0:1], batch['urgency']) + \
                     tnn.functional.mse_loss(mods[:,1:2], batch['certainty']) + \
                     tnn.functional.mse_loss(mods[:,2:3], batch['formality']) + \
                     tnn.functional.mse_loss(mods[:,3:4], batch['politeness'])
                div = -(gate * (gate.clamp(min=1e-8)).log()).sum(-1).mean()
                loss = el + il + 0.5*ml + args.diversity_coef*div
                vloss += loss.item(); vcnt += 1
                xm.mark_step()
        if vcnt>0: print(f"Epoch {epoch+1}: val_loss={vloss/vcnt:.4f}")
        if args.ckpt_dir and args.process_id == 0 and (best_val is None or (vcnt>0 and vloss/vcnt < best_val)):
            best_val = vloss/vcnt if vcnt>0 else None
            Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'snn': model.state_dict(), 'sbert': sbert.state_dict()}, str(Path(args.ckpt_dir)/f'ckpt_epoch_{epoch+1:04d}.pt'))


def main():
    parser = argparse.ArgumentParser(description='TPU v4-32 training for Emotion+Intent (SBERT-based)')
    parser.add_argument('--data', required=True, help='Path to emotions.jsonl')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--use-flax-sbert', action='store_true')
    parser.add_argument('--sbert-model-name', default=os.environ.get('SBERT_MODEL_NAME', 'sentence-transformers/all-mpnet-base-v2'))
    parser.add_argument('--sbert-max-len', type=int, default=int(os.environ.get('SBERT_MAX_LEN', '128')))
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
    parser.add_argument('--sbert-adapter-dim', type=int, default=int(os.environ.get('SBERT_ADAPTER_DIM', '256')))
    parser.add_argument('--sbert-dropout', type=float, default=float(os.environ.get('SBERT_DROPOUT', '0.1')))
    parser.add_argument('--torch-xla', action='store_true', help='Use PyTorch/XLA training pipeline')
    args = parser.parse_args()

    # Torch/XLA path first, skip JAX init/prints entirely
    if args.torch_xla:
        try:
            import torch_xla.core.xla_model as xm
            print(f"[XLA] Supported devices: {xm.get_xla_supported_devices('TPU')}")
        except Exception:
            pass
        return run_torch_xla(args)

    # Initialize JAX distributed for TPU pods (run on ALL hosts with unique process_id)
    if args.num_processes > 1:
        print(f"Initializing JAX distributed: coord={args.coordinator_address}, num_processes={args.num_processes}, process_id={args.process_id}")
        jdist.initialize(coordinator_address=args.coordinator_address,
                         num_processes=args.num_processes,
                         process_id=args.process_id)
    print(f"Devices (pid {args.process_id}/{args.num_processes}): {jax.devices()}")
    tokenizer = None
    sbert_flax = None
    sbert_model = None
    if args.use_flax_sbert:
        if AutoTokenizer is None or FlaxAutoModel is None:
            raise RuntimeError('transformers not installed')
        tokenizer = AutoTokenizer.from_pretrained(args.sbert_model_name, use_fast=True)
        # Use bfloat16 on TPU to reduce memory; from_pt=True enables Torch→Flax conversion
        sbert_flax = FlaxAutoModel.from_pretrained(args.sbert_model_name, dtype=jnp.bfloat16, from_pt=True)
    else:
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
    train_processed = preprocess(train_records, sbert_model, sp_model_path=sp_path,
                                 hf_tokenizer=tokenizer if args.use_flax_sbert else None,
                                 sbert_max_len=args.sbert_max_len)
    val_processed   = preprocess(val_records, sbert_model, sp_model_path=sp_path,
                                 hf_tokenizer=tokenizer if args.use_flax_sbert else None,
                                 sbert_max_len=args.sbert_max_len)

    rng = random.PRNGKey(42)
    model = ConsciousnessAwareSNN(num_experts=args.num_experts,
                                   sbert_adapter_dim=args.sbert_adapter_dim,
                                   sbert_dropout=args.sbert_dropout)
    # Determine SBERT embedding dim for init
    sbert_hidden_size = 384
    if args.use_flax_sbert and sbert_flax is not None and hasattr(sbert_flax, 'config'):
        sbert_hidden_size = int(getattr(sbert_flax.config, 'hidden_size', 768))
    snn_params = model.init(
        {'params': rng},
        jnp.ones((2, sbert_hidden_size)),
        jnp.ones((2,128,10)),
        jnp.ones((2,128,3)),
        jnp.zeros((2,128), dtype=jnp.int32),
        jnp.zeros((2,128)),
        jnp.zeros((2,128)),
        jnp.zeros((2,128)),
        training=False
    )['params']
    params_all = {'snn': snn_params, 'sbert': (sbert_flax.params if args.use_flax_sbert else {})}

    steps = (max(1, len(train_processed)//args.batch_size))*args.epochs
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, max(10, steps//20), steps, args.final_lr)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=schedule, weight_decay=0.01))
    state = train_state.TrainState.create(apply_fn=None, params=params_all, tx=tx)

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

    def eval_step(params_all, batch):
        if args.use_flax_sbert:
            outputs = sbert_flax(
                input_ids=batch['sbert_input_ids'],
                attention_mask=batch['sbert_attention_mask'],
                params=params_all['sbert'],
                train=False
            )
            hidden = outputs.last_hidden_state  # (bf16)
            mask = batch['sbert_attention_mask'].astype(jnp.float32)
            denom = jnp.clip(jnp.sum(mask, axis=1, keepdims=True), a_min=1.0)
            sbert_emb = (jnp.sum(hidden * mask[..., None], axis=1) / denom).astype(jnp.float32)
        else:
            sbert_emb = batch['sbert_embedding']
        out = model.apply({'params': params_all['snn']},
                          sbert_emb, batch['pos_tags'], batch['syntax_features'],
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
    accum_steps = max(1, int(os.environ.get('GRAD_ACCUM_STEPS', os.environ.get('grad_accum_steps', '1'))))
    for epoch in range(args.epochs):
        metrics_buf = []
        accum_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)
        k = 0
        for step, batch in enumerate(batches(train_processed, bs=args.batch_size, shuffle=True)):
            grads, metrics = compute_grads(model, sbert_flax, args.use_flax_sbert, state.params, batch,
                                           label_smoothing=args.label_smoothing, diversity_coef=args.diversity_coef)
            accum_grads = jax.tree_util.tree_map(lambda a,g: a+g, accum_grads, grads)
            k += 1
            if k == accum_steps:
                avg_grads = jax.tree_util.tree_map(lambda g: g / accum_steps, accum_grads)
                state = state.apply_gradients(grads=avg_grads)
                accum_grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), state.params)
                k = 0
            metrics_buf.append(metrics)
            if (step+1) % 10 == 0 and (args.process_id == 0):
                avg = jnp.mean(jnp.array([m['loss'] for m in metrics_buf[-10:]]))
                print(f"  epoch {epoch+1} step {step+1}: loss={float(avg):.4f}")
        # flush leftover grads
        if k > 0:
            avg_grads = jax.tree_util.tree_map(lambda g: g / k, accum_grads)
            state = state.apply_gradients(grads=avg_grads)
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
