"""
Constrained SUBLEQ transformer matching the oracle's architectural footprint.

d_model=32, n_layers=4, n_heads=8, d_ff=64, ReLU activation.
Two variants:
  - with LayerNorm  (layer_norm=True)
  - without LayerNorm (layer_norm=False)

Trained on the round2 task (32 cells, 8-bit, SEQ_LEN=33).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))

from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE


class Identity(nn.Module):
    """Drop-in replacement for LayerNorm when layer_norm=False."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ConstrainedFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class ConstrainedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class ConstrainedBlock(nn.Module):
    """Pre-LN (or no-LN) transformer block with ReLU FFN."""

    def __init__(self, d_model, n_heads, d_ff, dropout, layer_norm=True):
        super().__init__()
        LN = nn.LayerNorm if layer_norm else Identity
        self.norm1 = LN(d_model)
        self.attn = ConstrainedAttention(d_model, n_heads, dropout)
        self.norm2 = LN(d_model)
        self.ffn = ConstrainedFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ConstrainedSUBLEQTransformer(nn.Module):
    """
    Oracle-footprint transformer: d_model=32, 4 layers, 8 heads, d_ff=64, ReLU.
    Trained on round2 task (SEQ_LEN=33, VOCAB_SIZE=256).
    """

    def __init__(self, d_model=32, n_heads=8, n_layers=4, d_ff=64,
                 vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
                 dropout=0.1, layer_norm=True):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.uses_layer_norm = layer_norm

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.type_emb = nn.Embedding(2, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ConstrainedBlock(d_model, n_heads, d_ff, dropout, layer_norm=layer_norm)
            for _ in range(n_layers)
        ])

        LN = nn.LayerNorm if layer_norm else Identity
        self.final_norm = LN(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

        self.register_buffer('pos_indices', torch.arange(seq_len).unsqueeze(0))
        type_ids = torch.zeros(seq_len, dtype=torch.long)
        type_ids[0] = 0
        type_ids[1:] = 1
        self.register_buffer('type_indices', type_ids.unsqueeze(0))

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        B, S = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(self.pos_indices[:, :S].expand(B, -1))
        typ = self.type_emb(self.type_indices[:, :S].expand(B, -1))
        h = self.emb_dropout(tok + pos + typ)
        for layer in self.layers:
            h = layer(h)
        return self.output_head(self.final_norm(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def load_constrained_model(ckpt_path, device='cpu'):
    """Load a constrained model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    model = ConstrainedSUBLEQTransformer(
        d_model=config.get('d_model', 32),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 4),
        d_ff=config.get('d_ff', 64),
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=0.0,
        layer_norm=config.get('layer_norm', True),
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, config
