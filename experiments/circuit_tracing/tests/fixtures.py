"""
Shared test fixtures: minimal mock transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MockFFN(nn.Module):
    def __init__(self, d_model=8, d_ff=16):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.relu(self.w1(x)))


class _MockAttn(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** 0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head
        qkv = self.qkv(x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class _MockBlock(nn.Module):
    def __init__(self, d_model=8, n_heads=2, d_ff=16):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MockAttn(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _MockFFN(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MockConstrainedModel(nn.Module):
    """Minimal model matching the ConstrainedSUBLEQTransformer interface."""

    def __init__(self, d_model=8, n_heads=2, n_layers=2, d_ff=16,
                 vocab_size=16, seq_len=5):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.type_emb = nn.Embedding(2, d_model)   # presence marks as 'constrained'
        self.layers = nn.ModuleList([
            _MockBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self.register_buffer('pos_indices',
                             torch.arange(seq_len).unsqueeze(0))
        type_ids = torch.zeros(seq_len, dtype=torch.long)
        type_ids[0] = 0
        type_ids[1:] = 1
        self.register_buffer('type_indices', type_ids.unsqueeze(0))

    def forward(self, x):
        B, S = x.shape
        h = (self.token_emb(x)
             + self.pos_emb(self.pos_indices[:, :S].expand(B, -1))
             + self.type_emb(self.type_indices[:, :S].expand(B, -1)))
        for layer in self.layers:
            h = layer(h)
        return self.output_head(self.final_norm(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
