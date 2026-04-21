"""
Activation extraction for circuit tracing.

Supports two model families:
  - ConstrainedSUBLEQTransformer  (constrained_ln / constrained_no_ln)
  - HandCodedSUBLEQ               (round1 analytic model)

Returned dict keys for N layers:
  embedding               (B, T, d_model)   post-embedding, pre-layer-0
  residual_post_attn_{i}  (B, T, d_model)   after attention residual, layer i
  residual_{i}            (B, T, d_model)   after FFN residual, layer i
  attn_weights_{i}        (B, H, T, T)      softmax attention weights, layer i
  head_out_{i}            (B, H, T, d_model) per-head OV contribution to residual
  ffn_pre_{i}             (B, T, d_ff)      pre-activation FFN values, layer i
  ffn_post_{i}            (B, T, d_ff)      post-activation FFN values, layer i
  logits                  (B, T, vocab)
"""

import torch
import torch.nn.functional as F


# ─── helpers ──────────────────────────────────────────────────────────────────

def _split_head_outputs_constrained(layer, normed_x):
    """
    Extract per-head attention weights and OV contributions for a
    ConstrainedBlock (uses a single fused qkv Linear).

    Returns
    -------
    attn_weights : (B, H, T, T)
    head_out     : (B, H, T, d_model)   each head's OV contribution to residual
    """
    B, T, D = normed_x.shape
    H = layer.attn.n_heads
    d = layer.attn.d_head

    qkv = layer.attn.qkv(normed_x).reshape(B, T, 3, H, d).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]          # each (B, H, T, d)

    scores = (q @ k.transpose(-2, -1)) / layer.attn.scale
    attn = F.softmax(scores, dim=-1)            # (B, H, T, T)

    v_out = attn @ v                            # (B, H, T, d)

    # out_proj weight shape: (D, H*d) — split column-wise per head
    W_O = layer.attn.out_proj.weight            # (D, H*d)
    W_O_heads = W_O.view(D, H, d)              # (D, H, d)

    # head_out[b,h,t,m] = sum_d W_O_heads[m,h,d] * v_out[b,h,t,d]
    head_out = torch.einsum('mhd,bhtd->bhtm', W_O_heads, v_out)   # (B, H, T, D)

    # out_proj bias is shared; caller may add model.layers[i].attn.out_proj.bias
    return attn, head_out


def _split_head_outputs_hardcoded(layer, x):
    """
    Extract per-head attention weights and OV contributions for a
    TransformerBlock from HandCodedSUBLEQ (separate W_q, W_k, W_v, W_o).

    Returns
    -------
    attn_weights : (B, H, T, T)
    head_out     : (B, H, T, d_model)
    """
    B, T, D = x.shape
    H = layer.attn.n_heads
    d = layer.attn.d_head

    # Separate projection matrices
    Q = layer.attn.W_q(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
    K = layer.attn.W_k(x).view(B, T, H, d).transpose(1, 2)
    V = layer.attn.W_v(x).view(B, T, H, d).transpose(1, 2)

    scores = Q @ K.transpose(-2, -1) / (d ** 0.5)
    attn = F.softmax(scores, dim=-1)   # (B, H, T, T)

    v_out = attn @ V                   # (B, H, T, d)

    # W_o weight shape: (D, H*d) — same slicing as above
    W_O = layer.attn.W_o.weight        # (D, H*d)
    W_O_heads = W_O.view(D, H, d)     # (D, H, d)

    head_out = torch.einsum('mhd,bhtd->bhtm', W_O_heads, v_out)   # (B, H, T, D)
    return attn, head_out


# ─── public API ───────────────────────────────────────────────────────────────

def get_all_activations(model, inputs):
    """
    Extract all intermediate activations from a transformer forward pass.

    Automatically detects model type (constrained vs. hardcoded) from the
    presence of a ``norm1`` attribute on the first layer.

    Parameters
    ----------
    model  : nn.Module   ConstrainedSUBLEQTransformer or HandCodedSUBLEQ
    inputs : LongTensor  (B, T) token indices

    Returns
    -------
    dict with keys described in module docstring
    """
    model.eval()
    B, T = inputs.shape
    acts = {}

    with torch.no_grad():
        is_constrained = hasattr(model, 'type_emb')

        if is_constrained:
            tok = model.token_emb(inputs)
            pos = model.pos_emb(model.pos_indices[:, :T].expand(B, -1))
            typ = model.type_emb(model.type_indices[:, :T].expand(B, -1))
            h = tok + pos + typ
        else:
            # HandCodedSUBLEQ: token + position embedding only
            dev = inputs.device
            h = model.tok_emb(inputs) + model.pos_emb(torch.arange(T, device=dev))

        acts['embedding'] = h.clone()

        for i, layer in enumerate(model.layers):
            # ── Attention ──────────────────────────────────────────────────
            normed = layer.norm1(h) if hasattr(layer, 'norm1') else h

            if is_constrained:
                attn_w, head_out = _split_head_outputs_constrained(layer, normed)
            else:
                attn_w, head_out = _split_head_outputs_hardcoded(layer, h)

            acts[f'attn_weights_{i}'] = attn_w      # (B, H, T, T)
            acts[f'head_out_{i}'] = head_out         # (B, H, T, D)

            full_attn = head_out.sum(dim=1)          # sum over heads → (B, T, D)
            # add out_proj bias once (constrained model has bias; hardcoded does not)
            if is_constrained and layer.attn.out_proj.bias is not None:
                full_attn = full_attn + layer.attn.out_proj.bias

            h = h + full_attn
            acts[f'residual_post_attn_{i}'] = h.clone()

            # ── FFN ────────────────────────────────────────────────────────
            normed2 = layer.norm2(h) if hasattr(layer, 'norm2') else h

            ffn = layer.ffn
            pre = ffn.w1(normed2)
            acts[f'ffn_pre_{i}'] = pre.clone()       # (B, T, d_ff)

            post = _capture_ffn_post(ffn, normed2)
            acts[f'ffn_post_{i}'] = post.clone()     # (B, T, d_ff)

            ffn_out = ffn.w2(post)
            h = h + ffn_out
            acts[f'residual_{i}'] = h.clone()

        acts['pre_output'] = h.clone()

        if is_constrained:
            h_norm = model.final_norm(h)
            logits = model.output_head(h_norm)
        else:
            # HandCodedSUBLEQ reads DV (dim 0) directly; return raw residual as logits proxy
            from round1_constructed.model import DV, VALUE_MIN, VALUE_MAX, VOCAB_SIZE
            values = h[:, :, DV]
            output_tokens = values.round().clamp(VALUE_MIN, VALUE_MAX).long() + 32768
            logits = torch.full((B, T, VOCAB_SIZE), -100.0, device=inputs.device)
            logits.scatter_(2, output_tokens.unsqueeze(2), 100.0)

        acts['logits'] = logits

    return acts


def _capture_ffn_post(ffn, normed_input):
    """
    Run the FFN and capture post-activation values (the input to w2) via a hook.

    This approach works regardless of which activation function the FFN uses
    (ReLU, GELU, etc.) by hooking into w2's forward pre-hook.
    """
    captured = {}

    def _pre_hook(module, inp):
        captured['post'] = inp[0].detach()

    handle = ffn.w2.register_forward_pre_hook(_pre_hook)
    try:
        with torch.no_grad():
            ffn(normed_input)
    finally:
        handle.remove()

    return captured['post']


def get_head_ov_weights(model, layer_idx):
    """
    Return the OV circuit weight matrix W_OV for each head at a given layer.

    W_OV_h = W_V_h @ W_O_h   shape (d_model, d_model)

    This is the linear map from a source token's residual stream to the
    destination token's residual stream via head h.

    Returns
    -------
    W_OV : Tensor  (H, d_model, d_model)
    """
    layer = model.layers[layer_idx]
    is_constrained = hasattr(model, 'type_emb')

    if is_constrained:
        attn = layer.attn
        D = attn.out_proj.weight.shape[0]
        H = attn.n_heads
        d = attn.d_head

        # qkv weight: (3*D, D); rows 2*D: = V weights
        W_V_all = attn.qkv.weight[2 * D:, :]       # (D, D) = (H*d, D)
        W_V_all = W_V_all.view(H, d, D)             # (H, d, D)  — W_V for each head

        W_O_all = attn.out_proj.weight.view(D, H, d)  # (D, H, d)
        W_O_heads = W_O_all.permute(1, 2, 0)           # (H, d, D)

        # W_OV_h = W_V_h^T @ W_O_h^T  →  maps source (D) → dest (D)
        # x @ W_V_h → d_head; then @ W_O_h → D
        # W_OV_h[i,j] = sum_k W_V_h[k,i] * W_O_h[k,j]
        W_OV = torch.einsum('hki,hkj->hij', W_V_all, W_O_heads)  # (H, D, D)
    else:
        attn = layer.attn
        D = attn.W_o.weight.shape[0]
        H = attn.n_heads
        d = attn.d_head

        W_V_all = attn.W_v.weight.view(H, d, D)     # (H, d, D)
        W_O_all = attn.W_o.weight.view(D, H, d)
        W_O_heads = W_O_all.permute(1, 2, 0)         # (H, d, D)

        W_OV = torch.einsum('hki,hkj->hij', W_V_all, W_O_heads)  # (H, D, D)

    return W_OV
