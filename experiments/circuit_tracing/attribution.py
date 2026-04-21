"""
Attribution graph computation for SUBLEQ circuit tracing.

Two levels of analysis:

1. OV Circuit (weight-space)
   W_OV_h = W_V_h @ W_O_h  — the linear map from source token residual stream
   to destination token residual stream via head h.  Computed once from weights.

2. Attribution Graph (activation-space, per-example)
   Frozen-Jacobian attribution following the approach in:
   "Circuit Tracing: Revealing Computational Graphs in Language Models"
   (Anthropic, 2025).

   For attention: with frozen attention weights A_h, the Jacobian of head h's
   output at query position q with respect to input at key position k is
       J_{q←k}^h = A_h[q,k] · W_OV_h
   This is exact under the frozen-A assumption.

   For FFN: with frozen activation mask (set of active neurons), the Jacobian of
   the FFN output with respect to the FFN input is
       J_FFN = W2 @ diag(mask) @ W1_eff
   where W1_eff incorporates the LayerNorm Jacobian.  We compute the LayerNorm
   Jacobian analytically (closed form for standard LN).
"""

import torch
import torch.nn.functional as F
import numpy as np

from .extract_activations import get_head_ov_weights, get_all_activations


# ─── OV circuit (weight-space) ────────────────────────────────────────────────

def compute_ov_circuits(model):
    """
    Return W_OV for every head in every layer.

    Returns
    -------
    ov : dict  layer_idx → Tensor (H, d_model, d_model)
    """
    return {i: get_head_ov_weights(model, i) for i in range(len(model.layers))}


def ov_eigenspectrum(W_OV):
    """
    Singular value decomposition of each head's OV circuit.

    A large top singular value indicates a low-rank "copy" or "move" circuit.

    Parameters
    ----------
    W_OV : Tensor (H, d_model, d_model)

    Returns
    -------
    singular_values : Tensor (H, d_model)
    """
    results = []
    for h in range(W_OV.shape[0]):
        _, s, _ = torch.linalg.svd(W_OV[h].float())
        results.append(s)
    return torch.stack(results)   # (H, d_model)


# ─── LayerNorm Jacobian ───────────────────────────────────────────────────────

def layernorm_jacobian(x, ln_module):
    """
    Compute the Jacobian d(LN(x)) / d(x) analytically.

    For LayerNorm with learned scale γ and shift β:
        LN(x)_i = γ_i * x_hat_i + β_i
        x_hat = (x - μ) / σ

    The Jacobian (d_model × d_model) is:
        J_ij = γ_i / σ * (δ_ij - 1/D - x_hat_i * x_hat_j)

    Parameters
    ----------
    x         : Tensor (..., d_model)  input to LayerNorm
    ln_module : nn.LayerNorm

    Returns
    -------
    J : Tensor (..., d_model, d_model)  Jacobian at each position
    """
    D = x.shape[-1]
    eps = ln_module.eps if hasattr(ln_module, 'eps') else 1e-5
    gamma = ln_module.weight   # (d_model,)

    mu = x.mean(dim=-1, keepdim=True)
    sigma = ((x - mu).pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()
    x_hat = (x - mu) / sigma   # (..., D)

    # Correct LN Jacobian (derived via quotient rule on x_hat = (x-mu)/sigma):
    #   J_ij = (γ_i / σ) * (δ_ij - 1/D - x_hat_i * x_hat_j / D)
    #
    # The /D on the outer product comes from d(σ)/d(x_j) = x_hat_j / D.
    # sigma: (..., 1); gamma: (D,)
    # gamma / sigma broadcasts to (..., D), unsqueeze(-1) → (..., D, 1)
    g_over_sigma = (gamma / sigma).unsqueeze(-1)  # (..., D, 1)

    eye = torch.eye(D, device=x.device, dtype=x.dtype)
    ones_over_D = torch.ones(D, D, device=x.device, dtype=x.dtype) / D
    outer = x_hat.unsqueeze(-1) * x_hat.unsqueeze(-2) / D      # (..., D, D)

    J = g_over_sigma * (eye - ones_over_D - outer)             # (..., D, D)
    return J


# ─── Per-example attribution graph ───────────────────────────────────────────

def compute_attn_attribution(model, acts, layer_idx):
    """
    Attribution from each source position to each destination position via
    attention at layer_idx.

    Under frozen attention, the contribution of position k to the residual
    stream update at position q via head h is:
        contrib[h, q, k] = A_h[q,k] * W_OV_h   (d_model × d_model matrix)

    We summarise per (q, k) pair by taking the Frobenius norm of the combined
    OV contribution, weighted by attention.

    Parameters
    ----------
    model      : nn.Module
    acts       : dict  from get_all_activations()
    layer_idx  : int

    Returns
    -------
    attn_attr : Tensor (H, T_q, T_k)  attention-weighted OV norm per head/position
    W_OV      : Tensor (H, d_model, d_model)
    """
    W_OV = get_head_ov_weights(model, layer_idx)    # (H, D, D)
    attn_w = acts[f'attn_weights_{layer_idx}']      # (B, H, T, T)
    B = attn_w.shape[0]

    # OV Frobenius norm per head: scalar representing "how much" head h moves
    ov_norm = W_OV.norm(dim=(-2, -1))               # (H,)

    # Attribution[h, q, k] = A[h, q, k] * ov_norm[h]
    attn_attr = attn_w.mean(0) * ov_norm.view(-1, 1, 1)  # (H, T, T)
    return attn_attr, W_OV


def compute_ffn_attribution(model, acts, layer_idx):
    """
    Attribution from the residual stream (post-attention) to each FFN neuron,
    and from each FFN neuron to the residual stream (post-FFN).

    With frozen activation mask, the full FFN Jacobian is:
        J = W2 @ diag(mask) @ W1_eff

    where W1_eff = W1 @ J_LN  (incorporates LayerNorm Jacobian when present).

    Parameters
    ----------
    model      : nn.Module
    acts       : dict  from get_all_activations()
    layer_idx  : int

    Returns
    -------
    in_attr   : Tensor (B, T, d_ff)  attribution from residual → each neuron (W1 projection norm)
    out_attr  : Tensor (B, T, d_ff)  each neuron's contribution norm to residual (W2 column norm * activation)
    J_full    : Tensor (B, T, d_model, d_model)  full frozen Jacobian of FFN layer
    """
    layer = model.layers[layer_idx]
    ffn = layer.ffn
    is_constrained = hasattr(model, 'type_emb')

    W1 = ffn.w1.weight   # (d_ff, d_model)
    W2 = ffn.w2.weight   # (d_model, d_ff)
    b1 = ffn.w1.bias if ffn.w1.bias is not None else 0
    b2 = ffn.w2.bias if ffn.w2.bias is not None else 0

    pre = acts[f'ffn_pre_{layer_idx}']    # (B, T, d_ff)
    # Activation mask: neurons that are active (pre > 0 for ReLU)
    mask = (pre > 0).float()              # (B, T, d_ff)

    # W1_eff: if there's a LayerNorm before FFN, multiply by its Jacobian
    residual_in = acts[f'residual_post_attn_{layer_idx}']   # (B, T, d_model)

    has_ln = hasattr(layer, 'norm2') and not isinstance(layer.norm2,
                                                        type(None))
    try:
        # Check if norm2 is a real LayerNorm (not Identity)
        _ = layer.norm2.weight
        has_real_ln = True
    except AttributeError:
        has_real_ln = False

    if has_real_ln:
        # J_LN: (B, T, d_model, d_model)
        J_LN = layernorm_jacobian(residual_in, layer.norm2)
        # W1_eff[b,t,n,m] = sum_k W1[n,k] * J_LN[b,t,k,m]
        # → (d_ff, d_model) @ (B,T,d_model,d_model) → (B,T,d_ff,d_model)
        W1_eff = torch.einsum('nk,btkm->btnm', W1, J_LN)  # (B,T,d_ff,d_model)
    else:
        # No LN: W1_eff is just W1 broadcast over batch and time
        B, T, _ = residual_in.shape
        W1_eff = W1.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # (B,T,d_ff,d_model)

    # Attribution: residual → neuron n = L2 norm of W1_eff row n (how strongly
    # residual stream directions project onto neuron n's input weight)
    in_attr = W1_eff.norm(dim=-1)   # (B, T, d_ff)

    # Attribution: neuron n → residual = |activation_n| * ||W2[:, n]||
    W2_col_norms = W2.norm(dim=0)   # (d_ff,)
    post = acts[f'ffn_post_{layer_idx}']  # (B, T, d_ff)
    out_attr = post.abs() * W2_col_norms.unsqueeze(0).unsqueeze(0)  # (B,T,d_ff)

    # Full frozen Jacobian J_full[b,t,m,n] = d(ffn_out[m]) / d(residual_in[n])
    # = sum_k W2[m,k] * mask[b,t,k] * W1_eff[b,t,k,n]
    J_full = torch.einsum('mk,btk,btkn->btmn', W2, mask, W1_eff)  # (B,T,D,D)

    return in_attr, out_attr, J_full


def build_attribution_graph(model, inputs, include_jacobians=True):
    """
    Build the full attribution graph for a batch of examples.

    Returns a dict with:
      attn_attr[i]   : Tensor (H, T, T)   attention attribution per layer (mean over batch)
      ffn_in_attr[i] : Tensor (T, d_ff)   residual→neuron attribution (mean over batch)
      ffn_out_attr[i]: Tensor (T, d_ff)   neuron→residual attribution (mean over batch)
      ffn_jacobian[i]: Tensor (T, D, D)   full FFN frozen Jacobian (mean over batch)
                       — only present if include_jacobians=True
      ov_circuits[i] : Tensor (H, D, D)   W_OV weight matrices
      attn_weights[i]: Tensor (H, T, T)   mean attention weights
    """
    acts = get_all_activations(model, inputs)
    n_layers = len(model.layers)
    graph = {}

    W_OV_all = compute_ov_circuits(model)

    for i in range(n_layers):
        attn_attr, W_OV = compute_attn_attribution(model, acts, i)
        graph[f'attn_attr_{i}'] = attn_attr
        graph[f'ov_circuits_{i}'] = W_OV
        graph[f'attn_weights_{i}'] = acts[f'attn_weights_{i}'].mean(0)  # (H,T,T)

        in_a, out_a, J = compute_ffn_attribution(model, acts, i)
        graph[f'ffn_in_attr_{i}'] = in_a.mean(0)    # (T, d_ff)
        graph[f'ffn_out_attr_{i}'] = out_a.mean(0)  # (T, d_ff)
        if include_jacobians:
            graph[f'ffn_jacobian_{i}'] = J.mean(0)  # (T, D, D)

    return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in graph.items()}


# ─── End-to-end path attribution ─────────────────────────────────────────────

def token_to_output_attribution(model, inputs, output_pos, output_dim=None):
    """
    Compute the total attribution from each input token position to a given
    output position using the full frozen-Jacobian graph.

    This chains all layer Jacobians together:
        J_total = J_L @ J_{L-1} @ ... @ J_1 @ J_embed

    Parameters
    ----------
    model      : nn.Module
    inputs     : LongTensor (B, T)
    output_pos : int   which token position's output to explain
    output_dim : int or None  if None, use the argmax logit direction

    Returns
    -------
    token_attr : Tensor (B, T)  L2 norm of attribution from each input token
    """
    acts = get_all_activations(model, inputs)
    B, T = inputs.shape
    n_layers = len(model.layers)

    # Build per-layer Jacobians for the path through output_pos
    # We track d(output_pos residual) / d(each position's residual after each layer)
    # Simplified: track only through the residual stream at output_pos

    # Full Jacobian of the residual stream at output_pos w.r.t. the embedding
    # is complex due to attention mixing positions. Here we compute a simpler
    # upper bound: the attribution through the direct path.

    # Layer i: residual[i][output_pos] = residual[i-1][output_pos]
    #           + attn_out[output_pos] + ffn_out[output_pos]
    # attn_out[output_pos] = sum_{q,k,h} A_h[output_pos, k] * W_OV_h @ residual_in[k]

    # For a cleaner analysis we compute the attribution of each input token
    # to the pre-output residual using the frozen Jacobian.

    _, _, ffn_jacobians = zip(*(
        compute_ffn_attribution(model, acts, i) for i in range(n_layers)
    ))

    # Start from identity at the output position in the final residual stream
    # and propagate backwards through the frozen Jacobians.
    # We handle cross-position mixing via the attention weights.

    # This is a simplified version that accumulates attribution layer by layer.
    D = acts['embedding'].shape[-1]

    # grad_wrt_residual[i][b, pos, d] = how much output at output_pos depends
    # on residual stream position `pos` after layer i.
    # Initialize: output position has gradient 1 in output direction.
    is_constrained = hasattr(model, 'type_emb')

    if output_dim is None:
        logits = acts['logits']   # (B, T, V)
        output_dim = logits[:, output_pos, :].argmax(-1)  # (B,) best token per example

    # Work backwards from the output head
    if is_constrained:
        W_out = model.output_head.weight    # (V, D)
        # Gradient of logit[output_pos, output_dim] w.r.t. final residual
        # = W_out[output_dim, :]  (one vector per batch element)
        if isinstance(output_dim, torch.Tensor):
            g = W_out[output_dim, :]        # (B, D)
        else:
            g = W_out[output_dim].unsqueeze(0).expand(B, -1)   # (B, D)

        # LN Jacobian for final norm
        h_prenorm = acts['pre_output']   # (B, T, D)
        J_ln_out = layernorm_jacobian(h_prenorm[:, output_pos, :], model.final_norm)
        # (B, D, D)
        g = torch.einsum('bd,bde->be', g, J_ln_out)  # (B, D)
    else:
        # HandCodedSUBLEQ reads dim 0 of residual stream directly
        g = torch.zeros(B, D, device=inputs.device)
        g[:, 0] = 1.0   # DV = 0

    # g is now (B, D): gradient w.r.t. residual at output_pos after last layer

    # We propagate through layers in reverse, accumulating attribution to each
    # input token position via the attention paths.
    token_attr = torch.zeros(B, T, device=inputs.device)

    for i in range(n_layers - 1, -1, -1):
        J_ffn = ffn_jacobians[i][:, output_pos, :, :]   # (B, D, D)
        # Gradient through FFN: residual update Jacobian
        # residual_out = residual_in + ffn_out
        # d(loss)/d(residual_in) += d(loss)/d(residual_out) @ J_ffn
        # But residual connection means d(loss)/d(residual_in) includes both paths.
        g_through_ffn = torch.einsum('bd,bde->be', g, J_ffn)  # (B, D)
        g = g + g_through_ffn   # residual connection at FFN

        # Gradient through attention: attention mixes positions
        # head h: attn_out[output_pos] += sum_k A_h[output_pos,k] * W_OV_h @ res[k]
        attn_w = acts[f'attn_weights_{i}']   # (B, H, T, T)
        W_OV = get_head_ov_weights(model, i) # (H, D, D)

        for k in range(T):
            for h_idx in range(attn_w.shape[1]):
                a_hqk = attn_w[:, h_idx, output_pos, k]   # (B,)
                # d(loss)/d(res[k]) += a_hqk * g @ W_OV_h
                contrib = a_hqk.unsqueeze(-1) * torch.einsum(
                    'bd,de->be', g, W_OV[h_idx])            # (B, D)
                if k == output_pos:
                    # Residual connection: gradient also flows directly
                    pass
                # Accumulate into token attribution (as norm)
                token_attr[:, k] += (contrib.norm(dim=-1) * a_hqk)

        # Gradient through attention residual connection:
        # residual = residual_in + attn_out
        # d(loss)/d(residual_in) += d(loss)/d(residual_out) [direct path]
        # (already included via residual connection logic)

    # Normalise
    token_attr = token_attr / (token_attr.sum(-1, keepdim=True) + 1e-9)
    return token_attr
