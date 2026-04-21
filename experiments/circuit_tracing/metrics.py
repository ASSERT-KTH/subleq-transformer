"""
Circuit metrics for comparing trained vs. hardcoded SUBLEQ circuits.

All score functions return values in [0, 1] where 1 = perfect match to the
expected circuit behaviour, and a random baseline is approximately 1/T
(uniform attention over T positions).

Reference-based interpretation: compute the same metrics on the hardcoded
model to get a 100%-reference, then express trained-model scores as a
fraction of that reference.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ─── Attention-pattern metrics ────────────────────────────────────────────────

def fetch_head_score(attn_weights, target_positions, query_pos=0):
    """
    Measure how well each head implements content-addressed fetching.

    A "fetch head" for operand X should, at query_pos (the PC token, position 0),
    attend to the token at position X+1 (where X is the relevant operand address).

    Parameters
    ----------
    attn_weights     : Tensor (N, H, T, T)  softmax attention weights
    target_positions : Tensor (N,)   int, expected target token position per example
    query_pos        : int           query position (0 = PC token)

    Returns
    -------
    scores : ndarray (H,)  mean attention weight at target position, in [0, 1]
    random_baseline : float  1/T, expected score for a uniform head
    """
    attn_weights = attn_weights.float()
    N, H, T, _ = attn_weights.shape
    target_positions = target_positions.long()

    # attn_weights[:, :, query_pos, :]  → (N, H, T_keys)
    attn_at_query = attn_weights[:, :, query_pos, :]   # (N, H, T)

    # For each example n, gather the weight at the target position
    # target_positions: (N,) → expand to (N, H, 1)
    tgt = target_positions.view(N, 1, 1).expand(N, H, 1)
    weight_at_target = attn_at_query.gather(dim=2, index=tgt).squeeze(2)  # (N, H)

    scores = weight_at_target.mean(dim=0).cpu().numpy()   # (H,)
    random_baseline = 1.0 / T
    return scores, random_baseline


def broadcast_head_score(attn_weights, src_pos=0, exclude_self=True):
    """
    Measure how well each head broadcasts from src_pos to all query positions.

    A "broadcast head" makes every query position attend to src_pos (position 0).

    Parameters
    ----------
    attn_weights : Tensor (N, H, T, T)
    src_pos      : int   the position being broadcast from (default 0)
    exclude_self : bool  if True, exclude query_pos == src_pos from averaging

    Returns
    -------
    scores          : ndarray (H,)  mean attention weight to src_pos, in [0, 1]
    random_baseline : float  1/T
    """
    attn_weights = attn_weights.float()
    N, H, T, _ = attn_weights.shape

    # attn_weights[:, :, :, src_pos] → (N, H, T_queries)
    weight_to_src = attn_weights[:, :, :, src_pos]   # (N, H, T)

    if exclude_self:
        mask = torch.ones(T, dtype=torch.bool)
        mask[src_pos] = False
        weight_to_src = weight_to_src[:, :, mask]    # (N, H, T-1)

    scores = weight_to_src.mean(dim=(0, 2)).cpu().numpy()   # (H,)
    random_baseline = 1.0 / T
    return scores, random_baseline


def attention_entropy(attn_weights):
    """
    Entropy of each head's attention distribution (averaged over queries and examples).

    Low entropy → sharp / concentrated attention (fetch-like).
    High entropy → diffuse attention.

    Returns
    -------
    entropy : ndarray (H,)  in nats
    """
    attn_weights = attn_weights.float()
    # attn_weights: (N, H, T, T); entropy over key dimension
    eps = 1e-9
    ent = -(attn_weights * (attn_weights + eps).log()).sum(dim=-1)  # (N, H, T)
    return ent.mean(dim=(0, 2)).cpu().numpy()   # (H,)


# ─── FFN neuron metrics ────────────────────────────────────────────────────────

def neuron_concept_correlation(ffn_pre, concept_vals, token_pos=0):
    """
    Pearson correlation between each FFN neuron's pre-activation and a scalar concept.

    Parameters
    ----------
    ffn_pre      : Tensor (N, T, d_ff)  pre-ReLU/GELU FFN activations
    concept_vals : Tensor (N,)          one scalar per example (e.g. mem[b]-mem[a])
    token_pos    : int                  which sequence position to probe (default 0 = PC)

    Returns
    -------
    corr : ndarray (d_ff,)  Pearson r for each neuron, in [-1, 1]
    """
    acts = ffn_pre[:, token_pos, :].float().cpu()   # (N, d_ff)
    vals = concept_vals.float().cpu()               # (N,)

    acts_z = acts - acts.mean(0, keepdim=True)
    vals_z = vals - vals.mean()

    # Use biased std throughout so that corr = 1.0 for perfectly correlated inputs
    cov = (acts_z * vals_z.unsqueeze(1)).mean(0)              # (d_ff,)
    std_acts = acts_z.pow(2).mean(0).sqrt().clamp(min=1e-8)   # biased std
    std_vals = float(vals_z.pow(2).mean().sqrt().clamp(min=1e-8))
    corr = (cov / (std_acts * std_vals)).numpy()
    return corr


def top_neuron_concepts(ffn_pre, concept_dict, token_pos=0, top_k=5):
    """
    For each concept in concept_dict, find the top_k neurons most correlated with it.

    Parameters
    ----------
    ffn_pre      : Tensor (N, T, d_ff)
    concept_dict : dict  name → Tensor (N,)
    token_pos    : int

    Returns
    -------
    results : dict  name → {'corr': ndarray(d_ff), 'top_neurons': list[int],
                              'top_corr': list[float]}
    """
    results = {}
    for name, vals in concept_dict.items():
        corr = neuron_concept_correlation(ffn_pre, vals, token_pos=token_pos)
        abs_corr = np.abs(corr)
        top_idx = np.argsort(abs_corr)[::-1][:top_k].tolist()
        results[name] = {
            'corr': corr,
            'top_neurons': top_idx,
            'top_corr': corr[top_idx].tolist(),
        }
    return results


# ─── Circuit completeness ─────────────────────────────────────────────────────

def mean_head_activations(model, dataloader, device='cpu'):
    """
    Compute the mean attention head output (summed over keys) for each head
    across the dataset. Used as the ablation baseline.

    Returns
    -------
    mean_head_out : dict  layer_idx → Tensor (H, T, d_model)
    """
    from .extract_activations import get_all_activations

    n_layers = len(model.layers)
    sums = {}
    counts = {}

    model.eval()
    total = 0

    with torch.no_grad():
        for inputs in dataloader:
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            inputs = inputs.to(device)
            acts = get_all_activations(model, inputs)
            B = inputs.shape[0]
            total += B

            for i in range(n_layers):
                ho = acts[f'head_out_{i}']   # (B, H, T, d_model)
                if i not in sums:
                    sums[i] = ho.sum(0)       # (H, T, d_model)
                    counts[i] = B
                else:
                    sums[i] += ho.sum(0)
                    counts[i] += B

    return {i: sums[i] / counts[i] for i in sums}


@torch.no_grad()
def circuit_completeness(model, inputs, targets, circuit_heads,
                         mean_head_out, changed_positions=None, device='cpu'):
    """
    Ablate all attention heads NOT in circuit_heads to their mean activation,
    then measure token-level accuracy on the changed positions.

    Parameters
    ----------
    model          : nn.Module
    inputs         : LongTensor (B, T)
    targets        : LongTensor (B, T)
    circuit_heads  : dict  layer_idx → set of head indices to KEEP
    mean_head_out  : dict  layer_idx → Tensor (H, T, d_model)  ablation values
    changed_positions : Tensor (B, T) bool mask of positions to evaluate
                        (if None, evaluate all positions)
    device         : str

    Returns
    -------
    dict with keys 'full_acc', 'circuit_acc', 'ratio'
    """
    from .extract_activations import get_all_activations

    inputs = inputs.to(device)
    targets = targets.to(device)
    B, T = inputs.shape
    n_layers = len(model.layers)

    # ── Full model accuracy ────────────────────────────────────────────────
    acts_full = get_all_activations(model, inputs)
    preds_full = acts_full['logits'].argmax(-1)   # (B, T)

    # ── Ablated forward pass ───────────────────────────────────────────────
    # We re-run the forward pass, replacing ablated heads with their means.
    is_constrained = hasattr(model, 'type_emb')

    with torch.no_grad():
        if is_constrained:
            tok = model.token_emb(inputs)
            pos = model.pos_emb(model.pos_indices[:, :T].expand(B, -1))
            typ = model.type_emb(model.type_indices[:, :T].expand(B, -1))
            h = tok + pos + typ
        else:
            from round1_constructed.model import DV
            dev = inputs.device
            h = model.tok_emb(inputs) + model.pos_emb(torch.arange(T, device=dev))

        for i, layer in enumerate(model.layers):
            from .extract_activations import (
                _split_head_outputs_constrained,
                _split_head_outputs_hardcoded,
            )
            import torch.nn.functional as F

            normed = layer.norm1(h) if hasattr(layer, 'norm1') else h

            if is_constrained:
                attn_w, head_out = _split_head_outputs_constrained(layer, normed)
            else:
                attn_w, head_out = _split_head_outputs_hardcoded(layer, h)

            # Ablate heads not in circuit
            keep = circuit_heads.get(i, set())
            mho = mean_head_out[i].to(device)   # (H, T, d_model)

            for head_idx in range(head_out.shape[1]):
                if head_idx not in keep:
                    # replace with dataset mean (broadcast over batch)
                    head_out[:, head_idx, :, :] = mho[head_idx].unsqueeze(0)

            full_attn = head_out.sum(dim=1)
            if is_constrained and layer.attn.out_proj.bias is not None:
                full_attn = full_attn + layer.attn.out_proj.bias

            h = h + full_attn

            normed2 = layer.norm2(h) if hasattr(layer, 'norm2') else h
            ffn = layer.ffn
            from .extract_activations import _capture_ffn_post
            post = _capture_ffn_post(ffn, normed2)
            h = h + ffn.w2(post)

        if is_constrained:
            h_norm = model.final_norm(h)
            logits_ablated = model.output_head(h_norm)
        else:
            from round1_constructed.model import VALUE_MIN, VALUE_MAX, VOCAB_SIZE
            values = h[:, :, DV]
            output_tokens = values.round().clamp(VALUE_MIN, VALUE_MAX).long() + 32768
            logits_ablated = torch.full((B, T, VOCAB_SIZE), -100.0, device=device)
            logits_ablated.scatter_(2, output_tokens.unsqueeze(2), 100.0)

    preds_ablated = logits_ablated.argmax(-1)   # (B, T)

    def _accuracy(preds):
        correct = (preds == targets)
        if changed_positions is not None:
            mask = changed_positions.to(device)
            return (correct & mask).sum().item() / mask.sum().item()
        return correct.float().mean().item()

    full_acc = _accuracy(preds_full)
    circ_acc = _accuracy(preds_ablated)
    return {
        'full_acc': full_acc,
        'circuit_acc': circ_acc,
        'ratio': circ_acc / max(full_acc, 1e-9),
    }


# ─── Summary helpers ──────────────────────────────────────────────────────────

def summarise_head_roles(fetch_scores, broadcast_scores, random_baseline,
                         fetch_threshold=5.0, broadcast_threshold=5.0):
    """
    Label each head as 'fetch', 'broadcast', 'mixed', or 'other' based on how
    many times above the random baseline its score is.

    Parameters
    ----------
    fetch_scores     : ndarray (L, H)
    broadcast_scores : ndarray (L, H)
    random_baseline  : float   1/T

    Returns
    -------
    roles : list[list[str]]  roles[layer][head]
    """
    L, H = fetch_scores.shape
    roles = []
    for l in range(L):
        layer_roles = []
        for h in range(H):
            is_fetch = fetch_scores[l, h] > fetch_threshold * random_baseline
            is_bcast = broadcast_scores[l, h] > broadcast_threshold * random_baseline
            if is_fetch and is_bcast:
                layer_roles.append('mixed')
            elif is_fetch:
                layer_roles.append('fetch')
            elif is_bcast:
                layer_roles.append('broadcast')
            else:
                layer_roles.append('other')
        roles.append(layer_roles)
    return roles
