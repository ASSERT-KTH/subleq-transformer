#!/usr/bin/env python3
"""
Train constrained SUBLEQ transformer (oracle-footprint architecture).

d_model=32, n_layers=4, n_heads=8, d_ff=64, ReLU.
Two variants: with and without LayerNorm.

Usage:
    python train_constrained.py --variant ln   # with LayerNorm
    python train_constrained.py --variant no_ln # without LayerNorm
    python train_constrained.py --variant ln --seed 1
"""

import os
import sys
import time
import math
import random
import argparse

import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))
sys.path.insert(0, script_dir)

from subleq import pregenerate_data
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE
from subleq.data import CHANGE_WEIGHT
from constrained_model import ConstrainedSUBLEQTransformer


def weighted_cross_entropy(logits, targets, weight_mask):
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S),
                           reduction='none')
    loss = loss.reshape(B, S)
    loss = (loss * weight_mask).sum() / weight_mask.sum()
    return loss


def compute_accuracy(logits, targets, weight_mask=None):
    preds = logits.argmax(dim=-1)
    all_correct = (preds == targets).all(dim=1).float().mean().item()
    if weight_mask is not None:
        changed = weight_mask > 1.0
        if changed.any():
            changed_correct = ((preds == targets) | ~changed).all(dim=1).float().mean().item()
        else:
            changed_correct = all_correct
    else:
        changed_correct = all_correct
    return all_correct, changed_correct


def get_curriculum_range(step, total_steps):
    frac = step / total_steps
    if frac < 0.10:
        return (1, 2)
    elif frac < 0.25:
        return (1, 4)
    elif frac < 0.45:
        return (1, 6)
    else:
        return (1, 8)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Variant: {args.variant} (layer_norm={'yes' if args.variant == 'ln' else 'no'})")
    print(f"SEQ_LEN: {SEQ_LEN}, VOCAB_SIZE: {VOCAB_SIZE}")

    layer_norm = (args.variant == 'ln')

    model = ConstrainedSUBLEQTransformer(
        d_model=32,
        n_heads=8,
        n_layers=4,
        d_ff=64,
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        dropout=args.dropout,
        layer_norm=layer_norm,
    ).to(device)
    print(f"Parameters: {model.count_params():,}")

    save_dir = os.path.join(args.ckpt_base, f'constrained_{args.variant}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    instr_range = get_curriculum_range(0, args.total_steps)
    print(f"Generating {args.data_size} training examples (instr_range={instr_range})...")
    data_inp, data_out, data_mask = pregenerate_data(args.data_size, instr_range=instr_range)
    data_inp = data_inp.to(device)
    data_out = data_out.to(device)
    data_mask = data_mask.to(device)
    print(f"Data ready: {data_inp.shape}")

    best_acc = 0.0
    log_loss = log_full_acc = log_changed_acc = log_count = 0.0
    start_time = time.time()

    for step_num in range(1, args.total_steps + 1):
        model.train()

        idx = torch.randint(0, data_inp.size(0), (args.batch_size,))
        inp = data_inp[idx]
        out = data_out[idx]
        mask = data_mask[idx]

        logits = model(inp)
        loss = weighted_cross_entropy(logits, out, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            full_acc, changed_acc = compute_accuracy(logits, out, mask)
        log_loss += loss.item()
        log_full_acc += full_acc
        log_changed_acc += changed_acc
        log_count += 1

        if step_num % args.log_every == 0:
            avg_loss = log_loss / log_count
            avg_full = log_full_acc / log_count
            avg_changed = log_changed_acc / log_count
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Step {step_num:6d}/{args.total_steps} | "
                  f"loss={avg_loss:.4f} | "
                  f"full_acc={avg_full:.3f} | "
                  f"changed_acc={avg_changed:.3f} | "
                  f"lr={lr:.2e} | "
                  f"instr={get_curriculum_range(step_num, args.total_steps)} | "
                  f"time={elapsed:.0f}s")
            log_loss = log_full_acc = log_changed_acc = log_count = 0.0

        if step_num % args.regen_every == 0:
            instr_range = get_curriculum_range(step_num, args.total_steps)
            print(f"  Regenerating data (instr_range={instr_range})...")
            data_inp, data_out, data_mask = pregenerate_data(
                args.data_size, instr_range=instr_range)
            data_inp = data_inp.to(device)
            data_out = data_out.to(device)
            data_mask = data_mask.to(device)

        if step_num % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                eval_inp, eval_out, eval_mask = pregenerate_data(2000, instr_range=(1, 8))
                eval_inp = eval_inp.to(device)
                eval_out = eval_out.to(device)
                eval_mask = eval_mask.to(device)
                eval_logits = model(eval_inp)
                eval_full, eval_changed = compute_accuracy(eval_logits, eval_out, eval_mask)
                eval_loss = weighted_cross_entropy(eval_logits, eval_out, eval_mask).item()

            print(f"  EVAL step {step_num}: loss={eval_loss:.4f} | "
                  f"full_acc={eval_full:.3f} | changed_acc={eval_changed:.3f}")

            if eval_full > best_acc:
                best_acc = eval_full
                ckpt_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'step': step_num,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': {
                        'd_model': 32,
                        'n_heads': 8,
                        'n_layers': 4,
                        'd_ff': 64,
                        'layer_norm': layer_norm,
                        'variant': args.variant,
                        'seed': args.seed,
                    }
                }, ckpt_path)
                print(f"  New best: {best_acc:.3f} (saved)")

    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save({
        'step': args.total_steps,
        'model_state': model.state_dict(),
        'best_acc': best_acc,
        'config': {
            'd_model': 32,
            'n_heads': 8,
            'n_layers': 4,
            'd_ff': 64,
            'layer_norm': layer_norm,
            'variant': args.variant,
            'seed': args.seed,
        }
    }, final_path)
    print(f"\nTraining complete. Best accuracy: {best_acc:.3f}")
    print(f"Saved to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train constrained SUBLEQ transformer")
    parser.add_argument('--variant', type=str, choices=['ln', 'no_ln'], required=True,
                        help='ln = with LayerNorm, no_ln = without LayerNorm')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Default 0.0 (small model, less regularization needed)')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--total-steps', type=int, default=80000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-size', type=int, default=100000)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--log-every', type=int, default=200)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--regen-every', type=int, default=2000)
    parser.add_argument('--ckpt-base', type=str,
                        default=os.path.join(script_dir, 'checkpoints'))
    args = parser.parse_args()
    train(args)
