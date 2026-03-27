#!/usr/bin/env python3
"""
Train 5 seeds of the round2 SUBLEQ transformer.

Usage:
    python train_seeds.py --seed 0 --output-dir ../experiments/checkpoints/
    python train_seeds.py --seed 1 --output-dir ../experiments/checkpoints/
    ...

The seed-0 model is already in round2_trained/checkpoints/best_model.pt.
Seeds 1-4 need to be trained.
"""

import os
import sys
import time
import math
import random
import argparse

# Add round2_trained to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))

import torch
import torch.nn.functional as F

from subleq import MiniSUBLEQTransformer, pregenerate_data
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE
from subleq.data import CHANGE_WEIGHT


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Seed: {args.seed}, Device: {device}")

    model = MiniSUBLEQTransformer(
        d_model=256, n_heads=8, n_layers=6, d_ff=1024,
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=0.1,
    ).to(device)
    print(f"Parameters: {model.count_params():,}")

    total_steps = args.total_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    def lr_lambda(step):
        warmup = 1000
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Checkpoints at 10%, 25%, 50%, 75%, 100%
    checkpoint_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    checkpoint_steps = [int(f * total_steps) for f in checkpoint_fracs]
    checkpointed = set()

    os.makedirs(args.output_dir, exist_ok=True)
    out_prefix = os.path.join(args.output_dir, f"seed{args.seed}")

    print(f"Generating initial training data...")
    instr_range = get_curriculum_range(0, total_steps)
    data_inp, data_out, data_mask = pregenerate_data(args.data_size, instr_range=instr_range)
    data_inp = data_inp.to(device)
    data_out = data_out.to(device)
    data_mask = data_mask.to(device)

    best_acc = 0.0
    start_time = time.time()
    log_loss = log_full = log_changed = log_n = 0.0

    for step_num in range(1, total_steps + 1):
        model.train()
        idx = torch.randint(0, data_inp.size(0), (args.batch_size,))
        inp = data_inp[idx]
        out = data_out[idx]
        mask = data_mask[idx]

        logits = model(inp)
        loss = weighted_cross_entropy(logits, out, mask)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            full_acc, changed_acc = compute_accuracy(logits, out, mask)
        log_loss += loss.item(); log_full += full_acc; log_changed += changed_acc; log_n += 1

        if step_num % 1000 == 0:
            avg_loss = log_loss / log_n
            avg_full = log_full / log_n
            avg_changed = log_changed / log_n
            elapsed = time.time() - start_time
            print(f"Step {step_num:6d}/{total_steps} | loss={avg_loss:.4f} | "
                  f"full={avg_full:.3f} | changed={avg_changed:.3f} | "
                  f"instr={get_curriculum_range(step_num, total_steps)} | "
                  f"t={elapsed:.0f}s")
            log_loss = log_full = log_changed = log_n = 0.0

        if step_num % 5000 == 0:
            instr_range = get_curriculum_range(step_num, total_steps)
            data_inp, data_out, data_mask = pregenerate_data(args.data_size, instr_range=instr_range)
            data_inp = data_inp.to(device)
            data_out = data_out.to(device)
            data_mask = data_mask.to(device)

        # Save checkpoints at milestone fractions
        for frac, ckpt_step in zip(checkpoint_fracs, checkpoint_steps):
            if step_num == ckpt_step and ckpt_step not in checkpointed:
                checkpointed.add(ckpt_step)
                ckpt_path = f"{out_prefix}_frac{int(frac*100):03d}.pt"
                torch.save({
                    'model_state': model.state_dict(),
                    'step': step_num,
                    'config': {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
                    'seed': args.seed,
                    'frac': frac,
                    'acc': full_acc,
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path} (acc={full_acc:.4f})")

    # Save final model
    final_path = f"{out_prefix}_final.pt"
    torch.save({
        'model_state': model.state_dict(),
        'step': total_steps,
        'config': {'d_model': 256, 'n_heads': 8, 'n_layers': 6, 'd_ff': 1024},
        'seed': args.seed,
        'frac': 1.0,
        'best_acc': best_acc,
    }, final_path)
    print(f"Final model saved: {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output-dir', type=str, default='checkpoints')
    parser.add_argument('--total-steps', type=int, default=80000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--data-size', type=int, default=50000)
    args = parser.parse_args()
    train(args)
