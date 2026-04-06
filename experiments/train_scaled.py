#!/usr/bin/env python3
"""
Train MiniSUBLEQTransformer at arbitrary d_model (capacity scaling sweep).

Architecture: same as existing trained model (GELU, Pre-LN, n_layers=6, n_heads=8).
d_ff = 4 * d_model.  Vary only d_model: 32, 64, 128.  (256 already exists.)

Usage:
    python train_scaled.py --d-model 32  --seed 0
    python train_scaled.py --d-model 64  --seed 1
    python train_scaled.py --d-model 128 --seed 2
"""

import os, sys, time, math, random, argparse
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root  = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(repo_root, 'round2_trained'))

from subleq import pregenerate_data
from subleq.tokenizer import SEQ_LEN, VOCAB_SIZE
from subleq.data import CHANGE_WEIGHT
from subleq import MiniSUBLEQTransformer


def weighted_cross_entropy(logits, targets, weight_mask):
    B, S, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B*S, V), targets.reshape(B*S), reduction='none')
    loss = (loss.reshape(B, S) * weight_mask).sum() / weight_mask.sum()
    return loss


def compute_accuracy(logits, targets, weight_mask=None):
    preds = logits.argmax(dim=-1)
    all_correct = (preds == targets).all(dim=1).float().mean().item()
    if weight_mask is not None:
        changed = weight_mask > 1.0
        changed_correct = ((preds == targets) | ~changed).all(dim=1).float().mean().item() \
            if changed.any() else all_correct
    else:
        changed_correct = all_correct
    return all_correct, changed_correct


def get_curriculum_range(step, total_steps):
    f = step / total_steps
    if f < 0.10: return (1, 2)
    if f < 0.25: return (1, 4)
    if f < 0.45: return (1, 6)
    return (1, 8)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d  = args.d_model
    nh = 8                  # n_heads fixed (d_head = d/8)
    dff = 4 * d             # d_ff = 4 * d_model (same ratio as existing trained model)
    nl = 6                  # n_layers fixed

    print(f"Device:  {device}")
    print(f"d_model={d}, n_heads={nh}, n_layers={nl}, d_ff={dff}")
    print(f"SEQ_LEN={SEQ_LEN}, VOCAB_SIZE={VOCAB_SIZE}, seed={args.seed}")

    model = MiniSUBLEQTransformer(
        d_model=d, n_heads=nh, n_layers=nl, d_ff=dff,
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {model.count_params():,}")

    save_dir = os.path.join(args.ckpt_base, f'scaled_d{d}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        p = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    instr_range = get_curriculum_range(0, args.total_steps)
    print(f"Generating {args.data_size} examples (instr_range={instr_range})...")
    data_inp, data_out, data_mask = pregenerate_data(args.data_size, instr_range=instr_range)
    data_inp, data_out, data_mask = data_inp.to(device), data_out.to(device), data_mask.to(device)

    best_acc = 0.0
    log_loss = log_full = log_chg = log_n = 0.0
    t0 = time.time()

    for step in range(1, args.total_steps + 1):
        model.train()
        idx = torch.randint(0, data_inp.size(0), (args.batch_size,))
        logits = model(data_inp[idx])
        loss = weighted_cross_entropy(logits, data_out[idx], data_mask[idx])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step()

        with torch.no_grad():
            fa, ca = compute_accuracy(logits, data_out[idx], data_mask[idx])
        log_loss += loss.item(); log_full += fa; log_chg += ca; log_n += 1

        if step % args.log_every == 0:
            print(f"Step {step:6d}/{args.total_steps} | loss={log_loss/log_n:.4f} | "
                  f"full_acc={log_full/log_n:.3f} | changed_acc={log_chg/log_n:.3f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                  f"instr={get_curriculum_range(step, args.total_steps)} | "
                  f"time={time.time()-t0:.0f}s")
            log_loss = log_full = log_chg = log_n = 0.0

        if step % args.regen_every == 0:
            instr_range = get_curriculum_range(step, args.total_steps)
            print(f"  Regenerating data (instr_range={instr_range})...")
            data_inp, data_out, data_mask = pregenerate_data(args.data_size, instr_range=instr_range)
            data_inp, data_out, data_mask = data_inp.to(device), data_out.to(device), data_mask.to(device)

        if step % args.eval_every == 0:
            model.eval()
            with torch.no_grad():
                ei, eo, em = pregenerate_data(2000, instr_range=(1, 8))
                ei, eo, em = ei.to(device), eo.to(device), em.to(device)
                el = model(ei)
                ef, ec = compute_accuracy(el, eo, em)
                eloss = weighted_cross_entropy(el, eo, em).item()
            print(f"  EVAL step {step}: loss={eloss:.4f} | full_acc={ef:.3f} | changed_acc={ec:.3f}")

            if ef > best_acc:
                best_acc = ef
                torch.save({
                    'step': step, 'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(), 'best_acc': best_acc,
                    'config': {'d_model': d, 'n_heads': nh, 'n_layers': nl, 'd_ff': dff,
                               'dropout': args.dropout, 'seed': args.seed},
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  New best: {best_acc:.3f} (saved)")

    torch.save({
        'step': args.total_steps, 'model_state': model.state_dict(),
        'best_acc': best_acc,
        'config': {'d_model': d, 'n_heads': nh, 'n_layers': nl, 'd_ff': dff,
                   'dropout': args.dropout, 'seed': args.seed},
    }, os.path.join(save_dir, 'final_model.pt'))
    print(f"\nTraining complete. Best accuracy: {best_acc:.3f}")
    print(f"Saved to {save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model',     type=int,   required=True)
    parser.add_argument('--seed',        type=int,   default=0)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--weight-decay',type=float, default=0.01)
    parser.add_argument('--warmup-steps',type=int,   default=500)
    parser.add_argument('--total-steps', type=int,   default=80000)
    parser.add_argument('--batch-size',  type=int,   default=256)
    parser.add_argument('--data-size',   type=int,   default=100000)
    parser.add_argument('--grad-clip',   type=float, default=1.0)
    parser.add_argument('--log-every',   type=int,   default=200)
    parser.add_argument('--eval-every',  type=int,   default=2000)
    parser.add_argument('--regen-every', type=int,   default=2000)
    parser.add_argument('--ckpt-base',   type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'))
    args = parser.parse_args()
    train(args)
