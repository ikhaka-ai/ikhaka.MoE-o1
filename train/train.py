Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI

"""
The training loop. Everything before this file was plumbing; this is what
you actually invoke:

    python3 -m train.train --config debug --shards-dir data/shards \\
        --max-steps 500 --checkpoint-dir checkpoints/debug-run

Expects data/tokenize.py to have already produced a manifest.json under
--shards-dir/<domain>/ for every domain in the chosen config -- it fails
fast with a clear message if one is missing, rather than silently training
on three domains instead of four.

Resume is automatic: if --checkpoint-dir already has a checkpoint, this
picks up from the saved step rather than starting over. That's the whole
point of train/checkpoint.py existing.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from data.loader import MixedDomainLoader, ShardedTokenLoader     # noqa: E402
from model import BASE_CONFIG, DEBUG_CONFIG, MoEConfig, MoETransformer  # noqa: E402
from train.checkpoint import (CheckpointConfig, CheckpointManager,  # noqa: E402
                              capture_rng, restore_rng)

log = logging.getLogger(__name__)

CONFIGS = {"debug": DEBUG_CONFIG, "base": BASE_CONFIG}


def build_loader(cfg: MoEConfig, shards_dir: Path, batch_size: int,
                 seed: int) -> MixedDomainLoader:
    loaders = {}
    missing = []
    for domain in cfg.domains:
        manifest = shards_dir / domain / "manifest.json"
        if not manifest.exists():
            missing.append(domain)
            continue
        loaders[domain] = ShardedTokenLoader(manifest, cfg.seq_len, seed=seed)
    if missing:
        raise FileNotFoundError(
            f"no manifest.json for domain(s) {missing} under {shards_dir} -- "
            f"run: python3 -m data.tokenize --domain <domain> first"
        )
    return MixedDomainLoader(loaders, cfg.domain_to_id, batch_size)


def lr_schedule(step: int, decay_steps: int, warmup_steps: int,
                peak_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to min_lr_ratio * peak_lr. Standard
    recipe (GPT-2/LLaMA/Chinchilla-family runs all use some variant); no
    part of this is MoE-specific.

    `decay_steps` is the planned TOTAL run length the schedule decays
    across -- deliberately separate from `max_steps` below, which is just
    where THIS process happens to stop. A preemption that stops a run
    early must not retroactively compress the cosine decay for the steps
    already taken; decay_steps stays fixed across a resume, only max_steps
    (this invocation's stopping point) changes."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)


def train(args: argparse.Namespace, loss_history: list | None = None) -> MoETransformer:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = CONFIGS[args.config]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = MoETransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    loader = build_loader(cfg, Path(args.shards_dir), args.batch_size, args.seed)
    batches = iter(loader)

    ckpt = CheckpointManager(CheckpointConfig(
        local_dir=Path(args.checkpoint_dir),
        save_every_seconds=args.save_every_seconds,
        save_every_steps=args.save_every_steps,
    ))

    start_step = 0
    state = ckpt.load_latest(map_location=device)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        loader.load_state_dict(state["loader"])
        restore_rng(state["rng"])
        model.load_aux_state(state["aux"])
        start_step = state["step"] + 1
        log.info("resumed at step %d", start_step)
    else:
        log.info("starting from scratch: %s config, %d total / %d active params",
                 args.config, model.num_params(), model.num_params(active_only=True))

    decay_steps = args.lr_decay_steps or args.max_steps
    t0 = time.time()
    tokens_seen = 0
    for step in range(start_step, args.max_steps):
        lr = lr_schedule(step, decay_steps, args.warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y, domain_ids = next(batches)
        x, y, domain_ids = x.to(device), y.to(device), domain_ids.to(device)

        _, loss = model(x, domain_ids, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if loss_history is not None:
            loss_history.append((step, loss.item()))

        tokens_seen += x.numel()
        if step % args.log_every == 0 or step == args.max_steps - 1:
            elapsed = time.time() - t0
            tok_per_s = tokens_seen / max(elapsed, 1e-9)
            util = torch.stack([b.moe.utilization() for b in model.blocks]).mean(0)
            util_str = ", ".join(f"{d}={u:.2f}" for d, u in zip(cfg.domains, util.tolist()))
            log.info(f"step {step:6d}  loss {loss.item():.4f}  lr {lr:.2e}  "
                     f"{tok_per_s:,.0f} tok/s  util[{util_str}]")

        if ckpt.should_save(step) or step == args.max_steps - 1:
            ckpt.save(step, {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loader": loader.state_dict(),
                "rng": capture_rng(),
                "aux": model.aux_state(),
            })
            if ckpt.preempted:
                log.warning("preempted at step %d, exiting cleanly", step)
                ckpt.close()
                return model

    ckpt.close()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=list(CONFIGS), default="debug")
    p.add_argument("--shards-dir", default="data/shards")
    p.add_argument("--checkpoint-dir", default="checkpoints/run")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--lr-decay-steps", type=int, default=None,
                  help="planned total run length for the cosine decay; "
                       "defaults to --max-steps if unset. Set this "
                       "explicitly and keep it FIXED across resumes of the "
                       "same run -- see lr_schedule()'s docstring")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--save-every-seconds", type=float, default=1500.0)
    p.add_argument("--save-every-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=1337)
    train(p.parse_args())


if __name__ == "__main__":
    main()
