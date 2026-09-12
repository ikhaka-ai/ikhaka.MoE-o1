"""
Phase 2 from the design document: before spending $20-26 on the real 20B-
token run, spend about $5 confirming the model doesn't diverge and that the
loss curve extrapolates to something sane at 20B tokens.

Each rung is an independent, fully-annealed short run at its own token
budget -- not a slice of one long schedule. That's deliberate: the question
this answers is "does loss at token budget D look like a healthy point on
a scaling curve", which needs each D to have its own complete warmup+decay,
the same way the real run will.

Run:
    python3 -m train.ladder --config base --shards-dir data/shards

Debug/local dry run (fabricated-shard territory, seconds not hours):
    python3 -m train.ladder --config debug --shards-dir /tmp/fake_shards \\
        --rungs 2000,8000,32000 --target-tokens 200000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from model import BASE_CONFIG, DEBUG_CONFIG                # noqa: E402
from train.train import build_loader, train                 # noqa: E402

log = logging.getLogger(__name__)

CONFIGS = {"debug": DEBUG_CONFIG, "base": BASE_CONFIG}

# Design doc Section 9, Phase 2's rungs.
DEFAULT_RUNGS = "100_000_000,500_000_000,2_000_000_000"


def tokens_to_steps(tokens: int, batch_size: int, seq_len: int) -> int:
    steps = tokens // (batch_size * seq_len)
    return max(steps, 1)


def fit_power_law(tokens: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    """Fits log(loss) = log(A) - alpha*log(tokens) by least squares.

    This ignores the irreducible-loss floor a full Chinchilla fit would
    include (L = E + A/D^alpha) -- with only 3 points, fitting a 3rd free
    parameter isn't well-conditioned. A bare power law is enough for a
    sanity check: what matters here is whether the extrapolation is
    plausible, not a publishable scaling exponent."""
    log_t, log_l = np.log(tokens), np.log(losses)
    neg_alpha, log_A = np.polyfit(log_t, log_l, 1)
    return float(np.exp(log_A)), float(-neg_alpha)


def run_rung(config_name, cfg, shards_dir: Path, checkpoint_root: Path,
            tokens: int, batch_size: int, lr: float, warmup_steps: int,
            seed: int, log_every: int) -> dict:
    steps = tokens_to_steps(tokens, batch_size, cfg.seq_len)
    ckpt_dir = checkpoint_root / f"rung_{tokens}"
    args = argparse.Namespace(
        config=config_name, shards_dir=str(shards_dir),
        checkpoint_dir=str(ckpt_dir), batch_size=batch_size,
        max_steps=steps, lr=lr, lr_decay_steps=steps,
        warmup_steps=min(warmup_steps, max(1, steps // 10)),
        save_every_seconds=1e9, save_every_steps=None,
        log_every=log_every, seed=seed,
    )
    history: list[tuple[int, float]] = []
    log.info(f"--- rung: {tokens:,} tokens -> {steps:,} steps ---")
    train(args, loss_history=history)

    if not history:
        # Rung was already fully trained in a prior session (this call's
        # train() resumed straight to the end and never ran a step) --
        # its loss curve lived only in that earlier process's memory and
        # didn't survive the disconnect. Report it as complete without
        # loss data rather than fabricate a number or crash.
        print(f"  (rung {tokens:,} tokens was already complete from a "
             f"prior session -- no loss curve available this run)")
        return {"tokens": tokens, "steps": steps, "final_loss": None,
                "first_loss": None, "history": []}

    losses = np.array([l for _, l in history])
    if np.isnan(losses).any() or np.isinf(losses).any():
        raise RuntimeError(
            f"rung {tokens:,} tokens produced NaN/Inf loss -- this rung "
            f"diverged. Do not proceed to the full run; check learning "
            f"rate, gradient clipping, and data before retrying."
        )

    tail = max(3, len(losses) // 10)
    final_loss = float(losses[-tail:].mean())
    return {"tokens": tokens, "steps": steps, "final_loss": final_loss,
            "first_loss": float(losses[0]), "history": history}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=list(CONFIGS), default="base")
    p.add_argument("--shards-dir", default="data/shards")
    p.add_argument("--checkpoint-root", default="checkpoints/ladder")
    p.add_argument("--rungs", default=DEFAULT_RUNGS,
                   help="comma-separated token budgets, smallest first")
    p.add_argument("--target-tokens", type=int, default=20_000_000_000,
                   help="the real run's planned budget, to project loss at "
                        "(design doc's 'Recommended' 20B)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = CONFIGS[args.config]
    rungs = sorted(int(t) for t in args.rungs.split(","))
    if len(rungs) < 3:
        log.warning(f"only {len(rungs)} rung(s) given -- a power-law fit "
                    f"needs at least 3 points to be meaningful")

    results = []
    for tokens in rungs:
        results.append(run_rung(
            args.config, cfg, Path(args.shards_dir),
            Path(args.checkpoint_root), tokens, args.batch_size, args.lr,
            args.warmup_steps, args.seed, args.log_every,
        ))

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 64)
    print(f"{'tokens':>14}  {'steps':>10}  {'first loss':>10}  {'final loss':>10}")
    for r in results:
        print(f"{r['tokens']:>14,}  {r['steps']:>10,}  "
              f"{r['first_loss']:>10.4f}  {r['final_loss']:>10.4f}")

    REGRESSION_MARGIN = 1.01
    regressed = [r for r in results
                if r["final_loss"] > r["first_loss"] * REGRESSION_MARGIN]
    if regressed:
        bad = ", ".join(f"{r['tokens']:,} ({r['first_loss']:.2f}\u2192"
                        f"{r['final_loss']:.2f})" for r in regressed)
        print(f"\nWARNING: rung(s) ended with a HIGHER loss than they "
              f"started: {bad}. This is a stronger signal than the "
              f"cross-rung check below -- a rung that regresses within "
              f"itself is unhealthy even if later, longer rungs look "
              f"better on paper (a decaying LR can cosmetically shrink the "
              f"loss of a run that already diverged, without it ever "
              f"recovering to a reasonable value). Check the learning rate "
              f"and gradient clipping before trusting the fit below.")

    increasing = [i for i in range(1, len(results))
                 if results[i]["final_loss"] > results[i - 1]["final_loss"]]
    if increasing:
        bad = ", ".join(f"{results[i]['tokens']:,}" for i in increasing)
        print(f"\nWARNING: loss got WORSE at a larger token budget for "
              f"rung(s) {bad}. This should not happen on a healthy scaling "
              f"curve -- check for an LR that's too high, a data pipeline "
              f"bug feeding a rung bad shards, or a genuinely undertrained "
              f"tiny rung (widen --warmup-steps relative to --rungs if the "
              f"smallest rung is only a handful of steps).")

    tokens_arr = np.array([r["tokens"] for r in results], dtype=float)
    loss_arr = np.array([r["final_loss"] for r in results])
    A, alpha = fit_power_law(tokens_arr, loss_arr)
    projected = A * args.target_tokens ** (-alpha)
    print(f"\nfitted power law: loss \u2248 {A:.3f} * tokens^-{alpha:.3f}")
    print(f"projected loss at {args.target_tokens:,} tokens: {projected:.4f}")
    print("=" * 64)

    if not increasing and not regressed:
        print(f"\nno divergence detected across {len(results)} rungs. "
              f"safe to proceed to Phase 3 (design doc Section 9).")


if __name__ == "__main__":
    main()
