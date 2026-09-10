Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI

"""
The real test of train.py and checkpoint.py: not just "does it run", but
"does an interrupted run resume to EXACTLY where an uninterrupted run would
have gone" -- the property the whole checkpoint design exists to guarantee.

Run: python3 -m tests.test_train
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from model import DEBUG_CONFIG                      # noqa: E402
from train.train import build_loader, train          # noqa: E402


def make_fake_shards(root: Path, domain: str, n_shards: int,
                     tokens_per_shard: int, vocab: int, seed: int) -> None:
    out_dir = root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    manifest = {"domain": domain, "tokenizer": "fake", "docs_consumed": 0,
                "shards": [], "total_tokens": 0}
    for i in range(n_shards):
        arr = rng.randint(0, vocab, size=tokens_per_shard).astype(np.uint16)
        name = f"{domain}_{i:03d}.npy"
        np.save(out_dir / name, arr)
        manifest["shards"].append({"file": name, "tokens": tokens_per_shard})
        manifest["total_tokens"] += tokens_per_shard
    (out_dir / "manifest.json").write_text(json.dumps(manifest))


def args_for(shards_dir, ckpt_dir, max_steps, lr_decay_steps=10,
             save_every_steps=None):
    return argparse.Namespace(
        config="debug", shards_dir=str(shards_dir),
        checkpoint_dir=str(ckpt_dir), batch_size=6, max_steps=max_steps,
        lr=1e-3, lr_decay_steps=lr_decay_steps, warmup_steps=2,
        save_every_seconds=1e9, save_every_steps=save_every_steps,
        log_every=1000, seed=1337,
    )


def main():
    root = Path("/tmp/train_test")
    shutil.rmtree(root, ignore_errors=True)
    shards_dir = root / "shards"
    cfg = DEBUG_CONFIG
    for i, d in enumerate(cfg.domains):
        make_fake_shards(shards_dir, d, n_shards=6, tokens_per_shard=400,
                         vocab=cfg.vocab_size, seed=i)

    # ---------------------------------------------------------------- (1)
    ckpt_uninterrupted = root / "ckpt_uninterrupted"
    loss_full: list = []
    model_full = train(args_for(shards_dir, ckpt_uninterrupted, max_steps=10),
                       loss_history=loss_full)
    assert not any(torch.isnan(p).any() for p in model_full.parameters()), (
        "NaN in model parameters after 10 steps"
    )
    print("[1] uninterrupted 10-step run completed, no NaNs")

    # ---------------------------------------------------------------- (2)
    ckpt_a = root / "ckpt_a"
    loss_phase1: list = []
    train(args_for(shards_dir, ckpt_a, max_steps=5), loss_history=loss_phase1)

    manifest_a = json.loads((ckpt_a / "latest.json").read_text())
    assert manifest_a["step"] == 4, manifest_a   # 0-indexed, so step 4 is the 5th
    print(f"[2] first phase stopped cleanly at step {manifest_a['step']}")

    # ---------------------------------------------------------------- (3)
    # resume into a FRESH set of objects (fresh model/optimizer/loader --
    # this is what actually happens across a Colab session boundary) and
    # run 5 more steps
    loss_phase2: list = []
    resumed_model = train(args_for(shards_dir, ckpt_a, max_steps=10),
                          loss_history=loss_phase2)
    manifest_resumed = json.loads((ckpt_a / "latest.json").read_text())
    assert manifest_resumed["step"] == 9, manifest_resumed
    print(f"[3] resume continued to step {manifest_resumed['step']} "
          f"(started fresh objects from the step-4 checkpoint)")

    # loss trajectory is the sharper signal: it's what the earlier
    # checkpointing design was actually trying to guarantee, and it should
    # match far more tightly than raw parameters, since it's only one
    # forward pass deep rather than an accumulation of many optimizer steps
    combined = loss_phase1 + loss_phase2
    assert [s for s, _ in combined] == list(range(10)), combined
    for (step, l1), (_, l2) in zip(loss_full, combined):
        assert abs(l1 - l2) < 1e-3, (
            f"step {step}: loss diverged, {l1:.6f} vs {l2:.6f} -- "
            f"resume picked up the wrong data or optimizer state"
        )
    print(f"[3b] loss trajectory matches at all 10 steps "
          f"(uninterrupted vs interrupted+resumed), within 1e-3")

    # ---------------------------------------------------------------- (4)
    # the actual guarantee: interrupted-then-resumed and uninterrupted
    # should follow an IDENTICAL loss trajectory, since both consumed the
    # same 10 batches in the same order against the same LR schedule.
    #
    # Parameters themselves are compared too, but loosely (atol=1e-4, not
    # 1e-6): PyTorch's CPU matmul is multi-threaded, and summing the same
    # numbers in a different thread-partition order is not bit-identical
    # floating point -- this is ordinary non-associativity between two
    # separate process launches, not a resume defect. A genuine resume bug
    # (wrong optimizer state, wrong RNG, wrong data position) produces
    # differences many orders of magnitude larger than this, so atol=1e-4
    # is still a strict, meaningful check.
    max_diff = 0.0
    for (n1, p1), (n2, p2) in zip(model_full.named_parameters(),
                                  resumed_model.named_parameters()):
        max_diff = max(max_diff, (p1 - p2).abs().max().item())
    assert max_diff < 1e-4, (
        f"max parameter divergence {max_diff:.2e} is too large to be "
        f"floating-point noise -- resume is likely dropping state"
    )
    print(f"[4] uninterrupted vs interrupted+resumed: max parameter "
          f"divergence {max_diff:.2e} (floating-point noise only; a real "
          f"resume bug would show up as >>1e-4). resume is correct.")

    # ---------------------------------------------------------------- (5)
    # checkpoint pruning: run long enough with frequent saves to exceed
    # keep_last, confirm old step dirs actually get removed
    ckpt_b = root / "ckpt_b"
    train(args_for(shards_dir, ckpt_b, max_steps=8, save_every_steps=1))
    step_dirs = sorted(ckpt_b.glob("step_*"))
    assert len(step_dirs) == 3, (
        f"expected keep_last=3 step dirs, found {len(step_dirs)}: {step_dirs}"
    )
    print(f"[5] pruning ok: {len(step_dirs)} step directories retained "
          f"(keep_last=3), older ones removed")

    shutil.rmtree(root, ignore_errors=True)
    print("\nall train.py / checkpoint.py checks passed.")


if __name__ == "__main__":
    main()
