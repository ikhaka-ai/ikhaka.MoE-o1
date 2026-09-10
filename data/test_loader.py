Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI

"""
Phase 0-style check for the data pipeline: fabricate tiny shards directly
(no tokenize.py, no network) and verify the loader's shapes, resumability,
and epoch behaviour -- then feed a batch straight into MoETransformer to
prove data/ and model/ actually fit together end to end.

Run: python3 -m tests.test_loader
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from data.loader import MixedDomainLoader, ShardedTokenLoader  # noqa: E402
from model import DEBUG_CONFIG, MoETransformer                  # noqa: E402


def make_fake_shards(root: Path, domain: str, n_shards: int,
                     tokens_per_shard: int, vocab: int, seed: int) -> Path:
    out_dir = root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    manifest = {"domain": domain, "tokenizer": "fake", "docs_consumed": 0,
                "shards": [], "total_tokens": 0}
    for i in range(n_shards):
        # each shard's tokens encode their own shard index in the low bits
        # so a resumed read is checkable by hand, not just by shape
        arr = (rng.randint(0, vocab, size=tokens_per_shard) & 0xFF0) | i
        arr = arr.astype(np.uint16)
        name = f"{domain}_{i:03d}.npy"
        np.save(out_dir / name, arr)
        manifest["shards"].append({"file": name, "tokens": tokens_per_shard})
        manifest["total_tokens"] += tokens_per_shard
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return out_dir / "manifest.json"


def main():
    root = Path("/tmp/loader_test")
    shutil.rmtree(root, ignore_errors=True)
    cfg = DEBUG_CONFIG
    seq_len = cfg.seq_len

    manifests = {
        d: make_fake_shards(root, d, n_shards=4, tokens_per_shard=300,
                            vocab=cfg.vocab_size, seed=i)
        for i, d in enumerate(cfg.domains)
    }

    # ---------------------------------------------------------------- (1)
    loader = ShardedTokenLoader(manifests["code"], seq_len=seq_len, seed=7)
    it = iter(loader)
    x, y = next(it)
    assert x.shape == (seq_len,) and y.shape == (seq_len,), (x.shape, y.shape)
    assert torch.equal(x[1:], y[:-1]), "y should be x shifted by one token"
    print(f"[1] ShardedTokenLoader shapes ok: x{tuple(x.shape)}, "
          f"y shifted-by-one confirmed")

    # ---------------------------------------------------------------- (2)
    # byte-identical resume: run N steps, save, resume a FRESH loader from
    # that state, confirm it produces exactly what the original would have
    # produced next.
    loader_a = ShardedTokenLoader(manifests["mathematics"], seq_len=seq_len, seed=3)
    it_a = iter(loader_a)
    for _ in range(5):
        next(it_a)
    saved_state = loader_a.state_dict()
    expected_x, expected_y = next(it_a)

    loader_b = ShardedTokenLoader(manifests["mathematics"], seq_len=seq_len, seed=3)
    loader_b.load_state_dict(saved_state)
    got_x, got_y = next(iter(loader_b))
    assert torch.equal(expected_x, got_x) and torch.equal(expected_y, got_y), (
        "resumed loader diverged from the uninterrupted one"
    )
    print(f"[2] resume ok: fresh loader from saved state reproduces the "
          f"exact next batch")

    # ---------------------------------------------------------------- (3)
    # epoch wraparound: exhaust all shards, confirm epoch increments and a
    # DIFFERENT shard order is used (since it's seed+epoch, not just seed)
    loader_c = ShardedTokenLoader(manifests["physics"], seq_len=seq_len, seed=11)
    order_epoch0 = loader_c._shard_order(0)
    order_epoch1 = loader_c._shard_order(1)
    assert order_epoch0 != order_epoch1, (
        "epoch 0 and epoch 1 produced the same shard order -- "
        "shard order isn't actually varying with epoch"
    )
    it_c = iter(loader_c)
    tokens_per_epoch = 4 * 300  # n_shards * tokens_per_shard
    steps_per_epoch = tokens_per_epoch // seq_len
    for _ in range(steps_per_epoch + 1):
        next(it_c)
    assert loader_c.pos.epoch >= 1, (
        f"expected to have wrapped into epoch >= 1, got {loader_c.pos.epoch}"
    )
    print(f"[3] epoch wraparound ok: epoch is now {loader_c.pos.epoch}, "
          f"shard order changes with epoch")

    # ---------------------------------------------------------------- (4)
    loaders = {d: ShardedTokenLoader(m, seq_len=seq_len, seed=1)
              for d, m in manifests.items()}
    mixed = MixedDomainLoader(loaders, cfg.domain_to_id, batch_size=9)
    # uneven batch size (9) over 4 domains forces the largest-remainder
    # apportionment to actually do something non-trivial
    assert sum(mixed.rows_per_domain.values()) == 9
    print(f"[4] row allocation for batch_size=9 over 4 domains: "
          f"{mixed.rows_per_domain} (sums to 9)")

    # ---------------------------------------------------------------- (5)
    missing_domain_to_id = {**cfg.domain_to_id, "chemistry": 4}
    try:
        MixedDomainLoader(loaders, missing_domain_to_id, batch_size=8)
        raise AssertionError("expected ValueError for a domain with no loader")
    except ValueError as e:
        print(f"[5] missing-domain guard ok: {e}")

    # ---------------------------------------------------------------- (6)
    # the actual point of this file: data/ output feeds model/ input
    batch_iter = iter(mixed)
    x, y, domain_ids = next(batch_iter)
    assert x.shape == (9, seq_len)
    assert domain_ids.shape == (9,)
    assert set(domain_ids.tolist()) <= set(cfg.domain_to_id.values())

    model = MoETransformer(cfg)
    logits, loss = model(x, domain_ids, targets=y)
    assert logits.shape == (9, seq_len, cfg.vocab_size)
    loss.backward()
    print(f"[6] end-to-end ok: a real MixedDomainLoader batch trains the "
          f"real model. loss={loss.item():.3f}")

    # ---------------------------------------------------------------- (7)
    mixed_state = mixed.state_dict()
    assert set(mixed_state) == set(cfg.domains)
    loaders_2 = {d: ShardedTokenLoader(m, seq_len=seq_len, seed=1)
                for d, m in manifests.items()}
    mixed_2 = MixedDomainLoader(loaders_2, cfg.domain_to_id, batch_size=9)
    mixed_2.load_state_dict(mixed_state)
    print(f"[7] MixedDomainLoader state_dict/load_state_dict round-trips "
          f"across all {len(mixed_state)} domains")

    shutil.rmtree(root, ignore_errors=True)
    print("\nall loader checks passed. data/ and model/ are wired together.")


if __name__ == "__main__":
    main()
