"""
Reads what data/tokenize.py wrote and turns it into the (input_ids, targets,
domain_ids) batches model.MoETransformer.forward() expects.

Two classes, one per level of the problem:

  ShardedTokenLoader  -- one domain's stream of individual (x, y) sequences,
                         reading shards in an order that is a pure function
                         of (seed, epoch) rather than a stored permutation.
  MixedDomainLoader    -- combines one ShardedTokenLoader per domain into a
                         single batch carrying a per-sequence domain_id,
                         matching exactly how train/smoke_test.py already
                         exercises the model (a batch with several domains
                         mixed together, not a batch-per-domain scheme).

Both are resumable: state_dict()/load_state_dict() round-trip through
checkpointing.py's CheckpointManager unchanged from the sketch discussed
earlier in this project.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------
# Single-domain loader
# --------------------------------------------------------------------------

@dataclass
class LoaderPosition:
    epoch: int = 0
    shard_cursor: int = 0     # index into this epoch's shuffled shard order
    token_offset: int = 0     # position within the currently open shard


class ShardedTokenLoader:
    """
    Reads one domain's shards, as listed in its manifest.json -- never by
    globbing the directory, so a shard that tokenize.py hasn't finished
    writing yet is simply invisible rather than half-read.

    Shard order is re-derived from (seed, epoch) on every epoch rather than
    stored as an explicit permutation, which is what keeps a saved position
    down to three integers (see LoaderPosition) instead of a shard-count-
    sized array -- the same trick used for the token-level shard order in
    the earlier checkpointing design.
    """

    def __init__(self, manifest_path: Path, seq_len: int, seed: int = 1337):
        self.manifest_path = Path(manifest_path)
        manifest = json.loads(self.manifest_path.read_text())
        self.domain = manifest["domain"]
        self.shard_dir = self.manifest_path.parent
        self.shards: list[dict] = manifest["shards"]
        if not self.shards:
            raise ValueError(
                f"{manifest_path} lists zero shards -- has tokenize.py "
                f"finished a run for this domain yet?"
            )
        self.seq_len = seq_len
        self.seed = seed
        self.pos = LoaderPosition()
        self._tokens: np.ndarray | None = None
        self._loaded_shard: int | None = None

    def _shard_order(self, epoch: int) -> list[int]:
        order = list(range(len(self.shards)))
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def _ensure_shard(self, shard_idx: int) -> None:
        if self._loaded_shard != shard_idx:
            path = self.shard_dir / self.shards[shard_idx]["file"]
            self._tokens = np.load(path, mmap_mode="r")
            self._loaded_shard = shard_idx

    def state_dict(self) -> dict:
        return {"epoch": self.pos.epoch, "shard_cursor": self.pos.shard_cursor,
                "token_offset": self.pos.token_offset}

    def load_state_dict(self, sd: dict) -> None:
        self.pos = LoaderPosition(**sd)
        self._loaded_shard = None   # force a reload on the next access

    def __iter__(self):
        """Yields (input_ids, target_ids) pairs, each shape (seq_len,).
        Runs forever, wrapping to a freshly-shuffled epoch when shards run
        out -- exhaustion is expected and silent, since every domain here
        has far more available tokens than the 5B-token take (design doc
        Section 5), so repeats across epochs are the normal case, not an
        error condition."""
        span = self.seq_len + 1
        while True:
            order = self._shard_order(self.pos.epoch)
            while self.pos.shard_cursor < len(order):
                shard_idx = order[self.pos.shard_cursor]
                self._ensure_shard(shard_idx)
                assert self._tokens is not None

                while self.pos.token_offset + span <= len(self._tokens):
                    lo = self.pos.token_offset
                    buf = torch.from_numpy(
                        self._tokens[lo:lo + span].astype(np.int64)
                    )
                    self.pos.token_offset += span - 1   # 1-token overlap:
                        # target[-1] of this window becomes input[0] of the
                        # next only if span-1 is the stride; using span-1
                        # here means NO overlap between consecutive windows,
                        # matching standard next-token pretraining framing
                    yield buf[:-1], buf[1:]

                self.pos.shard_cursor += 1
                self.pos.token_offset = 0
            self.pos.epoch += 1
            self.pos.shard_cursor = 0


# --------------------------------------------------------------------------
# Mixed-domain batch assembly
# --------------------------------------------------------------------------

class MixedDomainLoader:
    """
    Pulls from one ShardedTokenLoader per domain to build batches carrying
    a per-SEQUENCE domain_id, exactly the shape DemixMoE.forward() expects
    -- see train/smoke_test.py, which exercises the model with precisely
    this kind of uneven, mixed-domain batch.

    `mix` sets how many rows of each batch come from which domain. Default
    is equal split, matching the design doc's equal 5B-token take per
    domain (Section 5) -- change it deliberately if you want the model to
    see more of one domain than another, rather than let it drift from
    whatever order dict() happens to iterate in.
    """

    def __init__(self, loaders: dict[str, ShardedTokenLoader],
                 domain_to_id: dict[str, int], batch_size: int,
                 mix: dict[str, float] | None = None):
        missing = set(domain_to_id) - set(loaders)
        if missing:
            raise ValueError(f"no loader provided for domains {missing} -- "
                             f"every domain in domain_to_id needs a shard "
                             f"manifest before training can start")
        self.loaders = loaders
        self.domain_to_id = domain_to_id
        self.batch_size = batch_size

        mix = mix or {d: 1.0 for d in loaders}
        total = sum(mix.values())
        self.rows_per_domain = self._allocate_rows(mix, total, batch_size)

        self._iters = {d: iter(loader) for d, loader in loaders.items()}

    @staticmethod
    def _allocate_rows(mix: dict[str, float], total: float,
                       batch_size: int) -> dict[str, int]:
        """Turns proportions into integer row counts that sum EXACTLY to
        batch_size -- largest-remainder apportionment, so no domain is
        silently dropped to a rounding error on an odd batch size."""
        raw = {d: batch_size * w / total for d, w in mix.items()}
        floors = {d: int(v) for d, v in raw.items()}
        remainder = batch_size - sum(floors.values())
        # give the leftover rows to the domains with the largest fractional
        # part, so the allocation is deterministic and reproducible
        fracs = sorted(raw, key=lambda d: raw[d] - floors[d], reverse=True)
        for d in fracs[:remainder]:
            floors[d] += 1
        assert sum(floors.values()) == batch_size
        return floors

    def state_dict(self) -> dict:
        return {d: loader.state_dict() for d, loader in self.loaders.items()}

    def load_state_dict(self, sd: dict) -> None:
        for d, loader in self.loaders.items():
            if d in sd:
                loader.load_state_dict(sd[d])
        self._iters = {d: iter(loader) for d, loader in self.loaders.items()}

    def __iter__(self):
        while True:
            xs, ys, domain_ids = [], [], []
            for domain, n_rows in self.rows_per_domain.items():
                did = self.domain_to_id[domain]
                for _ in range(n_rows):
                    x, y = next(self._iters[domain])
                    xs.append(x); ys.append(y); domain_ids.append(did)

            # Shuffle row order within the batch -- otherwise every batch
            # is domain-sorted (all code rows, then all maths rows, ...),
            # which costs nothing for correctness but would make loss
            # curves logged per-batch harder to read.
            order = list(range(self.batch_size))
            random.shuffle(order)
            x = torch.stack([xs[i] for i in order])
            y = torch.stack([ys[i] for i in order])
            d = torch.tensor([domain_ids[i] for i in order], dtype=torch.long)
            yield x, y, d
