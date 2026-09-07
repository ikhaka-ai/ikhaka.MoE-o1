#tokenizes each domain's corpus once. Writes immutable uint16 shards and never repeats the work.

#Run per domain, on a high-core-count CPU instance (this is CPU-bound and is going to take days at 5B tokens on a laptop)

#target-tokens for code is 5_000_000_00
#target-tokens for mathematics is 5_000_000_00
#target-tokens for physics is 5_000_000_00
#target_tokens for general is 5_000_000_00

#the dataloader.py file is what is read by the manifest which implies a partially-written shard(killed mid-run) cannot be picked up by accident.
#A shard is only added to the manifest after it's fully written and renamed from its .tmp path

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path #handles file paths using OOP
import numpy as np

#datasets/transformers are imported inside functions that need them which implies data.loader says lightweight.

#Corpus registry - one row per expert, matching model.py/config.py domain order exactly. If you add a fifth domain, add it here AND in model/config.py's domains tuples.

@dataclass
class CorpusSpec:
    domain: str
    hf_dataset: str
    hf_config: str | None
    split: str
    text_field: str
    streaming_kwargs: dict = field(default_factory=dict)

#registary to organize different text collections
CORPUS_REGISTRY: dict[str, CorpusSpec] = {
    "code": CorpusSpec(
        domain="code",
        hf_dataset="bigcode/the-stack-v2-dedup",
        hf_config=None,
        split="train",
        text_field="content",
    ),
    "mathematics": CorpusSpec(
        domain:"mathematics",
        hf_dataset="HuggingFaceTB/finemath",
        hf_config="finemath-4plus",
        split="train"
        text_field="text",
    ),
    "physics": CorpusSpec(
        domain="physics",
        hf_dataset="allenai/peS2o",
        hf_config="v3",
        split="train",
        text_field="text",
    ),
    "general": CorpusSpec(
        domain="general",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="default",
        split="train",
        text_field="text"m
    ),
}

DEFAULT_TOKENIZER = 100_000_000
EOS_TOKEN_ID_FALLBACK = 0

#returns a shuffled, streaming iterator of raw text
def _open_stream(spec: CorpusSpec, seed: int, skip_docs: int=0):
    from datasets import load_dataset

    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    if skip_docs:
        ds = ds.skip(skip_docs)

    return ds

def _load_manifest(out_dir: Path)->dict:
    manifest_path = out_dir/"manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())

    return {"domain": out_dir.name,"shards": [],"total_tokens": 0,"docs_consumed": 0, "tokenizer": None}

def _save_manifest(out_dir: Path, manifest: dict)->None:
    tmp = out_dir/"manifest.json.tmp"
    tmp.write_text(json.dumps(manifest,indent=2))
    os.replace(tmp, out_dir/"manifest.json")

def tokenize_domain(domain: str, target_tokens: int, out_root: Path,
                     tokenizer_name: str = DEFAULT_TOKENIZER,
                     shard_tokens: int = SHARD_TOKENS,
                     seed: int = 1337) -> None:
    from transformers import AutoTokenizer
 
    if domain not in CORPUS_REGISTRY:
        raise ValueError(f"unknown domain {domain!r}; add it to "
                         f"CORPUS_REGISTRY and to model/config.py's domains "
                         f"tuple first")
    spec = CORPUS_REGISTRY[domain]
    out_dir = out_root / domain
    out_dir.mkdir(parents=True, exist_ok=True)
 
    manifest = _load_manifest(out_dir)
    if manifest["tokenizer"] not in (None, tokenizer_name):
        raise RuntimeError(
            f"{out_dir} was already tokenised with "
            f"{manifest['tokenizer']!r}; refusing to mix tokenizers in one "
            f"domain's shards. Use a fresh out_dir for {tokenizer_name!r}."
        )
    manifest["tokenizer"] = tokenizer_name
 
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else EOS_TOKEN_ID_FALLBACK
    assert tok.vocab_size < 65536, (
        f"tokenizer vocab_size {tok.vocab_size} doesn't fit in uint16 -- "
        f"shards would silently wrap around and corrupt token ids"
    )
 
    stream = _open_stream(spec, seed, skip_docs=manifest["docs_consumed"])
 
    buf: list[int] = []
    shard_idx = len(manifest["shards"])
    docs_this_run = 0
    t0 = time.time()
 
    def flush_shard():
        nonlocal buf, shard_idx
        arr = np.array(buf[:shard_tokens], dtype=np.uint16)
        name = f"{domain}_{shard_idx:05d}.npy"
        tmp_path = out_dir / (name + ".tmp")
        final_path = out_dir / name
        with open(tmp_path, "wb") as f:
            np.save(f, arr)   # write via file handle -- np.save would
                              # otherwise append its own .npy suffix to a
                              # path that doesn't already end in .npy,
                              # producing "*.npy.tmp.npy" instead
        os.replace(tmp_path, final_path)   # shard only appears atomically
        manifest["shards"].append({"file": name, "tokens": int(arr.size)})
        manifest["total_tokens"] += int(arr.size)
        _save_manifest(out_dir, manifest)   # manifest updated only after
                                             # the shard it references exists
        buf = buf[shard_tokens:]
        shard_idx += 1
        elapsed = time.time() - t0
        rate = manifest["total_tokens"] / max(elapsed, 1e-9)
        print(f"[{domain}] shard {shard_idx-1:05d} written "
              f"({manifest['total_tokens']/1e6:.1f}M / "
              f"{target_tokens/1e6:.0f}M tokens, {rate/1e6:.2f}M tok/s)")
 
    for doc in stream:
        if manifest["total_tokens"] >= target_tokens:
            break
        text = doc.get(spec.text_field)
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eos_id)   # document boundary -- the model should never
                              # learn to treat two unrelated documents as
                              # one continuous stream
        buf.extend(ids)
        docs_this_run += 1
        manifest["docs_consumed"] += 1
 
        while len(buf) >= shard_tokens:
            flush_shard()
 
    # Final partial shard: written and counted, but genuinely partial --
    # the loader treats every shard in the manifest identically regardless
    # of size, so this is safe, just slightly smaller than the rest.
    if buf and manifest["total_tokens"] < target_tokens:
        flush_shard()
 
    print(f"[{domain}] done: {manifest['total_tokens']/1e9:.3f}B tokens "
          f"across {len(manifest['shards'])} shards, "
          f"{manifest['docs_consumed']} documents consumed")
 def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", required=True, choices=list(CORPUS_REGISTRY))
    p.add_argument("--target-tokens", type=int, default=5_000_000_000)
    p.add_argument("--shard-tokens", type=int, default=SHARD_TOKENS)
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--out-dir", type=Path, default=Path("data/shards"))
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()
 
    tokenize_domain(args.domain, args.target_tokens, args.out_dir,
                    args.tokenizer, args.shard_tokens, args.seed)
 
 
if __name__ == "__main__":
    main()

