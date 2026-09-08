"""
Phase 1 from the design document: tokenise each domain's corpus once,
write immutable uint16 shards, and never repeat the work.

Run per domain, on a high-core-count CPU instance (this is CPU-bound and
genuinely takes days at 5B tokens on a laptop -- see design doc Section 5):

    python3 -m data.tokenize --domain code           --target-tokens 5_000_000_000
    python3 -m data.tokenize --domain mathematics    --target-tokens 5_000_000_000
    python3 -m data.tokenize --domain physics        --target-tokens 5_000_000_000
    python3 -m data.tokenize --domain general        --target-tokens 5_000_000_000

Each run writes to data/shards/<domain>/ :
    <domain>_00000.npy, <domain>_00001.npy, ...   (uint16, fixed token count)
    manifest.json                                  (shard list, counts, provenance)

The manifest is what data/loader.py reads -- it never globs the directory,
so a partially-written shard (killed mid-run) can't be picked up by
accident. A shard is only added to the manifest after it's fully written
and renamed from its .tmp path, matching the atomic-commit pattern used
for training checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# datasets/transformers are only imported inside functions that need them,
# so `python3 -m data.loader` (no HF dependency) stays lightweight -- see
# data/loader.py, which is what actually runs on the training box.


# --------------------------------------------------------------------------
# Corpus registry -- one row per expert, matching model/config.py's domain
# order exactly. If you add a fifth domain, add it here AND in
# model/config.py's `domains` tuple; nothing here defaults silently.
# --------------------------------------------------------------------------

@dataclass
class CorpusSpec:
    domain: str
    hf_dataset: str
    hf_config: str | None
    split: str
    text_field: str
    streaming_kwargs: dict = field(default_factory=dict)
    # True for domains that bypass datasets.load_dataset() entirely -- see
    # _open_stream_direct's docstring for why (Xet protocol failures,
    # script-based repos datasets refuses to load, etc.)
    use_direct_download: bool = False
    # For use_direct_download domains only: restricts file listing to this
    # path prefix within the repo. Needed when a repo bundles multiple
    # releases together (peS2o's repo has data/v1/, data/v2/, data/v3/ all
    # in one place) and only one is the documented, intended release.
    direct_path_prefix: str | None = None


# Design doc Section 5's source column, translated into loadable HF ids.
# These are the primary source per domain; if a listed dataset is
# unavailable or gated when you actually run this, that's the moment to
# widen the domain (per the design doc's note on physics) rather than
# silently substitute something narrower.
CORPUS_REGISTRY: dict[str, CorpusSpec] = {
    "code": CorpusSpec(
        # NOT the-stack-v2-dedup: despite the name, that dataset ships only
        # metadata and Software Heritage blob IDs, not file content -- every
        # row's text field is empty, and reconstructing real content
        # requires a separate AWS-credentialed download from Software
        # Heritage's S3 bucket (hours by itself, per BigCode's own docs).
        # starcoderdata is BigCode's earlier corpus that ships real content
        # directly in the parquet files -- what this pipeline actually needs.
        # Gated separately from the-stack-v2-dedup -- accept its license at
        # huggingface.co/datasets/bigcode/starcoderdata before running this.
        domain="code",
        hf_dataset="bigcode/starcoderdata",
        hf_config=None,
        split="train",
        text_field="content",
        use_direct_download=True,
    ),
    "mathematics": CorpusSpec(
        domain="mathematics",
        hf_dataset="HuggingFaceTB/finemath",
        hf_config="finemath-4plus",
        split="train",
        text_field="text",
    ),
    "physics": CorpusSpec(
        # NOT hf_config="v3": that isn't a documented release (an
        # undocumented data/v3/ folder exists in the repo, but the README
        # only describes and recommends v1 and v2). peS2o also ships via an
        # old-style loading script (peS2o.py), which newer `datasets`
        # versions refuse to run without trust_remote_code -- and even with
        # that, it's a script the maintainers, not this project, control.
        # use_direct_download bypasses datasets entirely: list the repo's
        # actual files, restrict to the documented v2 release via
        # direct_path_prefix, and read its .json.gz files straight.
        domain="physics",
        hf_dataset="allenai/peS2o",
        hf_config=None,
        split="train",
        text_field="text",
        use_direct_download=True,
        direct_path_prefix="data/v2/train-",
    ),
    "general": CorpusSpec(
        domain="general",
        hf_dataset="HuggingFaceFW/fineweb-edu",
        hf_config="default",
        split="train",
        text_field="text",
    ),
}

# StarCoder2's tokenizer: BPE, vocab_size 49,152 -- matches
# model/config.py's MoEConfig.vocab_size exactly, and was trained on a
# code+web mix, which is a reasonable shared vocabulary across all four
# domains here (rather than a code-only or web-only tokenizer).
DEFAULT_TOKENIZER = "bigcode/starcoder2-7b"

SHARD_TOKENS = 100_000_000   # ~200MB per shard as uint16
EOS_TOKEN_ID_FALLBACK = 0    # overwritten by the tokenizer's real eos id


def _open_stream(spec: CorpusSpec, seed: int, skip_docs: int = 0,
                 shuffle_buffer: int = 10_000):
    """Returns a shuffled, streaming iterator of raw text, resumed past
    `skip_docs` documents if resuming a partial run.

    Dispatches on `spec.use_direct_download`, set explicitly per domain in
    CORPUS_REGISTRY rather than inferred from anything structural -- each
    domain was switched to the direct path only after confirming it hits a
    specific, named problem with the normal `datasets` streaming route
    (Xet CAS failures for code, a blocked loading script for physics).
    Don't pre-emptively switch a domain that hasn't shown a problem: the
    normal path lets `datasets`' own config/file resolution do work
    (finding which files belong to which config) that would be easy to
    get subtly wrong by hand.
    """
    if spec.use_direct_download:
        return _open_stream_direct(spec, seed, skip_docs)

    from datasets import load_dataset

    ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split,
                       streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    if skip_docs:
        ds = ds.skip(skip_docs)
    return ds


def _iter_parquet_rows(local_path: str):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(local_path)
    for batch in pf.iter_batches(batch_size=1000):
        yield from batch.to_pylist()


def _iter_jsonl_gz_rows(local_path: str):
    import gzip
    with gzip.open(local_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


_DIRECT_READERS = {
    ".parquet": _iter_parquet_rows,
    ".json.gz": _iter_jsonl_gz_rows,
}


def _open_stream_direct(spec: CorpusSpec, seed: int, skip_docs: int):
    """Downloads whole files one at a time via hf_hub_download (plain
    HTTP, following redirects, with huggingface_hub's own retry logic) and
    reads rows out of each locally -- slower to reach the first row than
    true streaming, since it waits for a full file rather than a row at a
    time, but avoids both the Xet CAS protocol and, for script-based
    repos, `datasets`' refusal to run untrusted loading code.

    Handles whichever of `_DIRECT_READERS`' extensions the repo's files
    actually use (parquet for starcoderdata, json.gz for peS2o) -- checked
    per file, so a repo mixing formats would still work, though none of
    the currently-configured ones do.

    `skip_docs` is honoured at the row level across files in shuffle
    order, matching what the datasets-streaming path guarantees for
    resume: rerunning with the same seed reproduces the same file order,
    and skip_docs fast-forwards past whichever rows were already consumed.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    all_files = api.list_repo_files(spec.hf_dataset, repo_type="dataset")
    if spec.direct_path_prefix:
        all_files = [f for f in all_files if f.startswith(spec.direct_path_prefix)]
    files = sorted(f for f in all_files
                   if any(f.endswith(ext) for ext in _DIRECT_READERS))
    if not files:
        raise RuntimeError(
            f"no files with a supported extension ({list(_DIRECT_READERS)}) "
            f"found in {spec.hf_dataset}"
            + (f" under prefix {spec.direct_path_prefix!r}" if spec.direct_path_prefix else "")
            + " -- check the repo layout, it may not match the "
              "direct-download path's assumptions"
        )
    order = list(range(len(files)))
    random.Random(seed).shuffle(order)

    doc_idx = 0
    for i in order:
        filename = files[i]
        ext = next(e for e in _DIRECT_READERS if filename.endswith(e))
        local_path = hf_hub_download(spec.hf_dataset, filename, repo_type="dataset")
        for row in _DIRECT_READERS[ext](local_path):
            if doc_idx < skip_docs:
                doc_idx += 1
                continue
            doc_idx += 1
            yield row


def _load_manifest(out_dir: Path) -> dict:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {"domain": out_dir.name, "shards": [], "total_tokens": 0,
            "docs_consumed": 0, "tokenizer": None}


def _save_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = out_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, out_dir / "manifest.json")   # atomic on POSIX


def tokenize_domain(domain: str, target_tokens: int, out_root: Path,
                     tokenizer_name: str = DEFAULT_TOKENIZER,
                     shard_tokens: int = SHARD_TOKENS,
                     seed: int = 1337, shuffle_buffer: int = 10_000,
                     eos_token_id_override: int | None = None) -> None:
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
    if eos_token_id_override is not None:
        eos_id = eos_token_id_override
    else:
        eos_id = tok.eos_token_id if tok.eos_token_id is not None else EOS_TOKEN_ID_FALLBACK
    assert tok.vocab_size < 65536, (
        f"tokenizer vocab_size {tok.vocab_size} doesn't fit in uint16 -- "
        f"shards would silently wrap around and corrupt token ids"
    )
    # StarCoder2's config.json/generation_config.json ship a leftover GPT-2
    # bos/eos_token_id (50256) that doesn't fit its own 49,152-token
    # vocabulary -- a known upstream bug (see bigcode/starcoder2-15b
    # discussion #14). tok.eos_token_id itself resolves correctly from the
    # tokenizer's own special_tokens_map.json, not the buggy config field,
    # but this assertion exists so ANY tokenizer with a similarly
    # inconsistent config fails loudly here rather than writing an
    # out-of-vocabulary id into shards that only surfaces as an embedding
    # index-out-of-range crash hours into real training.
    assert 0 <= eos_id < tok.vocab_size, (
        f"{tokenizer_name}'s eos_token_id ({eos_id}) is outside its own "
        f"vocab_size ({tok.vocab_size}) -- refusing to use it as a "
        f"document separator. Pass an explicit --eos-token-id override "
        f"if you've confirmed a different, valid id to use instead."
    )

    stream = _open_stream(spec, seed, skip_docs=manifest["docs_consumed"],
                          shuffle_buffer=shuffle_buffer)

    buf: list[int] = []
    shard_idx = len(manifest["shards"])
    docs_this_run = 0
    empty_text_count = 0
    # tracks tokens accumulated THIS RUN, independent of manifest["total_tokens"]
    # (which only advances when a shard actually flushes to disk -- see the
    # bug this fixes: for a target smaller than shard_tokens, that value
    # would never move, and the loop below would never stop early)
    buf_total_tokens_seen = manifest["total_tokens"]
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
        if buf_total_tokens_seen >= target_tokens:
            break
        text = doc.get(spec.text_field)
        if not text:
            docs_this_run += 1
            empty_text_count += 1
            # Fail fast rather than silently burning bandwidth: this is
            # exactly the failure mode that cost real time against
            # the-stack-v2-dedup, whose rows carry Software Heritage blob
            # IDs instead of content -- every row's text field was empty,
            # and nothing caught it until a download eventually crashed
            # tens of minutes in. If the first WINDOW documents are all
            # empty, the text_field is almost certainly wrong for this
            # dataset; stop immediately instead of continuing to download
            # multi-gigabyte files that will never produce a token.
            window = 50
            if docs_this_run >= window and empty_text_count == docs_this_run:
                raise RuntimeError(
                    f"the first {window} documents from {spec.hf_dataset} "
                    f"all had an empty '{spec.text_field}' field -- this "
                    f"almost certainly means text_field is wrong for this "
                    f"dataset (or it doesn't ship content directly at all, "
                    f"as with the-stack-v2-dedup). Check the dataset's "
                    f"actual schema before retrying; continuing would just "
                    f"download more files and find nothing."
                )
            continue
        ids = tok.encode(text, add_special_tokens=False)
        ids.append(eos_id)   # document boundary -- the model should never
                              # learn to treat two unrelated documents as
                              # one continuous stream
        buf.extend(ids)
        buf_total_tokens_seen += len(ids)
        docs_this_run += 1
        manifest["docs_consumed"] += 1

        while len(buf) >= shard_tokens:
            flush_shard()

    # Final partial shard: written and counted, but genuinely partial --
    # the loader treats every shard in the manifest identically regardless
    # of size, so this is safe, just slightly smaller than the rest.
    # Unconditional: docs_consumed was already advanced for every document
    # whose tokens are sitting in buf, so leaving them unflushed here would
    # silently lose real, already-counted data.
    if buf:
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
    p.add_argument("--shuffle-buffer", type=int, default=10_000,
                  help="documents buffered before the stream starts "
                       "yielding. Use a small value (e.g. 100) for a fast "
                       "--target-tokens dry run; leave at the default for "
                       "a real multi-billion-token pull")
    p.add_argument("--eos-token-id", type=int, default=None,
                  help="override the document-separator token id, in case "
                       "a tokenizer's own eos_token_id doesn't fit its "
                       "vocab (see the StarCoder2 config bug this file "
                       "guards against)")
    args = p.parse_args()

    tokenize_domain(args.domain, args.target_tokens, args.out_dir,
                    args.tokenizer, args.shard_tokens, args.seed,
                    args.shuffle_buffer, args.eos_token_id)


if __name__ == "__main__":
    main()
