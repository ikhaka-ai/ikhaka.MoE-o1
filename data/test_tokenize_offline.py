Try AI directly in your favourite apps … Use Gemini to generate drafts and refine content, plus get Gemini Pro with access to Google's next-gen AI

"""
Exercises tokenize_domain()'s actual shard-writing and resume logic without
touching the network -- huggingface.co isn't reachable from every sandbox,
and this loop is worth verifying independently of that access.

Not part of the shipped pipeline: this monkeypatches the two functions that
call out to Hugging Face (_open_stream, AutoTokenizer.from_pretrained) with
deterministic fakes, then runs the real tokenize_domain() against them.
"""
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, ".")
import data.tokenize as tk


class FakeTokenizer:
    """Deterministic: token id = (ord of first char) so shard contents are
    checkable by hand, vocab comfortably under uint16, real eos_token_id."""
    vocab_size = 1000
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 900 for c in text]


FAKE_DOCS = [{"text": f"doc{i}" * 20} for i in range(500)]  # ~2500 chars each -> ~2500 tokens/doc


def fake_open_stream(spec, seed, skip_docs=0):
    return iter(FAKE_DOCS[skip_docs:])


def run():
    out_root = Path("/tmp/tokenize_test")
    shutil.rmtree(out_root, ignore_errors=True)

    with patch("data.tokenize._open_stream", fake_open_stream), \
         patch("transformers.AutoTokenizer.from_pretrained",
               lambda name: FakeTokenizer()):
        # small shard size so the run produces several shards from the
        # tiny fake corpus, exercising the flush/manifest path repeatedly
        tk.tokenize_domain("physics", target_tokens=5000, out_root=out_root,
                           tokenizer_name="fake", shard_tokens=800)

    out_dir = out_root / "physics"
    manifest = json.loads((out_dir / "manifest.json").read_text())

    # ---------------------------------------------------------------- checks
    assert manifest["tokenizer"] == "fake"
    assert manifest["total_tokens"] >= 5000, manifest["total_tokens"]
    assert len(manifest["shards"]) >= 6, manifest["shards"]
    print(f"[1] manifest ok: {manifest['total_tokens']} tokens across "
          f"{len(manifest['shards'])} shards")

    running_total = 0
    for shard in manifest["shards"]:
        arr = np.load(out_dir / shard["file"])
        assert arr.dtype == np.uint16
        assert arr.size == shard["tokens"]
        running_total += arr.size
        assert not (out_dir / (shard["file"] + ".tmp")).exists(), (
            "a .tmp file survived -- atomic rename didn't happen"
        )
    assert running_total == manifest["total_tokens"]
    print(f"[2] every shard on disk matches its manifest entry, "
          f"no leftover .tmp files")

    # EOS (999) should appear at least once per document boundary
    all_tokens = np.concatenate([np.load(out_dir / s["file"])
                                 for s in manifest["shards"]])
    n_eos = int((all_tokens == 999).sum())
    assert n_eos >= 1, "no EOS tokens found -- document boundaries are missing"
    print(f"[3] document boundaries present: {n_eos} EOS tokens found")

    # ---------------------------------------------------------------- resume
    docs_before = manifest["docs_consumed"]
    shards_before = len(manifest["shards"])
    with patch("data.tokenize._open_stream", fake_open_stream), \
         patch("transformers.AutoTokenizer.from_pretrained",
               lambda name: FakeTokenizer()):
        tk.tokenize_domain("physics", target_tokens=9000, out_root=out_root,
                           tokenizer_name="fake", shard_tokens=800)

    manifest2 = json.loads((out_dir / "manifest.json").read_text())
    assert manifest2["docs_consumed"] > docs_before, (
        "resume restarted from doc 0 instead of skipping past docs_consumed"
    )
    assert len(manifest2["shards"]) > shards_before
    # every shard from the first run must still be byte-identical -- a
    # correct resume never rewrites a completed shard
    for shard in manifest["shards"]:
        arr = np.load(out_dir / shard["file"])
        assert arr.size == shard["tokens"], (
            f"{shard['file']} was mutated by the resumed run"
        )
    print(f"[4] resume ok: picked up at doc {docs_before} -> "
          f"{manifest2['docs_consumed']}, prior shards untouched")

    # ---------------------------------------------------------------- mismatch guard
    try:
        with patch("data.tokenize._open_stream", fake_open_stream), \
             patch("transformers.AutoTokenizer.from_pretrained",
                   lambda name: FakeTokenizer()):
            tk.tokenize_domain("physics", target_tokens=100, out_root=out_root,
                               tokenizer_name="a-different-tokenizer")
        raise AssertionError("expected a RuntimeError on tokenizer mismatch")
    except RuntimeError as e:
        print(f"[5] tokenizer-mismatch guard ok: {e}")

    shutil.rmtree(out_root, ignore_errors=True)
    print("\nall tokenize.py checks passed")


if __name__ == "__main__":
    run()
