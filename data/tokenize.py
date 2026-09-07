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

