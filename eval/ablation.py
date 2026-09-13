"""
Ablation for experts means that for tokens routed to a particular expert, skip the MoE branch entirely rather than zeroing it's weights or substituting another expert's output.

For every (ablated expert, evaluated domain) pair, prints the held-out
loss with that expert disabled versus the real model's loss on that same
domain. The diagonal of that grid -- an expert's effect on its OWN
domain -- is what Section 10 is actually asking about; the off-diagonal
entries are included too, since a genuinely useful ablation should also
show whether removing (say) the math expert quietly hurts code loss,
which would suggest some cross-domain leakage the hard-routing design
isn't supposed to allow.
 
"Ablating" an expert here means: for tokens routed to that expert, skip
the MoE branch entirely rather than zeroing its weights or substituting
another expert's output. This tests the sharpest, simplest question --
"how much does this expert's FFN contribute over doing nothing" -- and
is a direct manipulation of DemixMoE's forward pass, not a separate
model variant that needs its own checkpoint.

"""
import argparse
import logging
import contextlib as contextmanager
from pathlib import Path
import torch

from eval.common import CONFIGS, held_out_shard_paths, iter_eval_windows, load_model

from eval.held_out_loss import domain_loss

log = logging.getLogger(__name__)

@contextmanager
def ablate_expert():

def main():
    
