"""
Ablation for experts means that for tokens routed to a particular expert, skip the MoE branch entirely rather than zeroing it's weights or substituting another expert's output.
"""
import argparse
import logging
import contextlib as contextmanager
from pathlib import Path
import torch

from eval.common import CONFIGS, held_out_shard_paths, iter_eval_windows, load_model

from eval.held_out_loss import domain_loss
