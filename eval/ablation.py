"""
Ablation for experts means that for tokens routed to a particular expert, skip the MoE branch entirely rather than zeroing it's weights or substituting another expert's output.
"""
import argparse
import logging
import contextlib as contextmanager
from pathlib import Path
