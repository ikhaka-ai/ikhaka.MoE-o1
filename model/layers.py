#shared trunk. Every layer in this file runs identically regardless of the domain
#a token belongs to. This enables the four experts to work together later on.
#They all read and write to the same residual stream shaped by the same attention.

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MoEConfig
