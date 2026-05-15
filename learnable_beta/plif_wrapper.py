"""
plif_wrapper.py — Parametric LIF node for learnable per-layer beta
===================================================================
Implements Option 2 from notes_v2.txt §3.2 (manual softplus subclass).

Option 1 (spikingjelly.activation_based.neuron.ParametricLIFNode) was
skipped because:
  - spikingjelly version 0.0.0.0.12 does not have activation_based
  - clock_driven.ParametricLIFNode, if present, is single-step and
    incompatible with the MultiStep pattern used throughout the codebase

Option 2 is equally valid and completely transparent for the report.
Reparameterisation: tau = 1 + softplus(_raw_tau), guaranteeing tau > 1
(beta > 0) throughout training. One scalar _raw_tau per LIF layer.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class ParametricMultiStepLIFNode(MultiStepLIFNode):
    """
    Drop-in replacement for MultiStepLIFNode with one learnable tau per layer.

    tau  = 1 + softplus(_raw_tau)   →  tau in (1, inf)
    beta = 1 - 1/tau                →  beta in (0, 1)

    Initialise at init_tau (default = codebase default = 2.0).
    Extra parameters per layer: exactly 1 scalar (_raw_tau).
    """

    def __init__(self, init_tau: float = 2.0, **kwargs):
        kwargs.pop('tau', None)
        super().__init__(tau=float(init_tau), **kwargs)

        offset = max(float(init_tau) - 1.0, 1e-6)
        self._raw_tau = nn.Parameter(torch.tensor(math.log(math.expm1(offset))))

        if 'tau' in self.__dict__:
            del self.__dict__['tau']

    @property
    def tau(self) -> torch.Tensor:
        return 1.0 + F.softplus(self._raw_tau)

    @tau.setter
    def tau(self, value):
        pass

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 - 1.0 / self.tau

    def extra_repr(self) -> str:
        t = self.tau.item()
        return f'tau={t:.4f}(learnable) beta={1-1/t:.4f} v_threshold={self.v_threshold}'

def count_extra_params(model: nn.Module) -> int:
    """Count learnable _raw_tau scalars (one per PLIF layer)."""
    return sum(1 for m in model.modules()
               if isinstance(m, ParametricMultiStepLIFNode))

def collect_betas(model: nn.Module) -> list:
    """Return list of {layer_index, layer_name, beta, tau} dicts."""
    out, idx = [], 0
    for name, m in model.named_modules():
        if isinstance(m, ParametricMultiStepLIFNode):
            with torch.no_grad():
                t = m.tau.item()
            out.append({'layer_index': idx, 'layer_name': name,
                        'tau': t, 'beta': 1.0 - 1.0 / t})
            idx += 1
    return out
