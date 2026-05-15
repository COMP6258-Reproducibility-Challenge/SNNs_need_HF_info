"""
plif_nodes.py — Parametric LIF node with learnable membrane time constant
==========================================================================
Used exclusively by train_plif.py (Experiment B).

Design choices (document for the report):
------------------------------------------
1. We subclass MultiStepLIFNode from spikingjelly.clock_driven (the old API
   used throughout the MaxFormer codebase) rather than ParametricLIFNode from
   spikingjelly.activation_based (new API).  The two APIs are not interoperable
   because activation_based uses a step-mode context manager while clock_driven
   uses explicit T-step sequences.

2. tau is reparameterised as:
       tau = 1.0 + softplus(_raw_tau)
   This enforces tau > 1 (equivalently beta > 0) throughout training.
   Initialisation: _raw_tau is set so that tau(0) = init_tau exactly.

3. One scalar _raw_tau per LIF layer (NOT per channel/neuron).  This introduces
   exactly one extra float per LIF layer.  The helper count_extra_params()
   counts them for the per-run JSON.

4. The surrogate gradient in SpikingJelly is applied at the spiking threshold,
   not at tau.  Making tau a Parameter only affects the membrane-update equation.
   Gradients flow through tau via the membrane voltage, which in turn affects
   future spike probabilities.

Note on prior art: Learnable membrane time constants have appeared as PLIF
(Fang et al., 2021, "Incorporating Learnable Membrane Time Constants") and GLIF
(Yao et al., 2022) in the SNN literature.  This implementation is NOT a novel
contribution — it is a probe of the Max-Former paper's theory.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepLIFNode

class ParametricMultiStepLIFNode(MultiStepLIFNode):
    """
    MultiStepLIFNode with a learnable tau parameter per layer.

    The membrane update equation (standard LIF in SpikingJelly clock_driven):
        V[n] = V[n-1] * (1 - 1/tau) + x[n]    (decay_input=False form)
    or
        V[n] = V[n-1] * (1 - 1/tau) + x[n]/tau (decay_input=True form)

    tau is computed as 1 + softplus(_raw_tau), guaranteeing tau in (1, inf)
    and beta = 1 - 1/tau in (0, 1).
    """

    def __init__(self, init_tau: float = 2.0, **kwargs):
        kwargs.pop('tau', None)
        super().__init__(tau=float(init_tau), **kwargs)

        offset = max(float(init_tau) - 1.0, 1e-6)
        raw_init = math.log(math.expm1(offset))
        self._raw_tau = nn.Parameter(torch.tensor(raw_init))

        if 'tau' in self.__dict__:
            del self.__dict__['tau']

    @property
    def tau(self) -> torch.Tensor:
        """Effective tau as a differentiable tensor: tau = 1 + softplus(_raw_tau)."""
        return 1.0 + F.softplus(self._raw_tau)

    @tau.setter
    def tau(self, value):
        pass

    @property
    def beta(self) -> torch.Tensor:
        """Effective decay factor beta = 1 - 1/tau in (0, 1)."""
        return 1.0 - 1.0 / self.tau

    def extra_repr(self) -> str:
        tau_val = self.tau.item()
        return (f'tau={tau_val:.4f} (learnable), '
                f'beta={1 - 1/tau_val:.4f}, '
                f'v_threshold={self.v_threshold}')

def count_extra_params(model: nn.Module) -> int:
    """Count the number of learnable _raw_tau scalars in the model."""
    return sum(
        1
        for m in model.modules()
        if isinstance(m, ParametricMultiStepLIFNode)
    )

def collect_layer_betas(model: nn.Module) -> list:
    """
    Return a list of dicts with per-layer beta values.
    Each dict has: layer_index, layer_name, beta (float), tau (float).
    """
    results = []
    idx = 0
    for name, m in model.named_modules():
        if isinstance(m, ParametricMultiStepLIFNode):
            with torch.no_grad():
                tau_val = m.tau.item()
                beta_val = m.beta.item()
            results.append({
                'layer_index': idx,
                'layer_name': name,
                'tau': tau_val,
                'beta': beta_val,
            })
            idx += 1
    return results
