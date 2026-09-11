"""Spectral normalisation used by the Matting Anything decoder.

Vendored from Matting Anything (https://github.com/SHI-Labs/Matting-Anything),
Copyright (c) 2023 SHI Labs, MIT License, which in turn adapts
https://github.com/heykeetae/Self-Attention-GAN.

The implementation is kept byte-for-byte compatible with the upstream module so
that published Matting Anything checkpoints load without any key remapping.
"""

import torch
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    """
    Normalises a vector to unit L2 norm.

    Args:
        v (torch.Tensor): The vector to normalise.
        eps (float, optional): Small constant guarding against division by zero.
                               Default is 1e-12.

    Returns:
        torch.Tensor: The normalised vector.
    """
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """
    Wraps a module so its weight matrix is divided by its largest singular value.

    The wrapper stores three buffers (`weight_u`, `weight_v`, `weight_bar`) rather than a
    plain `weight`, which is why Matting Anything checkpoints contain those key names.
    During evaluation the power iteration is skipped so results stay deterministic.

    Attributes:
        module (nn.Module): The wrapped module whose weight is normalised.
        name (str): Name of the weight attribute being normalised.
        power_iterations (int): Number of power iterations used during training.
    """

    def __init__(self, module, name="weight", power_iterations=1):
        """
        Initialises the spectral norm wrapper.

        Args:
            module (nn.Module): The module to wrap, typically a convolution.
            name (str, optional): The weight attribute to normalise. Default is `"weight"`.
            power_iterations (int, optional): Power iterations per training step. Default is 1.
        """
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(
                torch.mv(torch.t(w.view(height, -1).data), u.data)
            )
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        """
        Applies the wrapped module with a spectrally normalised weight.

        Args:
            *args: Positional arguments forwarded to the wrapped module.

        Returns:
            torch.Tensor: The wrapped module's output.
        """
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


__all__ = ["SpectralNorm", "l2normalize"]
