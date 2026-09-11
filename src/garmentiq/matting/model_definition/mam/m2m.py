"""Mask-to-matte (M2M) decoder used by Matting Anything.

Vendored from Matting Anything (https://github.com/SHI-Labs/Matting-Anything),
Copyright (c) 2023 SHI Labs, MIT License, which in turn adapts MGMatting
(https://github.com/yucornetto/MGMatting).

The layer names and shapes are kept identical to upstream so that published
Matting Anything checkpoints load without key remapping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from garmentiq.matting.model_definition.mam import ops


def conv5x5(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """5x5 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=5,
        stride=stride,
        padding=2,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """
    Residual upsampling block used throughout the M2M decoder.

    When `stride > 1` the block upsamples with a transposed convolution, otherwise it
    applies a plain convolution. Both branches are spectrally normalised.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        upsample=None,
        norm_layer=None,
        large_kernel=False,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv5x5 if large_kernel else conv3x3
        if self.stride > 1:
            self.conv1 = ops.SpectralNorm(
                nn.ConvTranspose2d(
                    inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False
                )
            )
        else:
            self.conv1 = ops.SpectralNorm(conv(inplanes, inplanes))
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = ops.SpectralNorm(conv(inplanes, planes))
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out


class SAM_Decoder_Deep(nn.Module):
    """
    Progressively upsamples SAM image embeddings into an alpha matte.

    At each scale the decoder re-injects the source image and the coarse guidance mask,
    so the matte stays anchored to the original detail rather than drifting from the
    low-resolution embedding. Alpha is predicted at three output strides (1, 4 and 8)
    which the caller fuses.

    Attributes:
        inplanes (int): Channel count of the incoming SAM embedding (256).
        midplanes (int): Channel count feeding the final upsampling stage.
    """

    def __init__(
        self,
        nc,
        layers,
        block=BasicBlock,
        norm_layer=None,
        large_kernel=False,
        late_downsample=False,
    ):
        """
        Builds the decoder.

        Args:
            nc (int): Channel count of the SAM embedding fed to the decoder (256).
            layers (list[int]): Number of residual blocks per decoder stage.
            block (nn.Module, optional): Residual block class. Default is `BasicBlock`.
            norm_layer (nn.Module, optional): Normalisation layer. Default is `nn.BatchNorm2d`.
            large_kernel (bool, optional): Use 5x5 instead of 3x3 convolutions. Default is False.
            late_downsample (bool, optional): Widen the final stage. Default is False.
        """
        super(SAM_Decoder_Deep, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.large_kernel = large_kernel
        self.kernel_size = 5 if self.large_kernel else 3

        self.inplanes = 256
        self.late_downsample = late_downsample
        self.midplanes = 64 if late_downsample else 32

        self.conv1 = ops.SpectralNorm(
            nn.ConvTranspose2d(
                self.midplanes, 32, kernel_size=4, stride=2, padding=1, bias=False
            )
        )
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.midplanes, layers[3], stride=2)

        self.refine_OS1 = nn.Sequential(
            nn.Conv2d(
                32,
                32,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(
                32,
                1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
        )

        self.refine_OS4 = nn.Sequential(
            nn.Conv2d(
                64,
                32,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(
                32,
                1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
        )

        self.refine_OS8 = nn.Sequential(
            nn.Conv2d(
                128,
                32,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                bias=False,
            ),
            norm_layer(32),
            self.leaky_relu,
            nn.Conv2d(
                32,
                1,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
            ),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialise the last BN in each residual branch so the branch starts
        # as an identity, per https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                ops.SpectralNorm(conv1x1(self.inplanes + 4, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                ops.SpectralNorm(conv1x1(self.inplanes + 4, planes * block.expansion)),
                norm_layer(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes + 4,
                planes,
                stride,
                upsample,
                norm_layer,
                self.large_kernel,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    norm_layer=norm_layer,
                    large_kernel=self.large_kernel,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x_os16, img, mask):
        """
        Decodes SAM embeddings into alpha predictions at three output strides.

        Args:
            x_os16 (torch.Tensor): SAM image embedding, shape `(N, 256, H/16, W/16)`.
            img (torch.Tensor): Preprocessed image tensor, shape `(N, 3, H, W)`.
            mask (torch.Tensor): Coarse binary guidance mask, shape `(N, 1, h, w)`.

        Returns:
            dict: Keys `alpha_os1`, `alpha_os4`, `alpha_os8` (each in `[0, 1]`) and the
                  guidance `mask` resampled to full resolution.
        """
        ret = {}
        mask_os16 = F.interpolate(
            mask, x_os16.shape[2:], mode="bilinear", align_corners=False
        )
        img_os16 = F.interpolate(
            img, x_os16.shape[2:], mode="bilinear", align_corners=False
        )

        x = self.layer2(torch.cat((x_os16, img_os16, mask_os16), dim=1))

        x_os8 = self.refine_OS8(x)

        mask_os8 = F.interpolate(
            mask, x.shape[2:], mode="bilinear", align_corners=False
        )
        img_os8 = F.interpolate(img, x.shape[2:], mode="bilinear", align_corners=False)

        x = self.layer3(torch.cat((x, img_os8, mask_os8), dim=1))

        x_os4 = self.refine_OS4(x)

        mask_os4 = F.interpolate(
            mask, x.shape[2:], mode="bilinear", align_corners=False
        )
        img_os4 = F.interpolate(img, x.shape[2:], mode="bilinear", align_corners=False)

        x = self.layer4(torch.cat((x, img_os4, mask_os4), dim=1))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x_os1 = self.refine_OS1(x)

        x_os4 = F.interpolate(x_os4, scale_factor=4.0, mode="bilinear", align_corners=False)
        x_os8 = F.interpolate(x_os8, scale_factor=8.0, mode="bilinear", align_corners=False)

        x_os1 = (torch.tanh(x_os1) + 1.0) / 2.0
        x_os4 = (torch.tanh(x_os4) + 1.0) / 2.0
        x_os8 = (torch.tanh(x_os8) + 1.0) / 2.0

        mask_os1 = F.interpolate(
            mask, x_os1.shape[2:], mode="bilinear", align_corners=False
        )

        ret["alpha_os1"] = x_os1
        ret["alpha_os4"] = x_os4
        ret["alpha_os8"] = x_os8
        ret["mask"] = mask_os1

        return ret


def sam_decoder_deep(nc=256, **kwargs):
    """
    Builds the M2M decoder configuration used by the published Matting Anything weights.

    Args:
        nc (int, optional): SAM embedding channel count. Default is 256.
        **kwargs: Extra arguments forwarded to `SAM_Decoder_Deep`.

    Returns:
        SAM_Decoder_Deep: The constructed decoder.
    """
    return SAM_Decoder_Deep(nc, [2, 3, 3, 2], **kwargs)


__all__ = ["SAM_Decoder_Deep", "BasicBlock", "sam_decoder_deep"]
