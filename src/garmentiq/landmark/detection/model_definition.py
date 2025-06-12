import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """
    Creates a 3x3 convolutional layer with padding.

    Args:
        in_planes (int): Number of input channels.
        out_planes (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.

    Returns:
        nn.Conv2d: 3x3 convolution layer with specified parameters.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    """
    Basic residual block with two 3x3 convolutional layers.

    Attributes:
        expansion (int): Expansion factor for output channels (always 1 for BasicBlock).
        conv1 (nn.Conv2d): First convolutional layer.
        bn1 (nn.BatchNorm2d): Batch normalization after first conv.
        conv2 (nn.Conv2d): Second convolutional layer.
        bn2 (nn.BatchNorm2d): Batch normalization after second conv.
        downsample (nn.Module or None): Optional downsampling layer for residual connection.
        stride (int): Stride of the first convolution.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes BasicBlock.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels.
            stride (int, optional): Stride of the first convolution. Defaults to 1.
            downsample (nn.Module or None, optional): Downsampling layer for residual. Defaults to None.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the BasicBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual addition and activation.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck residual block with 1x1, 3x3, and 1x1 convolutions.

    Attributes:
        expansion (int): Expansion factor for output channels (usually 4 for Bottleneck).
        conv1 (nn.Conv2d): 1x1 convolution reducing channels.
        bn1 (nn.BatchNorm2d): Batch normalization after conv1.
        conv2 (nn.Conv2d): 3x3 convolution.
        bn2 (nn.BatchNorm2d): Batch normalization after conv2.
        conv3 (nn.Conv2d): 1x1 convolution expanding channels.
        bn3 (nn.BatchNorm2d): Batch normalization after conv3.
        downsample (nn.Module or None): Optional downsampling layer for residual connection.
        stride (int): Stride for the 3x3 convolution.
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        """
        Initializes Bottleneck block.

        Args:
            inplanes (int): Number of input channels.
            planes (int): Number of output channels before expansion.
            stride (int, optional): Stride for the 3x3 convolution. Defaults to 1.
            downsample (nn.Module or None, optional): Downsampling layer for residual. Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after residual addition and activation.
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class HighResolutionModule(nn.Module):
    """
    HighResolutionModule maintains high-resolution representations through multi-branch architecture and fusion.

    This module consists of several parallel branches with residual blocks, and fuse layers to combine
    features from different resolutions.

    Attributes:
        num_branches (int): Number of parallel branches.
        blocks (nn.Module): Residual block class (BasicBlock or Bottleneck).
        num_blocks (list[int]): Number of residual blocks per branch.
        num_inchannels (list[int]): Number of input channels for each branch.
        num_channels (list[int]): Number of channels per branch before expansion.
        fuse_method (str): Method to fuse multi-branch outputs ('SUM' supported).
        multi_scale_output (bool): Whether to output multi-scale features.
        branches (nn.ModuleList): The parallel branches.
        fuse_layers (nn.ModuleList or None): Layers that fuse features from branches.
    """

    def __init__(
        self,
        num_branches,
        blocks,
        num_blocks,
        num_inchannels,
        num_channels,
        fuse_method,
        multi_scale_output=True,
    ):
        """
        Initializes HighResolutionModule.

        Args:
            num_branches (int): Number of parallel branches.
            blocks (nn.Module): Residual block class (BasicBlock or Bottleneck).
            num_blocks (list[int]): Number of residual blocks per branch.
            num_inchannels (list[int]): Number of input channels for each branch.
            num_channels (list[int]): Number of channels per branch before expansion.
            fuse_method (str): Method to fuse multi-branch outputs.
            multi_scale_output (bool, optional): Output multi-scale features or not. Defaults to True.

        Raises:
            ValueError: If lengths of inputs do not match num_branches.
        """
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels
        )

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels
        )
        self.fuse_layers = self._make_fuse_layers()

    def _check_branches(
        self, num_branches, blocks, num_blocks, num_inchannels, num_channels
    ):
        """
        Validates that lengths of num_blocks, num_inchannels, and num_channels match num_branches.

        Args:
            num_branches (int): Number of branches.
            blocks (nn.Module): Block type.
            num_blocks (list[int]): Number of blocks per branch.
            num_inchannels (list[int]): Number of input channels per branch.
            num_channels (list[int]): Number of channels per branch.

        Raises:
            ValueError: If any length mismatch occurs.
        """
        if num_branches != len(num_blocks):
            error_msg = "NUM_BRANCHES({}) <> NUM_BLOCKS({})".format(
                num_branches, len(num_blocks)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = "NUM_BRANCHES({}) <> NUM_CHANNELS({})".format(
                num_branches, len(num_channels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = "NUM_BRANCHES({}) <> NUM_INCHANNELS({})".format(
                num_branches, len(num_inchannels)
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        """
        Constructs one branch of the module consisting of sequential residual blocks.

        Args:
            branch_index (int): Index of the branch.
            block (nn.Module): Residual block class.
            num_blocks (int): Number of residual blocks in this branch.
            num_channels (list[int]): Number of channels per branch.
            stride (int, optional): Stride of the first block. Defaults to 1.

        Returns:
            nn.Sequential: Sequential container of residual blocks.
        """
        downsample = None
        if (
            stride != 1
            or self.num_inchannels[branch_index]
            != num_channels[branch_index] * block.expansion
        ):
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample,
            )
        )
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(self.num_inchannels[branch_index], num_channels[branch_index])
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """
        Constructs all branches for the module.

        Args:
            num_branches (int): Number of branches.
            block (nn.Module): Residual block class.
            num_blocks (list[int]): Number of blocks per branch.
            num_channels (list[int]): Number of channels per branch.

        Returns:
            nn.ModuleList: List of branch modules.
        """
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """
        Constructs layers to fuse multi-resolution branch outputs.

        Returns:
            nn.ModuleList or None: Fuse layers or None if single branch.
        """
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1,
                                1,
                                0,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3,
                                        2,
                                        1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """
        Returns the number of input channels for each branch after block expansion.

        Returns:
            list[int]: Number of input channels per branch.
        """
        return self.num_inchannels

    def forward(self, x):
        """
        Forward pass through the HighResolutionModule.

        Args:
            x (list[torch.Tensor]): List of input tensors for each branch.

        Returns:
            list[torch.Tensor]: List of output tensors after multi-branch fusion.
        """
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(F.relu(y, inplace=True))

        return x_fuse


blocks_dict = {"BASIC": BasicBlock, "BOTTLENECK": Bottleneck}


class PoseHighResolutionNet(nn.Module):
    """
    High-Resolution Network (HRNet) tailored for garment landmark detection.

    The network maintains high-resolution representations through multiple stages and branches,
    fusing multi-scale features and finally predicting heatmaps or coordinates for landmarks.

    Attributes:
        inplanes (int): Initial number of input channels.
        conv1 (nn.Conv2d): Initial 3x3 convolution.
        bn1 (nn.BatchNorm2d): Batch normalization after conv1.
        conv2 (nn.Conv2d): Second 3x3 convolution.
        bn2 (nn.BatchNorm2d): Batch normalization after conv2.
        relu (nn.ReLU): ReLU activation.
        stage1_cfg (dict): Configuration for stage1.
        stage2_cfg (dict): Configuration for stage2.
        stage3_cfg (dict): Configuration for stage3.
        stage4_cfg (dict): Configuration for stage4.
        transition1 (nn.ModuleList): Transition layers between stages.
        stage2 (HighResolutionModule): Stage 2 module.
        transition2 (nn.ModuleList): Transition layers between stages.
        stage3 (HighResolutionModule): Stage 3 module.
        transition3 (nn.ModuleList): Transition layers between stages.
        stage4 (HighResolutionModule): Stage 4 module.
        final_layer (nn.Conv2d): Final convolution layer to output predictions.
        target_type (str): Output format type ('gaussian' or 'coordinate').
    """

    def __init__(self, **kwargs):
        """
        Initializes PoseHighResolutionNet with default HRNet configurations.

        Args:
            target_type (str, optional): Type of target output. Either "gaussian" or "coordinate". Defaults to "gaussian".
        """
        self.inplanes = 64
        # Hardcoded values from YAML MODEL.EXTRA
        extra = {
            "PRETRAINED_LAYERS": [
                "conv1",
                "bn1",
                "conv2",
                "bn2",
                "layer1",
                "transition1",
                "stage2",
                "transition2",
                "stage3",
                "transition3",
                "stage4",
            ],
            "FINAL_CONV_KERNEL": 1,
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [48, 96],
                "FUSE_METHOD": "SUM",
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192],
                "FUSE_METHOD": "SUM",
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": "BASIC",
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192, 384],
                "FUSE_METHOD": "SUM",
            },
        }

        self.model_name = "pose_hrnet"
        self.target_type = "gaussian"

        super(PoseHighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        # Stage2
        self.stage2_cfg = extra["STAGE2"]
        num_channels = self.stage2_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage2_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels
        )

        # Stage3
        self.stage3_cfg = extra["STAGE3"]
        num_channels = self.stage3_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage3_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels
        )

        # Stage4
        self.stage4_cfg = extra["STAGE4"]
        num_channels = self.stage4_cfg["NUM_CHANNELS"]
        block = blocks_dict[self.stage4_cfg["BLOCK"]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False
        )

        # Final layer
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=294,  # from MODEL.NUM_JOINTS
            kernel_size=extra["FINAL_CONV_KERNEL"],
            stride=1,
            padding=1 if extra["FINAL_CONV_KERNEL"] == 3 else 0,
        )

        self.pretrained_layers = extra["PRETRAINED_LAYERS"]

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
        Creates transition layers to match the number of channels between stages.

        Args:
            num_channels_pre_layer (list[int]): Channels from previous stage.
            num_channels_cur_layer (list[int]): Channels for current stage.

        Returns:
            nn.ModuleList: List of transition layers.
        """
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = (
                        num_channels_cur_layer[i]
                        if j == i - num_branches_pre
                        else inchannels
                    )
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Creates a layer composed of sequential residual blocks.

        Args:
            block (nn.Module): Residual block type (BasicBlock or Bottleneck).
            planes (int): Number of output channels.
            blocks (int): Number of blocks in this layer.
            stride (int, optional): Stride for the first block. Defaults to 1.

        Returns:
            nn.Sequential: Sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        """
        Constructs a stage consisting of one or more HighResolutionModules.

        Args:
            layer_config (dict): Configuration dictionary for the stage.
            num_inchannels (list[int]): Number of input channels for each branch.
            multi_scale_output (bool, optional): Output multi-scale features or not. Defaults to True.

        Returns:
            tuple:
                nn.Sequential: Stage module.
                list[int]: Number of output channels for each branch.
        """
        num_modules = layer_config["NUM_MODULES"]
        num_branches = layer_config["NUM_BRANCHES"]
        num_blocks = layer_config["NUM_BLOCKS"]
        num_channels = layer_config["NUM_CHANNELS"]
        block = blocks_dict[layer_config["BLOCK"]]
        fuse_method = layer_config["FUSE_METHOD"]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output,
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        """
        Forward pass of the PoseHighResolutionNet.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output heatmaps or coordinates for landmarks.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg["NUM_BRANCHES"]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg["NUM_BRANCHES"]):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg["NUM_BRANCHES"]):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        if self.model_name == "pose_hrnet" or "pose_metric_gcn":
            x = self.final_layer(y_list[0])
        else:
            x = y_list[0]

        if self.target_type == "gaussian":
            return x

        elif self.target_type == "coordinate":
            B, C, H, W = x.shape

            """B - cal x,y seperately"""
            h = F.softmax(x.view(B, C, H * W) * 1, dim=2)
            h = h.view(B, C, H, W)
            hx = h.sum(dim=2)  # (B, C, W)
            px = (hx * (torch.arange(W, device=h.device).float().view(1, 1, W))).sum(
                2, keepdim=True
            )
            hy = h.sum(dim=3)  # (B, C, H)
            py = (hy * (torch.arange(H, device=h.device).float().view(1, 1, H))).sum(
                2, keepdim=True
            )
            x = torch.cat([px, py], dim=2)
            return h, x
        else:
            raise NotImplementedError(f"{self.target_type} is unknown.")
