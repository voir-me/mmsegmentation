import torch
from torch import nn
from mmcv.cnn import DepthwiseSeparableConvModule, ConvModule
from ..builder import HEADS
from .fcn_head import BaseDecodeHead


@HEADS.register_module()
class DepthwiseSeparableFCNHead(BaseDecodeHead):
    """Depthwise-Separable Fully Convolutional Network for Semantic
    Segmentation.

    This head is implemented according to Fast-SCNN paper.
    Args:
        in_channels(int): Number of output channels of FFM.
        channels(int): Number of middle-stage channels in the decode head.
        concat_input(bool): Whether to concatenate original decode input into
            the result of several consecutive convolution layers.
            Default: True.
        num_classes(int): Used to determine the dimension of
            final prediction tensor.
        in_index(int): Correspond with 'out_indices' in FastSCNN backbone.
        norm_cfg (dict | None): Config of norm layers.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
        loss_decode(dict): Config of loss type and some
            relevant additional options.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 **kwargs):
        assert num_convs >= 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super(DepthwiseSeparableFCNHead, self).__init__(**kwargs)

        if num_convs == 0:
            assert self.in_channels == self.channels
            self.convs = nn.Identity()
        else:
            convs = [DepthwiseSeparableConvModule(
                        self.in_channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        norm_cfg=self.norm_cfg)]
            convs += [DepthwiseSeparableConvModule(
                        self.channels,
                        self.channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        norm_cfg=self.norm_cfg)
                for _ in range(num_convs - 1)]
            self.convs = nn.Sequential(*convs)

        if self.concat_input:
            self.conv_cat = DepthwiseSeparableConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm_cfg=self.norm_cfg)


    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        output = self.cls_seg(output)
        return output