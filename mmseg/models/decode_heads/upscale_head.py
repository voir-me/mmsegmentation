import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmseg.ops import resize

from ..builder import HEADS
from .fcn_head import BaseDecodeHead

""" RestorerModule is used in stages for upscaling
    General options:
        - skip connection could be set for each stage
        - skip connection operations 'add' and 'concat' is supported
        - 'add' operation will be optimized by removing one point-wise conv if number of channels are equal
        - channels param for stage is used to decrease or increase number of channel on each stage
        - if channels param is skipped, general head channels param will be used.
            If this is the case, number of channels will be the same across all stages. 

    Args:
        channels(int): should be set in config if increasing or decreasing of channels 
            is needed on this stage. Otherwise will be equal to head channels param
        out_channels(int): set by head logic depending on stages params.
            if channels for stages is not set, then will be equal to head channels
        upscale(dict): parameters for mmseg.ops.resize operation
            scale_factor(int): scale factor for scale operation
            mode='nearest': mode for scale operation
            align_corners=None
        conv(dict): parameters for mmseg conv operation 
        skip(dict): parameters for skip connection logic
            op='add'|'concat': operation for skip feature maps 
            in_index(int): index of output from previous level
            in_channels(int): number of channel of output set by in_index
        num_convs(int): number of sequential conv in this block
                        
"""
class RestorerModule(nn.Module):

    def __init__(self,
                 channels,
                 out_channels,
                 upscale,
                 conv,
                 skip=None,
                 num_convs=1,
                 kernel_size=3,
                 **kwargs):
        self.upscale = upscale
        assert num_convs > 0
        self.conv = conv
        self.skip = skip
        self.num_convs = num_convs
        if skip:
            assert skip.op
            assert skip.in_index
            assert skip.op in ['concat', 'add']

        super(RestorerModule, self).__init__(**kwargs)

        convs = [DepthwiseSeparableConvModule(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            **conv)
            for _ in range(num_convs-1)]
        convs.append(
            DepthwiseSeparableConvModule(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **conv))
        self.convs = nn.Sequential(*convs)

        if skip:
            assert skip.op
            assert skip.in_index
            if skip.op == 'concat':
                self.conv_skip = DepthwiseSeparableConvModule(
                    in_channels=channels + skip.in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    **conv)
            elif skip.op == 'add':
                if skip.in_channels == channels:
                    self.conv_skip = None  # optimizing in case of same channels
                else:
                    self.conv_skip = ConvModule(
                        skip.in_channels,
                        channels,
                        1,
                        **conv)
            else:
                raise ValueError(f'Operation {skip.op} is not supported')
        else:
            self.conv_skip = None


    def forward(self, input, skip_input=None):
        """Forward function."""
        output = resize(input,
                        scale_factor=self.upscale.scale_factor,
                        mode=self.upscale.mode,
                        align_corners=self.upscale.align_corners)
        if self.skip:
            assert skip_input is not None
            if self.skip.op == 'concat':
                assert output.shape[2] == skip_input.shape[2], 'height should be equal'
                assert output.shape[3] == skip_input.shape[3], 'width should be equal'
                #print('output.shape', output.shape)
                #print('skip_input.shape', skip_input.shape)
                output = torch.cat([output, skip_input], dim=1)
                output = self.conv_skip(output)

            elif self.skip.op == 'add':
                if self.conv_skip:
                    skip_output = self.conv_skip(skip_input)
                else:
                    skip_output = skip_input
                #print('skip_input.shape', skip_input.shape)
                #print('output.shape', output.shape)
                #print('skip_output.shape', skip_output.shape)
                assert output.shape[1] == skip_output.shape[1], 'number of channels should be equal'
                output = output + skip_output
            else:
                raise ValueError(f'Operation {self.skip.op} is not supported')

        output = self.convs(output)
        return output

@HEADS.register_module()
class UpscaleHead(BaseDecodeHead):
    """Decode head with Depthwise-Separable convs and upscale stages for Semantic
    Segmentation.


    This head implementation base on Fast-SCNN paper and ESRGan upscale approach.
    Args:
        in_channels(int): Number of output channels of FFM.
        channels(int): Number of middle-stage channels in the decode head.
        stages(dict): configuration of upscale steps. See RestorerModule.
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
                 in_channels,
                 channels,
                 stages,
                 kernel_size=3,
                 concat_input=False,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(UpscaleHead, self).__init__(
            in_channels=in_channels,
            channels=channels,
            **kwargs)
        self.concat_input = concat_input

        first_out_channels = channels
        if stages and len(stages) > 0:
            if stages[0].channels:
                first_out_channels = stages[0].channels
            for stage in stages:
                # propagate channel from head config
                if not stage.channels:
                    stage.channels = channels

        self.input_conv = DepthwiseSeparableConvModule(
            in_channels=in_channels,
            out_channels=first_out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        restorer_stages = []
        stages_out = stages[1:] + [None]
        for stage_in, stage_out in zip(stages, stages_out):
            restorer_stages.append(
                RestorerModule(
                    out_channels=stage_out.channels if stage_out else channels,
                    **stage_in)
            )
        self.restorer_stages=nn.Sequential(*restorer_stages)

    def forward(self, inputs):
        """Forward function."""
        output = self._transform_inputs(inputs)
        output = self.input_conv(output)
        for stage in self.restorer_stages:
            output = stage(output,
                           inputs[stage.skip.in_index] if stage.skip else None)
        output = self.cls_seg(output)
        return output