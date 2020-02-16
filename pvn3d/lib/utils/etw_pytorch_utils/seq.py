from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch.nn as nn
from .pytorch_utils import (
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    Conv1d,
    Conv2d,
    Conv3d,
    FC,
)


if False:
    # Workaround for type hints without depending on the `typing` module
    from typing import *


class Seq(nn.Sequential):
    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(
        self,
        out_size,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm1d,
    ):
        # type: (Seq, int, int, int, int, int, Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(
        self,
        out_size,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm2d,
    ):
        # type: (Seq, int, Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int], Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def conv3d(
        self,
        out_size,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=nn.init.kaiming_normal_,
        bias=True,
        preact=False,
        name="",
        norm_layer=BatchNorm3d,
    ):
        # type: (Seq, int, Tuple[int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Any, bool, Any, bool, bool, AnyStr) -> Seq

        self.add_module(
            str(self.count),
            Conv3d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name,
                norm_layer=norm_layer,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(
        self,
        out_size,
        activation=nn.ReLU(inplace=True),
        bn=False,
        init=None,
        preact=False,
        name="",
    ):
        # type: (Seq, int, Any, bool, Any, bool, AnyStr) -> None

        self.add_module(
            str(self.count),
            FC(
                self.current_channels,
                out_size,
                activation=activation,
                bn=bn,
                init=init,
                preact=preact,
                name=name,
            ),
        )
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):
        # type: (Seq, float) -> Seq

        self.add_module(str(self.count), nn.Dropout(p=0.5))
        self.count += 1

        return self

    def maxpool2d(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode,
            ),
        )
        self.count += 1

        return self
