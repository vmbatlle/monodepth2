# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from typing import List
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        convs = []
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            convs.append(ConvBlock(num_ch_in, num_ch_out))

        for s in self.scales:
            convs.append(Conv3x3(self.num_ch_dec[s], self.num_output_channels))

        self.decoder = nn.ModuleList(convs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        # type: (List[torch.Tensor]) -> torch.Tensor
        output: torch.Tensor = input_features[-1]

        # decoder
        x: torch.Tensor = input_features[-1]
        i = 9
        for mod in self.decoder:
            if (i >= 0):
                if i % 2 == 1:
                    x = mod(x)
                    seq = [upsample(x)]
                    if self.use_skips and i // 2 > 0:
                        seq += [input_features[i // 2 - 1]]
                    x = torch.cat(seq, 1)
                else:
                    x = mod(x)
            elif i == -1:
                output = self.sigmoid(mod(x))

            i -= 1

        return output
