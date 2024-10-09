# Copyright (c) 2024, Zhendong Peng (pzd17@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

import torch
from torch import nn
from vocos.feature_extractors import MelSpectrogramFeatures

from .decoder import DVAEDecoder
from .gfsq import GFSQ


class DVAE(nn.Module):
    def __init__(
        self,
        decoder_config: dict,
        encoder_config: Optional[dict] = None,
        vq_config: Optional[dict] = None,
        dim=512,
        device: torch.device = torch.device("cpu"),
    ):
        # DVAE Block modified from ChatTTS.
        super().__init__()
        coef = torch.rand(100)
        self.register_buffer("coef", coef.unsqueeze(0).unsqueeze_(2))

        # encoder
        self.downsample_conv = nn.Sequential(
            nn.Conv1d(100, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 4, 2, 1),
            nn.GELU(),
        )
        self.preprocessor_mel = MelSpectrogramFeatures()
        self.encoder: Optional[DVAEDecoder] = DVAEDecoder(**encoder_config)

        # decoder
        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        self.vq_layer = GFSQ(**vq_config)

    @torch.inference_mode()
    def forward(self, inp: torch.Tensor, mode: str = "decode") -> torch.Tensor:
        assert mode in ["encode", "decode"]
        if mode == "encode":
            mel = self.preprocessor_mel(inp) / self.coef.view(100, 1)
            x = self.downsample_conv(mel).unsqueeze_(0)
            x = self.encoder(x)
            ind = self.vq_layer(x)
            return ind

        vq_feats = self.vq_layer._embed(inp)
        vq_feats = (
            vq_feats.view(
                (vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2))
            )
            .permute(0, 2, 3, 1)
            .flatten(2)
        )
        return self.out_conv(self.decoder(x=vq_feats)) * self.coef
