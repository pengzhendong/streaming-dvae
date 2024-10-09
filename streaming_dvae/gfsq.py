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

import math
from typing import List

import torch
import torch.nn as nn
from vector_quantize_pytorch import GroupedResidualFSQ


class GFSQ(nn.Module):
    def __init__(self, dim: int, levels: List[int], G: int, R: int):
        # GFSQ Block modified from ChatTTS.
        super(GFSQ, self).__init__()
        self.quantizer = GroupedResidualFSQ(
            dim=dim, levels=list(levels), num_quantizers=R, groups=G
        )
        self.n_ind = math.prod(levels)
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose_(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.transpose_(1, 2)
        _, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous()
        ind = ind.view(ind.size(0), ind.size(1), -1)
        return ind.transpose_(1, 2)
