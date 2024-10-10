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

import torch
import yaml
from modelscope import snapshot_download
from streaming_vocos import StreamingVocos

from .dvae import DVAE


class StreamingDVAE:
    def __init__(
        self,
        repo_id: str = "pengzhendong/dvae",
        device: str = "cpu",
        chunk_size_ms: int = None,
        padding_ms: int = 40,
        vocos_chunk_size_ms: int = 300,
        vocos_padding_ms: int = None,
    ):
        self.device = device
        repo_dir = snapshot_download(repo_id)
        config = yaml.safe_load(open(f"{repo_dir}/config.yaml"))
        weights_path = f"{repo_dir}/pytorch_model.bin"
        weights = torch.load(weights_path, weights_only=True, mmap=True)

        dvae = DVAE(config["decoder"], config["encoder"], config["vq"])
        dvae.load_state_dict(weights)
        self.dvae = dvae.to(self.device)

        self.vocos = StreamingVocos(
            device=self.device,
            chunk_size_ms=vocos_chunk_size_ms,
            padding_ms=vocos_padding_ms,
        )
        if chunk_size_ms is None:
            chunk_size = self.vocos.chunk_size + self.vocos.padding
        else:
            chunk_size = chunk_size_ms / 10
        self.chunk_size = int(math.ceil(chunk_size / 2))
        self.padding = int(padding_ms / 10 / 2)

        self.cur_idx = -1
        self.num_quantizers = len(config["vq"]["levels"])
        self.caches_shape = (1, self.num_quantizers, self.chunk_size + 2 * self.padding)
        self.caches = torch.zeros(self.caches_shape, dtype=torch.long).to(self.device)

    def reset(self):
        self.cur_idx = -1
        self.caches = torch.zeros(self.caches_shape, dtype=torch.long).to(self.device)

    def extract_features(self, audio: torch.Tensor):
        return self.vocos.feature_extractor(audio)

    def encode(self, audio: torch.Tensor):
        return self.dvae(audio, mode="encode")

    def decode(self, codes: torch.Tensor, to_mel: bool = False):
        mel = self.dvae(codes, mode="decode")
        return mel if to_mel else self.vocos.decode(mel)

    def get_size(self):
        """
        Method to get the length of unprocessed codes or features.
        """
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def decode_caches(self, to_mel: bool = False):
        cur_size = self.get_size()
        if cur_size == 0:
            if to_mel:
                return torch.empty(1, self.num_quantizers, 0).to(self.device)
            return self.vocos.decode_caches()
        mel = self.decode(self.caches, to_mel=True)
        mel = mel[:, :, self.padding * 2 :]
        mel = mel[:, :, (self.chunk_size - cur_size) * 2 :]
        self.reset()
        if to_mel:
            return mel
        audios = list(self.vocos.streaming_decode(mel, is_last=True))
        return torch.cat(audios, dim=1)

    def streaming_decode(self, codes: torch.Tensor, to_mel=False, is_last=False):
        for idx, code in enumerate(torch.unbind(codes, dim=2)):
            self.caches = torch.roll(self.caches, shifts=-1, dims=2)
            self.caches[:, :, -1] = code
            self.cur_idx += 1
            is_last_code = is_last and idx == codes.shape[2] - 1
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last_code:
                continue
            mel = self.decode(self.caches, to_mel=True)
            mel = mel[:, :, self.padding * 2 :]
            if cur_size != self.chunk_size:
                mel = mel[:, :, (self.chunk_size - cur_size) * 2 :]
            if not is_last_code:
                mel = mel[:, :, : self.chunk_size * 2]
            else:
                self.reset()
            if to_mel:
                yield mel
            else:
                for audio in self.vocos.streaming_decode(mel, is_last):
                    yield audio

    def test_streaming_decode(self, wav_path: str):
        import torchaudio

        audio, sr = torchaudio.load(wav_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.to(self.device)
        audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=24000)
        mel = self.extract_features(audio)
        codes = self.encode(audio[0])

        mel_hat = []
        for code in torch.unbind(codes, dim=2):
            mel_hat += self.streaming_decode(code[:, :, None], to_mel=True)
        mel_hat.append(self.decode_caches(to_mel=True))
        mel_hat = torch.cat(mel_hat, dim=2)
        t = min(mel.shape[2], mel_hat.shape[2])
        similarity = torch.cosine_similarity(mel[:, :, :t], mel_hat).mean()
        print(mel.shape[2], mel_hat.shape[2], similarity)
