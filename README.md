# streaming-dvae


Streaming DVAE supports streaming reconstruction of mel features or audio from DVAE tokens.

## Usage

- To mel features

``` python
from streaming_dave import StreamingDVAE

mel = []
dvae = StreamingDVAE()
codes = dvae.encode(audio)

for code in torch.unbind(codes, dim=2):
    mel += dvae.streaming_decode(code[:, :, None], to_mel=True)
mel.append(dvae.decode_caches(to_mel=True))
mel = torch.cat(mel, dim=2)
```

- To audio

``` python
from streaming_dave import StreamingDVAE

audios = []
dvae = StreamingDVAE()
codes = dvae.encode(audio)

for code in torch.unbind(codes, dim=2):
    audios += dvae.streaming_decode(code[:, :, None])
audios.append(dvae.decode_caches())
audios = torch.cat(audios, dim=2)
```
