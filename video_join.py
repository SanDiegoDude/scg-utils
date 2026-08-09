"""Primitives shared by the pack's video-joining nodes.

Joining two clips is only partly about splicing tensors. Before a join is even
valid the two sides have to agree on resolution, frame rate, sample rate and
channel count, and each side's audio has to be exactly as long as its own
picture — otherwise the join hands its own error to everything downstream of
it. Those decisions are identical whatever the join means, so they live here
once and every joining node routes through them.

Owns: strategy resolution, geometry and rate conformance, audio conformance,
duration fitting, and the equal-power taper shape.
Not responsible for: what the join means. Whether the second clip follows the
first or overlaps it belongs to the node, not to these primitives.
"""

import logging
from dataclasses import dataclass

import torch

import comfy.utils

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("scg_utils.video_join")

SR_MODES = ["match A", "match B", "highest", "lowest"]
CHANNEL_MODES = ["max", "match A", "match B", "mono", "stereo"]
QUALITY = ["best", "fast"]
SIZE_MODES = ["match A", "match B", "match largest", "match smallest"]
SCALE_METHODS = ["lanczos", "bicubic", "bilinear", "area", "nearest-exact"]
SEAM_MODES = ["hard cut", "fade at seam (keeps length)"]


def _pick(mode, a, b):
    """Resolve a 'match ...' strategy against the two inputs' values."""
    match mode:
        case "match A":
            return a
        case "match B":
            return b
        case "highest":
            return max(a, b)
        case "lowest":
            return min(a, b)
    raise ValueError(f"unknown strategy {mode!r}")


def _pick_channels(mode, a, b):
    match mode:
        case "match A":
            return a
        case "match B":
            return b
        case "mono":
            return 1
        case "stereo":
            return 2
        case "max":
            return max(a, b)
    raise ValueError(f"unknown channel strategy {mode!r}")


def _pick_size(mode, wa, ha, wb, hb):
    match mode:
        case "match A":
            return wa, ha
        case "match B":
            return wb, hb
        case "match largest":
            return (wa, ha) if wa * ha >= wb * hb else (wb, hb)
        case "match smallest":
            return (wa, ha) if wa * ha <= wb * hb else (wb, hb)
    raise ValueError(f"unknown size strategy {mode!r}")


def _channels_of(audio):
    """Channel count of an AUDIO dict, whose waveform may be [B, C, L] or [C, L]."""
    waveform = audio["waveform"]
    return int(waveform.shape[1] if waveform.ndim == 3 else waveform.shape[0])


def _fit_channels(waveform, target):
    """Force [B, C, L] to `target` channels."""
    channels = int(waveform.shape[1])
    if channels == target:
        return waveform
    if target == 1:
        return waveform.mean(dim=1, keepdim=True)
    if channels == 1:
        return waveform.repeat(1, target, 1)
    if channels > target:
        return waveform[:, :target]
    pad = waveform[:, -1:].repeat(1, target - channels, 1)
    return torch.cat([waveform, pad], dim=1)


def _resample(waveform, src_sr, dst_sr, quality):
    if src_sr == dst_sr:
        return waveform
    if torchaudio is None:
        raise RuntimeError(
            f"resampling {src_sr} Hz to {dst_sr} Hz needs torchaudio, which is "
            "not available. Set the sample rate strategy to avoid resampling.")
    best = quality == "best"
    return torchaudio.functional.resample(
        waveform, src_sr, dst_sr,
        lowpass_filter_width=64 if best else 6,
        resampling_method="sinc_interp_kaiser" if best else "sinc_interp_hann")


def _conform(audio, sample_rate, channels, quality):
    """One clip's waveform at the target rate and channel count."""
    waveform = audio["waveform"]
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    waveform = _resample(waveform, int(audio["sample_rate"]), sample_rate, quality)
    return _fit_channels(waveform, channels)


def _set_duration(waveform, sample_rate, seconds):
    """Trim or silence-pad so the clip lasts exactly `seconds`.

    This is what stops a drift crossing the join: if a clip's sound is a few
    milliseconds off its own picture, every later clip inherits that offset.
    """
    want = max(1, int(round(seconds * sample_rate)))
    have = int(waveform.shape[-1])
    if have > want:
        return waveform[..., :want]
    if have < want:
        return torch.nn.functional.pad(waveform, (0, want - have))
    return waveform


def _fade_shape(n, dtype, device):
    t = torch.linspace(0, 1, n, dtype=dtype, device=device)
    return torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)


def _join(first, second, sample_rate, mode, ms):
    """Concatenate two conformed waveforms, tapering the seam in place if asked.

    The taper is applied inside each clip rather than by overlapping them, so
    the total length stays the sum of the two inputs. `ms` spans the whole
    transition, hence half of it on each side.
    """
    if mode == "hard cut" or ms <= 0:
        return torch.cat([first, second], dim=-1)
    k = min(max(1, int(sample_rate * ms / 2000.0)),
            first.shape[-1], second.shape[-1])
    out_g, in_g = _fade_shape(k, first.dtype, first.device)
    first = first.clone()
    second = second.clone()
    first[..., -k:] *= out_g
    second[..., :k] *= in_g
    return torch.cat([first, second], dim=-1)


def _resize(images, width, height, method, crop):
    if int(images.shape[2]) == width and int(images.shape[1]) == height:
        return images
    samples = images[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, method, crop)
    return samples.movedim(1, -1)


def _retime(images, src_fps, dst_fps):
    """Nearest-frame resample onto dst_fps, preserving real-time duration.

    Frames are duplicated or dropped, never blended: a blended frame is a
    picture that never existed, which reads worse than a repeat on motion.
    """
    if abs(src_fps - dst_fps) < 1e-6:
        return images
    have = int(images.shape[0])
    want = max(1, int(round(have * dst_fps / src_fps)))
    step = src_fps / dst_fps
    index = [min(have - 1, max(0, round(k * step))) for k in range(want)]
    return images[index]


@dataclass(frozen=True)
class ResolvedClip:
    """A join input's picture, sound, and rate, once its source is settled.

    Owns: the one shape every joining node's downstream math consumes,
    whether the input arrived as a real VIDEO or as IMAGE+AUDIO+fps.
    Not responsible for: choosing which source wins — that's
    resolve_clip_source's job — or any geometry/rate reconciliation, which
    stays in the pickers and conform helpers above.
    """

    images: torch.Tensor
    frame_rate: float
    audio: dict | None = None
    alpha: torch.Tensor | None = None


def resolve_clip_source(side, video, images, audio, fps):
    """One join side, from a VIDEO if wired, else IMAGE(+AUDIO, +fps).

    VIDEO always wins when both are wired: it already carries a true frame
    rate, so images/audio/fps underneath it are redundant rather than
    conflicting, and that choice is logged rather than made silently. `fps`
    therefore only ever matters on the IMAGE path — a VIDEO's own rate always
    overrides it.
    """
    if video is not None:
        if images is not None:
            _LOG.info(
                "scg_video_join: %s side has both video_%s and images_%s "
                "wired; using video_%s, ignoring images_%s/audio_%s/fps_%s.",
                side, side, side, side, side, side, side)
        parts = video.get_components()
        return ResolvedClip(parts.images, float(parts.frame_rate), parts.audio,
                            parts.alpha)
    if images is not None:
        return ResolvedClip(images, float(fps), audio)
    raise ValueError(
        f"{side} side has neither a VIDEO nor an IMAGE wired. Connect "
        f"video_{side}, or images_{side} (optionally with audio_{side}).")
