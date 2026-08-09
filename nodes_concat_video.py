"""SCG video concatenation — join two videos, A then B, with their soundtracks.

ComfyUI can splice IMAGE batches with a tensor cat, but that ignores every
question that actually decides whether the result plays correctly: differing
resolution, differing frame rate, differing sample rate, differing channel
counts, one side having no audio at all, and audio whose duration does not
match its own picture. Getting any of those wrong desyncs sound from picture,
and the error persists for the whole remainder of the timeline.

The rule this node follows: **each input is made self-consistent before the
join**, so its audio is exactly as long as its own video. Concatenating two
self-consistent clips is then just addition, and B cannot inherit a drift from
A. A seam fade is available that preserves length, because an overlap-add fade
shortens the audio while the video keeps every frame — which is precisely how
sound drifts ahead of picture down a chain.
"""

import logging
from fractions import Fraction

import torch

from comfy_api.latest import InputImpl, Types, io

from .video_join import (
    CHANNEL_MODES,
    QUALITY,
    SCALE_METHODS,
    SEAM_MODES,
    SIZE_MODES,
    SR_MODES,
    _channels_of,
    _conform,
    _fade_shape,
    _fit_channels,
    _join,
    _pick,
    _pick_channels,
    _pick_size,
    _resample,
    _resize,
    _retime,
    _set_duration,
)

_LOG = logging.getLogger("scg_utils.concat_video")


class SCGConcatenateVideo(io.ComfyNode):
    """Join two videos, A then B, carrying their soundtracks with them.

    Owns: resolution, frame rate, and audio conformance across the join, plus
    silence for a side that has no sound so the other side's audio cannot run
    under the wrong picture.
    Not responsible for: encoding or saving — it emits a VIDEO.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGConcatenateVideo",
            display_name="SCG Concatenate Video",
            category="scg-utils/video",
            description=(
                "Concatenate video_A then video_B with their audio, reconciling "
                "resolution, frame rate, sample rate and channels. Each clip's "
                "audio is fitted to its own picture length first, so the join "
                "cannot desync the remainder of the timeline."
            ),
            inputs=[
                io.Video.Input("video_A"),
                io.Video.Input("video_B"),
                io.Combo.Input("merge_strategy", options=SIZE_MODES, default="match A",
                               tooltip="Output resolution. The other clip is scaled to it."),
                io.Combo.Input("scale_method", options=SCALE_METHODS, default="lanczos",
                               tooltip="Filter used when a clip has to be scaled."),
                io.Combo.Input("crop", options=["center", "disabled"], default="center",
                               tooltip="Only matters when the two clips differ in aspect. "
                                       "'center' keeps proportions and loses the edges; "
                                       "'disabled' stretches to fit."),
                io.Combo.Input("frame_rate", options=SR_MODES, default="match A",
                               tooltip="Output frame rate. The other clip is retimed by "
                                       "duplicating or dropping frames, preserving its "
                                       "real-time duration."),
                io.Combo.Input("sample_rate", options=SR_MODES, default="highest",
                               tooltip="Output audio rate."),
                io.Combo.Input("channels", options=CHANNEL_MODES, default="max",
                               tooltip="Output audio channel count."),
                io.Combo.Input("resample_quality", options=QUALITY, default="best",
                               tooltip="Audio resampling filter width."),
                io.Combo.Input("audio_seam", options=SEAM_MODES, default="hard cut",
                               tooltip="An overlapping crossfade is deliberately not "
                                       "offered here: it shortens the audio while the "
                                       "video keeps every frame, which desyncs everything "
                                       "after the join."),
                io.Int.Input("seam_ms", default=0, min=0, max=500, step=1,
                             tooltip="Seam fade length in milliseconds."),
            ],
            outputs=[io.Video.Output(), io.Int.Output(display_name="frame_count")],
        )

    @classmethod
    def _stream(cls, video, width, height, fps, scale_method, crop):
        """One input's picture, conformed to the output geometry and rate."""
        parts = video.get_components()
        images = _retime(parts.images, float(parts.frame_rate), fps)
        images = _resize(images, width, height, scale_method, crop)
        if parts.alpha is not None:
            _LOG.warning("scg_concat_video: alpha channel dropped (not carried "
                         "through concatenation)")
        return parts, images

    @classmethod
    def _sound(cls, parts, frames, fps, sample_rate, channels, quality):
        """One input's audio at the output format, exactly as long as its picture."""
        seconds = frames / fps
        if parts.audio is None:
            waveform = torch.zeros(1, channels,
                                   max(1, int(round(seconds * sample_rate))))
        else:
            waveform = _conform(parts.audio, sample_rate, channels, quality)
        return _set_duration(waveform, sample_rate, seconds)

    @classmethod
    def execute(cls, video_A, video_B, merge_strategy, scale_method, crop,
                frame_rate, sample_rate, channels, resample_quality, audio_seam,
                seam_ms) -> io.NodeOutput:
        wa, ha = video_A.get_dimensions()
        wb, hb = video_B.get_dimensions()
        fps_a = float(video_A.get_frame_rate())
        fps_b = float(video_B.get_frame_rate())
        width, height = _pick_size(merge_strategy, wa, ha, wb, hb)
        fps = _pick(frame_rate, fps_a, fps_b)

        parts_a, imgs_a = cls._stream(video_A, width, height, fps, scale_method, crop)
        parts_b, imgs_b = cls._stream(video_B, width, height, fps, scale_method, crop)

        sr_a = int(parts_a.audio["sample_rate"]) if parts_a.audio else 0
        sr_b = int(parts_b.audio["sample_rate"]) if parts_b.audio else 0
        has_audio = bool(sr_a or sr_b)
        out_sr = _pick(sample_rate, sr_a or sr_b, sr_b or sr_a) if has_audio else 0
        out_ch = _pick_channels(
            channels,
            _channels_of(parts_a.audio) if parts_a.audio else 1,
            _channels_of(parts_b.audio) if parts_b.audio else 1)

        audio = None
        if has_audio:
            first = cls._sound(parts_a, imgs_a.shape[0], fps, out_sr, out_ch,
                               resample_quality)
            second = cls._sound(parts_b, imgs_b.shape[0], fps, out_sr, out_ch,
                                resample_quality)
            joined = _join(first, second, out_sr, audio_seam, seam_ms)
            audio = {"waveform": joined, "sample_rate": out_sr}

        images = torch.cat([imgs_a, imgs_b], dim=0)
        _LOG.info("scg_concat_video: %dx%d@%.4g + %dx%d@%.4g -> %dx%d@%.4g, "
                  "%d + %d = %d frames (%.3fs), audio %s",
                  wa, ha, fps_a, wb, hb, fps_b, width, height, fps,
                  imgs_a.shape[0], imgs_b.shape[0], images.shape[0],
                  images.shape[0] / fps,
                  f"{out_sr}Hz/{out_ch}ch" if has_audio else "none")
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=images, audio=audio,
            frame_rate=Fraction(fps).limit_denominator(1000)))
        return io.NodeOutput(video, int(images.shape[0]))
