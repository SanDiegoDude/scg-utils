"""SCG MiniMax H3 extension concatenation — join a clip to its own continuation.

An H3 extension does not begin where its source ended. The motion context pins
a run of the source's tail frames into the head of the new clip, so the first
`trim_frames` frames the sampler delivers are a re-generation of footage the
original already holds. Splice the two clips unchanged and that span plays
twice.

This is therefore the one join in the pack where consuming an overlap is
correct. The plain concatenate node refuses an overlapping crossfade because
there the two clips are different content, so an overlap would shorten the
sound while the picture kept every frame and desync everything downstream.
Here the overlap is the *same* content rendered twice: removing it from picture
and sound alike shortens the result by exactly the duplicate, and the timeline
stays true.

Owns: the overlap contract between the two clips, held identically in picture
and in sound, under both join modes.
Not responsible for: reconciling resolution, rate, sample rate and channels —
those live in video_join, shared with the plain concatenate node.
"""

import logging
from dataclasses import dataclass
from fractions import Fraction

import torch

from comfy_api.latest import InputImpl, Types, io

from .video_join import (
    CHANNEL_MODES,
    QUALITY,
    SCALE_METHODS,
    SIZE_MODES,
    SR_MODES,
    _channels_of,
    _conform,
    _fade_shape,
    _pick,
    _pick_channels,
    _pick_size,
    _resize,
    _retime,
    _set_duration,
    resolve_clip_source,
)

_LOG = logging.getLogger("scg_utils.concat_h3_extend")

H3_FPS = 24.0
SILENCE_SAMPLE_RATE = 44100

# MiniMaxH3MotionContext's VIDEO_RUN_GRID, plus the no-overlap case an
# already-trimmed extension arrives with.
OVERLAP_CHOICES = [0, 1, 5, 22, 39]

MATCH_ORIGINAL = "match original"
MATCH_EXTENSION = "match extension"
_RELABEL = {"match A": MATCH_ORIGINAL, "match B": MATCH_EXTENSION}
_CANONICAL = {v: k for k, v in _RELABEL.items()}

JOIN_TRIM = "trim extension head"
JOIN_BLEND = "blend over overlap"
JOIN_MODES = [JOIN_TRIM, JOIN_BLEND]


def _relabel(modes):
    """A shared strategy list in this node's original/extension wording."""
    return [_RELABEL.get(mode, mode) for mode in modes]


def _canonical(mode):
    """This node's wording back in the shared pickers' A/B terms."""
    return _CANONICAL.get(mode, mode)


SIZE_CHOICES = _relabel(SIZE_MODES)
RATE_CHOICES = _relabel(SR_MODES)
CHANNEL_CHOICES = _relabel(CHANNEL_MODES)


@dataclass(frozen=True)
class OutputFormat:
    """The single target both clips are conformed to before they meet.

    Owns: the numbers every stream has to agree on, and the frames-to-seconds
    and frames-to-samples conversions taken against them.
    Not responsible for: choosing them — the node resolves the strategies — or
    performing any conversion on a waveform or an image batch.
    """

    width: int
    height: int
    fps: float
    scale_method: str
    crop: str
    sample_rate: int
    channels: int
    quality: str

    def has_audio(self):
        return self.sample_rate > 0

    def seconds(self, frames):
        return frames / self.fps

    def samples(self, frames):
        return int(round(self.seconds(frames) * self.sample_rate))


class SCGMiniMaxH3ExtensionConcat(io.ComfyNode):
    """Join an original clip to its H3 extension, consuming the duplicated head.

    Owns: how much of the extension re-generates the original's tail, and the
    guarantee that picture and sound each lose exactly that span, so both join
    modes deliver the same length and neither drifts.
    Not responsible for: producing the extension, or the pinned-head trim that
    MiniMaxH3MotionContextTrim performs — an extension that already went
    through it arrives with an overlap of 0 and is simply appended. Also not
    responsible for resolving whether a side is a VIDEO or images+audio+fps —
    that's resolve_clip_source's job, shared with any future joining node.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGMiniMaxH3ExtensionConcat",
            display_name="SCG MiniMax H3 Video Extension Concatenator",
            category="scg-utils/video",
            description=(
                "Join an original clip to an SCG MiniMax H3 Video Extend result, "
                "consuming the extension's pinned head instead of playing it "
                "twice. Picture and sound lose the same span, so the output is "
                "always original + extension - overlap frames with its audio "
                "exactly that long. Each side takes a VIDEO or an IMAGE batch "
                "with optional AUDIO; VIDEO wins when both are wired."
            ),
            inputs=[
                io.Video.Input("video_original", optional=True,
                               tooltip="The ORIGINAL clip. Wins over "
                                       "images_original when both are wired. "
                                       "Required unless images_original is "
                                       "wired instead."),
                io.Image.Input("images_original", optional=True,
                               tooltip="Alternative to video_original: the "
                                       "ORIGINAL clip as raw frames. Ignored "
                                       "if video_original is wired."),
                io.Audio.Input("audio_original", optional=True,
                               tooltip="Soundtrack for images_original. "
                                       "Optional — omitted means silent, the "
                                       "same as a VIDEO with no audio track. "
                                       "Ignored if video_original is wired."),
                io.Float.Input("fps_original", default=24.0, min=1.0, max=240.0,
                               step=0.01,
                               tooltip="Frame rate of images_original — an "
                                       "IMAGE batch carries no rate of its "
                                       "own. Ignored (and irrelevant) once "
                                       "video_original is wired, since a "
                                       "VIDEO already reports its own rate."),
                io.Video.Input("video_extension", optional=True,
                               tooltip="The H3 extension clip. Wins over "
                                       "images_extension when both are wired. "
                                       "Required unless images_extension is "
                                       "wired instead."),
                io.Image.Input("images_extension", optional=True,
                               tooltip="Alternative to video_extension: the "
                                       "EXTENSION clip as raw frames. Ignored "
                                       "if video_extension is wired."),
                io.Audio.Input("audio_extension", optional=True,
                               tooltip="Soundtrack for images_extension. "
                                       "Optional — omitted means silent, the "
                                       "same as a VIDEO with no audio track. "
                                       "Ignored if video_extension is wired."),
                io.Float.Input("fps_extension", default=24.0, min=1.0, max=240.0,
                               step=0.01,
                               tooltip="Frame rate of images_extension — an "
                                       "IMAGE batch carries no rate of its "
                                       "own. H3's native decode rate is 24 "
                                       "fps, correct for the extension "
                                       "essentially always. Ignored (and "
                                       "irrelevant) once video_extension is "
                                       "wired, since a VIDEO already reports "
                                       "its own rate."),
                io.Combo.Input("overlap_frames", options=OVERLAP_CHOICES, default=22,
                               tooltip="How many of the extension's opening frames "
                                       "repeat the original's tail. The video VAE "
                                       "only resolves runs of 1, 5, 22 and 39, so "
                                       "those are the only counts a motion context "
                                       "can produce; 0 means the extension was "
                                       "already trimmed. Wire trim_frames instead "
                                       "of trusting this."),
                io.Combo.Input("resolution", options=SIZE_CHOICES,
                               default=MATCH_ORIGINAL,
                               tooltip="Output resolution. The other clip is scaled "
                                       "to it."),
                io.Combo.Input("scale_method", options=SCALE_METHODS, default="lanczos",
                               tooltip="Filter used when a clip has to be scaled."),
                io.Combo.Input("crop", options=["center", "disabled"], default="center",
                               tooltip="Only matters when the two clips differ in "
                                       "aspect. 'center' keeps proportions and loses "
                                       "the edges; 'disabled' stretches to fit."),
                io.Combo.Input("join", options=JOIN_MODES, default=JOIN_TRIM,
                               tooltip="What happens over the overlap. 'trim "
                                       "extension head' keeps the original's real "
                                       "frames, drops the extension's re-generated "
                                       "ones and hard cuts — the equivalent of "
                                       "Motion Context Trim plus a plain "
                                       "concatenation, and the predictable default. "
                                       "'blend over overlap' instead dissolves the "
                                       "original's real frames into the extension's "
                                       "rendering of the same frames, spreading the "
                                       "step in exposure and detail across the whole "
                                       "run at the cost of a double exposure "
                                       "wherever the two renderings disagree. Both "
                                       "produce identical length."),
                io.Combo.Input("frame_rate", options=RATE_CHOICES,
                               default=MATCH_ORIGINAL,
                               tooltip="Output frame rate. The other clip is retimed "
                                       "by duplicating or dropping frames, preserving "
                                       "its real-time duration."),
                io.Combo.Input("sample_rate", options=RATE_CHOICES, default="highest",
                               tooltip="Output audio rate."),
                io.Combo.Input("channels", options=CHANNEL_CHOICES, default="max",
                               tooltip="Output audio channel count."),
                io.Combo.Input("resample_quality", options=QUALITY, default="best",
                               tooltip="Audio resampling filter width."),
                io.Int.Input("trim_frames", optional=True, force_input=True,
                             tooltip="Preferred: wire this from SCG MiniMax H3 Video "
                                     "Extend's trim_frames output and the two can "
                                     "never disagree. Overrides overlap_frames "
                                     "whenever it is connected. A value outside "
                                     f"{OVERLAP_CHOICES} did not come off a 24 fps "
                                     "motion context; it is used as given and "
                                     "logged as a warning, because the encoder's "
                                     "own count is still a better answer than the "
                                     "dropdown's guess."),
            ],
            outputs=[
                io.Video.Output(),
                io.Int.Output(display_name="frame_count"),
                io.Image.Output(display_name="images"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def _resolve_overlap(cls, overlap_frames, trim_frames):
        """Frames of the extension's head that repeat the original's tail."""
        if trim_frames is None:
            return int(overlap_frames)
        overlap = max(0, int(trim_frames))
        if overlap not in OVERLAP_CHOICES:
            _LOG.warning(
                "scg_h3_extension_concat: wired trim_frames=%d is not one of the "
                "video VAE run lengths %s, so it did not come off a %g fps motion "
                "context. Using it as given; check source_fps and context_length "
                "on the extend node if the join repeats or skips content.",
                overlap, OVERLAP_CHOICES, H3_FPS)
        return overlap

    @classmethod
    def _resolve_format(cls, parts_o, parts_e, resolution, scale_method, crop,
                        frame_rate, sample_rate, channels, quality):
        """Every output property, resolved from the strategies in one place."""
        ho, wo = int(parts_o.images.shape[1]), int(parts_o.images.shape[2])
        he, we = int(parts_e.images.shape[1]), int(parts_e.images.shape[2])
        width, height = _pick_size(_canonical(resolution), wo, ho, we, he)
        fps = _pick(_canonical(frame_rate), float(parts_o.frame_rate),
                    float(parts_e.frame_rate))
        sr_o = int(parts_o.audio["sample_rate"]) if parts_o.audio else 0
        sr_e = int(parts_e.audio["sample_rate"]) if parts_e.audio else 0
        out_sr = (_pick(_canonical(sample_rate), sr_o or sr_e, sr_e or sr_o)
                  if sr_o or sr_e else SILENCE_SAMPLE_RATE)
        out_ch = _pick_channels(
            _canonical(channels),
            _channels_of(parts_o.audio) if parts_o.audio else 1,
            _channels_of(parts_e.audio) if parts_e.audio else 1)
        return OutputFormat(width, height, fps, scale_method, crop, out_sr,
                            out_ch, quality)

    @classmethod
    def _native_cut(cls, parts, overlap):
        """The overlap as a frame count on the extension's own timeline.

        trim_frames counts frames of H3's 24 fps decode, so overlap/24 seconds
        is the invariant, not the frame count: a retimed extension needs a
        different number of its own frames to cover the same instant.
        """
        fps = float(parts.frame_rate)
        total = int(parts.images.shape[0])
        cut = int(round(overlap / H3_FPS * fps))
        if overlap and abs(fps - H3_FPS) > 1e-3:
            _LOG.warning(
                "scg_h3_extension_concat: the extension reports %.4g fps, not "
                "H3's %g. Reading the %d frame overlap as %.4fs and cutting %d "
                "of its frames.", fps, H3_FPS, overlap, overlap / H3_FPS, cut)
        if cut >= total:
            raise ValueError(
                f"SCG MiniMax H3 Video Extension Concatenator: an overlap of "
                f"{overlap} frames consumes the whole {total} frame extension. "
                "Check overlap_frames, or whether the extension already went "
                "through Motion Context Trim (in which case the overlap is 0).")
        return cut

    @classmethod
    def _picture(cls, images, src_fps, fmt):
        """One stream's frames at the output geometry and rate."""
        return _resize(_retime(images, src_fps, fmt.fps), fmt.width, fmt.height,
                       fmt.scale_method, fmt.crop)

    @classmethod
    def _silence(cls, frames, fmt):
        return torch.zeros(1, fmt.channels, max(1, fmt.samples(frames)))

    @classmethod
    def _sound(cls, audio, frames, fmt):
        """One stream's waveform at the output format, as long as its own picture."""
        if audio is None:
            return cls._silence(frames, fmt)
        waveform = _conform(audio, fmt.sample_rate, fmt.channels, fmt.quality)
        return _set_duration(waveform, fmt.sample_rate, fmt.seconds(frames))

    @classmethod
    def _native_audio(cls, parts):
        """A clip's own waveform, locked to its own picture length first.

        Fitting before the cut is what puts the overlap boundary at a known
        instant: H3 ships a few milliseconds more sound than picture, and
        slicing an unfitted waveform would carry that error into the join.
        """
        if parts.audio is None:
            return None
        waveform = parts.audio["waveform"]
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        sample_rate = int(parts.audio["sample_rate"])
        seconds = int(parts.images.shape[0]) / float(parts.frame_rate)
        return {"waveform": _set_duration(waveform, sample_rate, seconds),
                "sample_rate": sample_rate}

    @classmethod
    def _extension_sound(cls, parts, overlap, span, body_frames, fmt):
        """The extension's sound cut at the overlap, in its own native terms.

        The cut precedes resampling and retiming so the span removed is exactly
        the overlap's real duration; cutting afterwards would round the boundary
        against the output rate on top of the native one, and picture and sound
        would then lose spans that differ by the rounding.
        """
        native = cls._native_audio(parts)
        if native is None:
            return (cls._silence(span, fmt) if span else None,
                    cls._silence(body_frames, fmt))
        rate = native["sample_rate"]
        cut = int(round(overlap / H3_FPS * rate))
        head = (cls._sound({"waveform": native["waveform"][..., :cut],
                            "sample_rate": rate}, span, fmt) if span else None)
        body = cls._sound({"waveform": native["waveform"][..., cut:],
                           "sample_rate": rate}, body_frames, fmt)
        return head, body

    @classmethod
    def _blend_pictures(cls, tail, head):
        """Dissolve two renderings of the same frames, linearly in level."""
        span = int(tail.shape[0])
        ramp = torch.linspace(0, 1, span, dtype=tail.dtype,
                              device=tail.device).view(-1, 1, 1, 1)
        return tail * (1 - ramp) + head * ramp

    @classmethod
    def _blend_sound(cls, first, head, body):
        """Mix the overlap's two renderings without changing the total length."""
        span = min(int(head.shape[-1]), int(first.shape[-1]))
        out_g, in_g = _fade_shape(span, first.dtype, first.device)
        mixed = first[..., -span:] * out_g + head[..., :span] * in_g
        peak = float(mixed.abs().max())
        if peak > 1.0:
            _LOG.warning(
                "scg_h3_extension_concat: the blended overlap peaks at %.2f. Both "
                "sides carry the same content, so an equal-power sum reinforces "
                "instead of cancelling. Left unscaled — switch to %r if it clips.",
                peak, JOIN_TRIM)
        return torch.cat([first[..., :-span], mixed, body], dim=-1)

    @classmethod
    def _blend_span(cls, original, head):
        """Frames the dissolve covers: the overlap, or the original if it is shorter."""
        span = min(int(head.shape[0]), int(original.shape[0]))
        if span < int(head.shape[0]):
            _LOG.warning(
                "scg_h3_extension_concat: the original is only %d frames, shorter "
                "than the %d frame overlap; dissolving across all of it.",
                span, int(head.shape[0]))
        return span

    @classmethod
    def execute(cls, overlap_frames, resolution, scale_method, crop, join,
                frame_rate, sample_rate, channels, resample_quality,
                video_original=None, images_original=None, audio_original=None,
                fps_original=24.0, video_extension=None, images_extension=None,
                audio_extension=None, fps_extension=24.0,
                trim_frames=None) -> io.NodeOutput:
        parts_o = resolve_clip_source("original", video_original,
                                      images_original, audio_original,
                                      fps_original)
        parts_e = resolve_clip_source("extension", video_extension,
                                      images_extension, audio_extension,
                                      fps_extension)
        overlap = cls._resolve_overlap(overlap_frames, trim_frames)
        fmt = cls._resolve_format(parts_o, parts_e, resolution, scale_method,
                                  crop, frame_rate, sample_rate, channels,
                                  resample_quality)
        cut = cls._native_cut(parts_e, overlap)
        if parts_o.alpha is not None or parts_e.alpha is not None:
            _LOG.warning("scg_h3_extension_concat: alpha channel dropped (not "
                         "carried through concatenation)")

        fps_e = float(parts_e.frame_rate)
        original = cls._picture(parts_o.images, float(parts_o.frame_rate), fmt)
        body = cls._picture(parts_e.images[cut:], fps_e, fmt)
        head = (cls._picture(parts_e.images[:cut], fps_e, fmt)
                if join == JOIN_BLEND and cut else None)
        span = cls._blend_span(original, head) if head is not None else 0

        images = (torch.cat([original, body], dim=0) if not span else
                  torch.cat([original[:-span],
                             cls._blend_pictures(original[-span:], head[:span]),
                             body], dim=0))

        first = cls._sound(parts_o.audio, int(original.shape[0]), fmt)
        head_wave, body_wave = cls._extension_sound(
            parts_e, overlap, span, int(body.shape[0]), fmt)
        waveform = (torch.cat([first, body_wave], dim=-1) if head_wave is None
                    else cls._blend_sound(first, head_wave, body_wave))
        audio = {"waveform": waveform, "sample_rate": fmt.sample_rate}

        cls._log(parts_o, parts_e, overlap, cut, join, fmt, images, audio)
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=images, audio=audio,
            frame_rate=Fraction(fmt.fps).limit_denominator(1000)))
        return io.NodeOutput(video, int(images.shape[0]), images, audio)

    @classmethod
    def _log(cls, parts_o, parts_e, overlap, cut, join, fmt, images, audio):
        frames = int(images.shape[0])
        held = (int(audio["waveform"].shape[-1]) / fmt.sample_rate if audio
                else frames / fmt.fps)
        _LOG.info(
            "scg_h3_extension_concat: %d @%.4g + %d @%.4g - %d overlap (%d of "
            "the extension's own frames, %.4fs) -> %d frames at %dx%d @%.4g "
            "(%.4fs), %s, audio %s, drift %.2fms",
            int(parts_o.images.shape[0]), float(parts_o.frame_rate),
            int(parts_e.images.shape[0]), float(parts_e.frame_rate),
            overlap, cut, overlap / H3_FPS, frames, fmt.width, fmt.height,
            fmt.fps, frames / fmt.fps, join,
            f"{fmt.sample_rate}Hz/{fmt.channels}ch" if audio else "none",
            abs(frames / fmt.fps - held) * 1000.0)
