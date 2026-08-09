"""SCG MiniMax H3 video extension — continue an existing clip past its last frame.

Stock H3 extension hands the model one still frame as an anchor, which carries
position but not velocity: whatever was moving at the cut is re-guessed from
rest, and the soundtrack restarts. This pins a RUN of the source's tail frames
so the motion lives inside the latent, plus its tail audio placed on the new
clip's timeline so the waveform is continued rather than imitated.

The continuation mechanism is ComfyUI-H3-Motion-Context's, reached through the
copy vendored in the ComfyUI-H3-Multishot pack. This node owns ingest — frame
rate, tail selection, sizing, and building the destination latent — and then
delegates. It deliberately does not reimplement the layout math.

Source footage that H3 did not generate is out of distribution: the model
reconstructs the pinned tail approximately, so the join sits between real
pixels and model pixels and will show more of a step than a generated-to-
generated chain does. Extending H3's own output is the clean case, and passing
source_latent keeps that path free of an audio VAE round trip.
"""

import logging
import sys

import nodes
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

_LOG = logging.getLogger("scg_utils.h3_extend")

FPS = 24
PACK = "ComfyUI-H3-Multishot"


def _motion_context_class():
    """The Motion Context node class, resolved out of the loaded custom nodes.

    Not a plain import: ComfyUI registers a custom node directory under its
    full filesystem path, so there is no stable module name to import, and
    loading the file again would apply its layout patch a second time on top
    of the first.
    """
    for name, module in list(sys.modules.items()):
        if name.endswith(".h3_motion_nodes") and hasattr(module, "MiniMaxH3MotionContext"):
            return module.MiniMaxH3MotionContext
    raise RuntimeError(
        f"SCG MiniMax H3 Video Extend needs the {PACK} pack installed and "
        "loaded — its motion-context machinery is what continues the clip.")


def _retime_tail(frames, source_fps, count):
    """The last `count` frames of `frames`, resampled onto H3's 24 fps grid.

    H3 reads pinned frames as consecutive 24 fps frames regardless of where
    they came from. Handing it a 30 fps tail unchanged understates the motion
    per frame by 24/30, and the model faithfully continues at that rate, so
    the extension plays back in slow motion. Sampling at source_fps/24 source
    frames per output frame preserves real-time velocity across the join.
    """
    total = int(frames.shape[0])
    if total < 1:
        raise ValueError("SCG MiniMax H3 Video Extend: source_video is empty")
    if abs(source_fps - FPS) < 1e-3:
        return frames[max(0, total - count):]

    step = source_fps / FPS
    if (count - 1) * step > total - 1:
        _LOG.warning(
            "h3_extend: %.4g fps source holds %d frames, %d needed for a %d "
            "frame run at 24 fps; the start of the run repeats frame 0",
            source_fps, total, int((count - 1) * step) + 1, count)
    index = [min(total - 1, max(0, round((total - 1) - (count - 1 - k) * step)))
             for k in range(count)]
    return frames[index]


def _log_ingest(source, tail, source_fps, width, height, audio, audio_frames,
                source_latent):
    """Report what ingest kept; everything outside the pinned window is dropped."""
    if source_latent is not None:
        note = "audio from source_latent"
    elif audio is not None:
        held = int(audio["waveform"].shape[-1]) / max(1, int(audio["sample_rate"]))
        note = f"audio {held:.2f}s -> {audio_frames / FPS:.2f}s"
    else:
        note = "no audio"
    _LOG.info(
        "h3_extend: kept %d of %d frames, %dx%d -> %dx%d, %.4g -> %d fps; %s",
        int(tail.shape[0]), int(source.shape[0]), int(source.shape[2]),
        int(source.shape[1]), width, height, source_fps, FPS, note)


class SCGMiniMaxH3VideoExtend(io.ComfyNode):
    """Conditioning + destination latent that continue a supplied clip.

    Owns: frame-rate normalization, tail selection, destination latent, prompt
    encoding, and the trim count the delivered clip needs.
    Not responsible for: the keyframe layout, audio coordinate rewriting, or
    decoding — the first two belong to the motion context node it delegates to,
    the last to the Trim node downstream.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGMiniMaxH3VideoExtend",
            display_name="SCG MiniMax H3 Video Extend",
            category="scg-utils/conditioning",
            description=(
                "Continue an existing video past its final frame, carrying motion "
                "and audio across the join instead of restarting from a still. "
                "Feed the result to a sampler, then trim the pinned head off the "
                "decoded clip with trim_frames."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True,
                                tooltip="What happens NEXT. Describe the continuation, "
                                        "not the footage already supplied."),
                io.Image.Input("source_video",
                               tooltip="The clip being extended. Only the tail is read, "
                                       "so passing the whole thing is fine."),
                io.Float.Input("source_fps", default=24.0, min=1.0, max=240.0, step=0.01,
                               tooltip="Frame rate of source_video. H3 is 24 fps and reads "
                                       "pinned frames as 24 fps no matter their origin, so "
                                       "an unconverted 30 fps tail continues at 80% speed. "
                                       "Set this correctly and the tail is resampled."),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                             tooltip="Length of the NEW clip at 24 fps, snapped to the 17k+5 "
                                     "grid (124 = ~5s; trained range ~124-362). The pinned "
                                     "head is included and comes off with trim_frames."),
                io.Int.Input("context_length", default=22, min=1, max=39, step=1,
                             tooltip="How many tail frames to pin. The video VAE only "
                                     "distinguishes runs of 1, 5, 22 and 39; anything else "
                                     "snaps down. 22 is the tested value."),
                io.Int.Input("audio_context_length", default=22, min=5, max=96, step=1,
                             tooltip="How much tail audio to pin, in frames, end-aligned with "
                                     "the pinned video. Ignored without audio."),
                io.Combo.Input("crop", options=["center", "disabled"], default="center",
                               tooltip="How to fit source frames to width/height. 'center' "
                                       "keeps proportions when the aspect differs; 'disabled' "
                                       "stretches."),
                io.Vae.Input("audio_vae", optional=True,
                             tooltip="H3 audio VAE. Required to continue source_audio."),
                io.Audio.Input("source_audio", optional=True,
                               tooltip="Soundtrack of source_video. Its tail is continued on "
                                       "the new clip's timeline."),
                io.Latent.Input("source_latent", optional=True,
                                tooltip="The source clip's H3 latent, when H3 generated it. "
                                        "Takes the audio tail straight from the latent, "
                                        "skipping an audio VAE round trip. Overrides "
                                        "source_audio."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(),
                io.Int.Output(display_name="trim_frames"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, source_video, source_fps, width, height,
                length, context_length, audio_context_length, crop,
                audio_vae=None, source_audio=None, source_latent=None) -> io.NodeOutput:
        motion_context = _motion_context_class()()
        latent, frame_count = _empty_av_latent(width, height, length)
        if context_length >= frame_count:
            raise ValueError(
                f"SCG MiniMax H3 Video Extend: pinning {context_length} frames into a "
                f"{frame_count} frame clip leaves nothing to generate. Raise length "
                "or lower context_length.")

        tail = _retime_tail(source_video, source_fps, context_length)
        _log_ingest(source_video, tail, source_fps, width, height, source_audio,
                    audio_context_length, source_latent)
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        cond, trim_frames = motion_context.apply(
            conditioning=cond,
            vae=vae,
            latent=latent,
            context_frames=tail,
            context_length=context_length,
            encode_mode="video",
            anchor_mode="head",
            crop=crop,
            audio_context_length=audio_context_length,
            audio_mode="timeline",
            context_latent=source_latent,
            audio_vae=audio_vae,
            context_audio=source_audio,
        )
        return io.NodeOutput(cond, latent, trim_frames)
