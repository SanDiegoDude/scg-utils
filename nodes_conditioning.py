import node_helpers
import torch

from .nodes_qwen import resolve_template, TEMPLATE_MODES


# --- low-level conditioning ops (mirror ComfyUI's stock nodes exactly) ---

def cond_zero_out(conditioning):
    """Replicates ConditioningZeroOut: zero the cross-attn tensor + pooled output."""
    out = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        out.append([torch.zeros_like(t[0]), d])
    return out


def cond_average(conditioning_to, conditioning_from, to_strength):
    """Replicates ConditioningAverage: tw = to*s + from*(1-s) (uses first 'from' cond)."""
    out = []
    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        t0 = cond_from[:, :t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

        tw = torch.mul(t1, to_strength) + torch.mul(t0, (1.0 - to_strength))
        t_to = conditioning_to[i][1].copy()
        if pooled_output_from is not None and pooled_output_to is not None:
            t_to["pooled_output"] = torch.mul(pooled_output_to, to_strength) + torch.mul(pooled_output_from, (1.0 - to_strength))
        elif pooled_output_from is not None:
            t_to["pooled_output"] = pooled_output_from

        out.append([tw, t_to])
    return out


def cond_set_range(conditioning, start, end):
    start = max(0.0, min(1.0, float(start)))
    end = max(0.0, min(1.0, float(end)))
    return node_helpers.conditioning_set_values(conditioning, {"start_percent": start, "end_percent": end})


def cond_dampen(conditioning, strength):
    """Scale a cond's contribution by averaging it toward its own zeroed version.

    strength == 1.0 is a passthrough, 0.0 silences it, and values above 1.0
    amplify (cond_average with a zeroed 'from' computes cond * strength).
    """
    if abs(strength - 1.0) < 1e-6:
        return list(conditioning)
    if strength <= 0.0:
        return cond_zero_out(conditioning)
    return cond_average(conditioning, cond_zero_out(conditioning), strength)


def prep_source(conditioning, strength, start, end, taper_mode, taper_target, taper_steps):
    """
    Apply strength (dampen toward zero), an optional strength taper across the
    timestep window, and the timestep range itself. Returns a list of cond
    entries (a union when tapering, since each sub-window is a separate gated cond).

    taper_mode:
      'off'     - flat 'strength' across [start, end].
      'normal'  - ramp strength -> taper_target across [start, end] (max to target).
      'reverse' - ramp taper_target -> strength across [start, end] (target to max).
    """
    if conditioning is None:
        return []

    start = max(0.0, min(1.0, float(start)))
    end = max(0.0, min(1.0, float(end)))
    if end <= start:
        return []

    if taper_mode == "off" or taper_steps < 2 or abs(taper_target - strength) < 1e-6:
        return list(cond_set_range(cond_dampen(conditioning, strength), start, end))

    out = []
    span = end - start
    for k in range(taper_steps):
        seg_start = start + span * (k / taper_steps)
        seg_end = start + span * ((k + 1) / taper_steps)
        frac = (k + 0.5) / taper_steps
        if taper_mode == "normal":
            s = strength + (taper_target - strength) * frac
        else:  # reverse
            s = taper_target + (strength - taper_target) * frac
        seg = cond_set_range(cond_dampen(conditioning, s), seg_start, seg_end)
        out += list(seg)
    return out


def cond_merge(cond_a, cond_b, style, strength):
    """style 'average' -> weighted blend (strength = weight of cond_a); 'combine' -> union."""
    if style == "combine":
        return list(cond_a) + list(cond_b)
    return cond_average(cond_a, cond_b, strength)


class SCGConditioningTrajectory:
    """
    Conditioning trajectory mixer for Qwen / Krea-style vision conditioning.

    Collapses a tangle of ConditioningAverage / ConditioningZeroOut /
    ConditioningSetTimestepRange / ConditioningCombine nodes into one box, and
    adds per-source strength tapering across the denoise trajectory.

    For each source (A, the optional B, and an optional text-only cond it can
    encode itself):
      - strength    : scales the cond by averaging it toward a zeroed cond
                      (the idiomatic ComfyUI "cond weight" trick).
      - start/end   : timestep range the cond is active over (0.0 = first step /
                      high sigma, 1.0 = last step).
      - taper/target: optionally ramp strength across the window between 'strength'
                      (the max) and 'taper_target'. 'normal' = max->target,
                      'reverse' = target->max. Honors start/end and strength.

    A and B are then merged (average = weighted blend, combine = union). The
    optional text-only cond is merged into that result with its own controls.

    Works with one, or zero, conditioning inputs:
      - one input: strengths/tapers it and optionally mixes in text.
      - zero inputs: behaves as a fancy text encoder - it encodes the text prompt
        (even if empty, like a normal text encoder) and applies the text strength /
        taper / start-stop. include_text_only and merge settings are ignored in
        this mode since there is nothing to merge with.

    Notes:
      - 'average' fuses tensors into one cond inheriting the first input's
        metadata; use 'combine' to keep both timestep windows independent.
      - Tapering emits a union of gated sub-windows, so it pairs most cleanly with
        'combine'. With 'average' the tapered side blends against the other's
        first window only.
      - When your image-conditioning prompts are blank and you want to inject an
        enhanced prompt, use text merge style 'combine' (it adds the text rather
        than blending the images away) and drive presence with text_strength.
    """

    MERGE_STYLES = ["average", "combine"]
    TAPER_MODES = ["off", "normal", "reverse"]

    @classmethod
    def INPUT_TYPES(cls):
        pct = {"min": 0.0, "max": 1.0, "step": 0.001}
        amt = {"min": 0.0, "max": 1.0, "step": 0.01}
        # Source strengths can amplify up to 2x (great for transfer); merge
        # weights stay 0..1 since they're A-vs-B / text blend proportions.
        amp = {"min": 0.0, "max": 2.0, "step": 0.01}
        return {
            "required": {},
            "optional": {
                "conditioning_a": ("CONDITIONING",),
                "a_strength": ("FLOAT", dict(amp, default=1.0, tooltip="Scales A toward zeroed conditioning (1.0 = full, 0.0 = silent, up to 2.0 = amplified). Also the 'max' endpoint when tapering.")),
                "a_start": ("FLOAT", dict(pct, default=0.0)),
                "a_end": ("FLOAT", dict(pct, default=1.0)),
                "a_taper": (cls.TAPER_MODES, {"default": "off", "tooltip": "Ramp A's strength across its window. normal = strength->target, reverse = target->strength."}),
                "a_taper_target": ("FLOAT", dict(amp, default=0.0, tooltip="The other endpoint of A's strength taper.")),

                "conditioning_b": ("CONDITIONING",),
                "b_strength": ("FLOAT", dict(amp, default=1.0, tooltip="Scales B toward zeroed conditioning (1.0 = full, 0.0 = silent, up to 2.0 = amplified). Also the 'max' endpoint when tapering.")),
                "b_start": ("FLOAT", dict(pct, default=0.0)),
                "b_end": ("FLOAT", dict(pct, default=1.0)),
                "b_taper": (cls.TAPER_MODES, {"default": "off", "tooltip": "Ramp B's strength across its window. normal = strength->target, reverse = target->strength."}),
                "b_taper_target": ("FLOAT", dict(amp, default=0.0, tooltip="The other endpoint of B's strength taper.")),

                "merge_style": (cls.MERGE_STYLES, {"default": "average", "tooltip": "average = weighted blend of A/B into one cond; combine = union (both applied)."}),
                "merge_strength": ("FLOAT", dict(amt, default=0.5, tooltip="Average mode only: weight of A vs B (1.0 = all A, 0.0 = all B). Ignored when combine.")),
                "taper_steps": ("INT", {"default": 8, "min": 2, "max": 64, "step": 1, "tooltip": "How many sub-windows a taper is sliced into (smoothness). Shared by A/B/text."}),

                "include_text_only": ("BOOLEAN", {"default": False, "tooltip": "Encode a text-only (no image) cond from clip + text_prompt and merge it in."}),
                "clip": ("CLIP",),
                "text_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "text_template": (TEMPLATE_MODES, {"default": "compose"}),
                "text_custom_template": ("STRING", {"multiline": True, "default": ""}),
                "text_strength": ("FLOAT", dict(amp, default=1.0, tooltip="Scales the text-only cond toward zero (presence knob, esp. in combine mode). Up to 2.0 = amplified.")),
                "text_start": ("FLOAT", dict(pct, default=0.0)),
                "text_end": ("FLOAT", dict(pct, default=1.0)),
                "text_taper": (cls.TAPER_MODES, {"default": "off", "tooltip": "Ramp the text cond's strength across its window."}),
                "text_taper_target": ("FLOAT", dict(amp, default=0.0, tooltip="The other endpoint of the text strength taper.")),
                "text_merge_style": (cls.MERGE_STYLES, {"default": "combine", "tooltip": "combine = ADD the text as an extra prompt (best when image prompts are blank); average = blend it into the A/B mix."}),
                "text_merge_strength": ("FLOAT", dict(amt, default=0.5, tooltip="Average mode only: amount of text mixed in (1.0 = all text, 0.0 = none). Ignored when combine.")),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "mix"
    CATEGORY = "scg-utils/conditioning"

    @staticmethod
    def _encode_text(clip, text_prompt, text_template, text_custom_template,
                     strength, start, end, taper, taper_target, taper_steps):
        llama_template = resolve_template(text_template, text_custom_template)
        tokens = clip.tokenize(text_prompt, images=[], llama_template=llama_template)
        text_cond = clip.encode_from_tokens_scheduled(tokens)
        return prep_source(text_cond, strength, start, end, taper, taper_target, taper_steps)

    def mix(self, conditioning_a=None,
            a_strength=1.0, a_start=0.0, a_end=1.0, a_taper="off", a_taper_target=0.0,
            conditioning_b=None,
            b_strength=1.0, b_start=0.0, b_end=1.0, b_taper="off", b_taper_target=0.0,
            merge_style="average", merge_strength=0.5, taper_steps=8,
            include_text_only=False, clip=None, text_prompt="",
            text_template="compose", text_custom_template="",
            text_strength=1.0, text_start=0.0, text_end=1.0,
            text_taper="off", text_taper_target=0.0,
            text_merge_style="combine", text_merge_strength=0.5):

        a = prep_source(conditioning_a, a_strength, a_start, a_end, a_taper, a_taper_target, taper_steps)
        b = prep_source(conditioning_b, b_strength, b_start, b_end, b_taper, b_taper_target, taper_steps)
        has_inputs = len(a) > 0 or len(b) > 0

        if not has_inputs:
            # Text-only mode: act as a fancy text encoder (taper + start/stop).
            # Encodes even an empty prompt, like a normal text encoder node.
            if clip is not None:
                return (self._encode_text(clip, text_prompt, text_template, text_custom_template,
                                          text_strength, text_start, text_end,
                                          text_taper, text_taper_target, taper_steps),)
            return ([],)

        if len(a) > 0 and len(b) > 0:
            result = cond_merge(a, b, merge_style, merge_strength)
        elif len(a) > 0:
            result = a
        else:
            result = b

        if include_text_only and clip is not None and text_prompt.strip() != "":
            text_prepped = self._encode_text(clip, text_prompt, text_template, text_custom_template,
                                             text_strength, text_start, text_end,
                                             text_taper, text_taper_target, taper_steps)
            if len(text_prepped) > 0:
                if text_merge_style == "combine":
                    result = list(result) + list(text_prepped)
                else:
                    # average: text_merge_strength = amount of text (intuitive direction).
                    result = cond_average(result, text_prepped, 1.0 - text_merge_strength)

        # Degenerate guard: never hand the sampler an empty conditioning.
        if len(result) == 0:
            if conditioning_a is not None:
                result = cond_zero_out(conditioning_a)
            elif conditioning_b is not None:
                result = cond_zero_out(conditioning_b)

        return (result,)
