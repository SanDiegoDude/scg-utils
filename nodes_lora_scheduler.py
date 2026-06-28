import logging
import uuid

import torch

import folder_paths
import comfy.utils
import comfy.lora
import comfy.lora_convert
import comfy.weight_adapter


# ---------------------------------------------------------------------------
# SCG LoRA Scheduler
#
# Tapers a LoRA's *model* (unet) strength across the denoise trajectory, the
# same way SCG Conditioning Mixer tapers conditioning: a strength_start ->
# strength_end ramp inside a clean start_percent / end_percent window.
#
# Implementation: bypass-mode LoRA injection.
#   ComfyUI normally bakes a LoRA into the model weights, which cannot change
#   per step, and the hook-based weight-swap path is incompatible with dynamic
#   / quantized (fp8) VRAM loading -- it walks the whole state dict and trips on
#   keys like `weight_scale` that aren't live module attributes.
#
#   Instead we use ComfyUI's bypass adapters (comfy.weight_adapter): the LoRA is
#   applied additively *inside each layer's forward* (out += scale * up(down(x))),
#   never touching base weights. Each adapter exposes a live `multiplier`, so a
#   small unet function wrapper updates that multiplier every step based on the
#   current sigma -> giving a true per-step taper that works on quantized /
#   dynamically loaded models (Krea2, fp8, etc.).
#
# Notes:
#   * CLIP is optional and applied statically at strength_clip (CLIP encodes once
#     and cannot be scheduled).
#   * Bypass mode adds a small per-step forward cost (the low-rank matmuls), but
#     avoids all weight re-baking.
#   * Stacking multiple SCG LoRA Schedulers works: each owns its own injection
#     key and updates only its own adapters, chaining the unet wrapper.
#   * Non-adapter LoRA entries (full `diff`/`set` weights, rare) cannot be
#     scheduled in bypass mode; they are applied statically at strength_start
#     with a warning.
# ---------------------------------------------------------------------------


_LORA_CACHE = {}


def _load_lora_file(lora_name):
    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    cached = _LORA_CACHE.get(lora_path, None)
    if cached is not None:
        return cached
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    lora = comfy.lora_convert.convert_lora(lora)
    _LORA_CACHE[lora_path] = lora
    return lora


def _ease(t, method):
    """Apply an easing curve to t in [0, 1]."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    if method == "ease_in":
        return t * t
    if method == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if method == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return t  # linear


def _make_taper(strength_start, strength_end, start_percent, end_percent,
                interpolation, cutoff_outside_window):
    start_percent = max(0.0, min(1.0, float(start_percent)))
    end_percent = max(0.0, min(1.0, float(end_percent)))
    if end_percent < start_percent:
        start_percent, end_percent = end_percent, start_percent
    span = end_percent - start_percent

    def taper(percent):
        if percent < start_percent:
            return 0.0 if cutoff_outside_window else strength_start
        if percent > end_percent:
            return 0.0 if cutoff_outside_window else strength_end
        if span <= 1e-9:
            return strength_end
        t = _ease((percent - start_percent) / span, interpolation)
        return strength_start + (strength_end - strength_start) * t

    return taper


def _sigma_to_percent(sigmas, timestep):
    """Map the current sigma to a 0..1 position in the sampling schedule."""
    s = sigmas.reshape(-1)
    n = int(s.shape[0])
    if n <= 1:
        return 0.0
    cur = float(timestep.reshape(-1)[0])
    idx = int(torch.argmin(torch.abs(s.float() - cur)).item())
    return idx / (n - 1)


def _make_scheduler_wrapper(adapters, taper, prev_wrapper):
    def wrapper(model_function, kwargs):
        input_x = kwargs["input"]
        timestep = kwargs["timestep"]
        c = kwargs["c"]

        def call_inner():
            if prev_wrapper is not None:
                return prev_wrapper(model_function, kwargs)
            return model_function(input_x, timestep, **c)

        try:
            transformer_options = c.get("transformer_options", {}) if isinstance(c, dict) else {}
            sigmas = transformer_options.get("sample_sigmas", None)
            if sigmas is None or not adapters:
                return call_inner()
            strength = taper(_sigma_to_percent(sigmas, timestep))
            for adapter in adapters:
                adapter.multiplier = strength
        except Exception:
            logging.exception("SCG LoRA Scheduler: failed to update strength, using last value")
        return call_inner()

    return wrapper


def _build_bypass(model_patcher, lora):
    """Build bypass adapters for the model. Returns (adapters, injections, regular_patches)."""
    key_map = comfy.lora.model_lora_keys_unet(model_patcher.model, {})
    loaded = comfy.lora.load_lora(lora, key_map)

    manager = comfy.weight_adapter.BypassInjectionManager()
    model_sd_keys = set(model_patcher.model.state_dict().keys())
    regular_patches = {}
    for key, patch_data in loaded.items():
        if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase):
            if key in model_sd_keys:
                manager.add_adapter(key, patch_data, strength=1.0)
        else:
            regular_patches[key] = patch_data

    injections = manager.create_injections(model_patcher.model)
    adapters = [hook.adapter for hook in manager.hooks]
    return adapters, injections, regular_patches


class SCGLoRAScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Diffusion model to apply the scheduled LoRA to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "LoRA file to schedule."}),
                "strength_start": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                             "tooltip": "Model LoRA strength at start_percent."}),
                "strength_end": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                           "tooltip": "Model LoRA strength at end_percent."}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                            "tooltip": "When the LoRA begins (0 = first step)."}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                          "tooltip": "When the LoRA ends (1 = last step)."}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear",
                                  "tooltip": "Shape of the strength taper between start and end."}),
                "cutoff_outside_window": ("BOOLEAN", {"default": True,
                                                      "tooltip": "Hold the LoRA at 0 before start_percent and after end_percent. Off = hold the start/end strength outside the window."}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Optional CLIP. The LoRA is applied to CLIP statically at strength_clip (CLIP runs once, so it cannot be scheduled)."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                            "tooltip": "Static LoRA strength for CLIP (only used when CLIP is connected)."}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    OUTPUT_TOOLTIPS = ("Model with the time-scheduled LoRA.", "CLIP with the (static) LoRA applied, or passthrough if not connected.")
    FUNCTION = "schedule"
    CATEGORY = "scg-utils/model"
    EXPERIMENTAL = True
    DESCRIPTION = ("Tapers a LoRA's model strength across the denoise trajectory (strength_start -> strength_end "
                   "inside a start_percent/end_percent window), the same way SCG Conditioning Mixer tapers "
                   "conditioning. Uses bypass-mode injection so it works on quantized / dynamically loaded "
                   "models. CLIP is optional and applied statically.")

    def schedule(self, model, lora_name, strength_start, strength_end, start_percent, end_percent,
                 interpolation, cutoff_outside_window, clip=None, strength_clip=1.0):
        lora = _load_lora_file(lora_name)
        inst_key = "scg_lora_scheduler_{}".format(uuid.uuid4().hex[:8])

        # CLIP: static bypass application (CLIP encodes once, nothing to schedule).
        out_clip = clip
        if clip is not None and strength_clip != 0.0:
            try:
                out_clip = clip.clone()
                ckey_map = comfy.lora.model_lora_keys_clip(out_clip.cond_stage_model, {})
                cloaded = comfy.lora.load_lora(lora, ckey_map)
                cmanager = comfy.weight_adapter.BypassInjectionManager()
                csd = set(out_clip.cond_stage_model.state_dict().keys())
                cadded = 0
                for key, patch_data in cloaded.items():
                    if isinstance(patch_data, comfy.weight_adapter.WeightAdapterBase) and key in csd:
                        cmanager.add_adapter(key, patch_data, strength=strength_clip)
                        cadded += 1
                if cadded > 0:
                    cinj = cmanager.create_injections(out_clip.cond_stage_model)
                    out_clip.patcher.set_injections(inst_key + "_clip", cinj)
            except Exception:
                logging.exception("SCG LoRA Scheduler: CLIP LoRA application failed; passing CLIP through")
                out_clip = clip

        # If there is no model effect at all, skip the model side entirely.
        if strength_start == 0.0 and strength_end == 0.0:
            return (model, out_clip)

        m = model.clone()
        adapters, injections, regular_patches = _build_bypass(m, lora)

        if regular_patches:
            logging.warning(
                "SCG LoRA Scheduler: %d non-adapter LoRA entries cannot be scheduled in bypass mode; "
                "applying them statically at strength_start=%s.", len(regular_patches), strength_start)
            m.add_patches(regular_patches, strength_start)

        if not adapters and not regular_patches:
            logging.warning("SCG LoRA Scheduler: no LoRA keys matched this model; returning model unchanged.")
            return (m, out_clip)

        if injections:
            m.set_injections(inst_key, injections)

        taper = _make_taper(strength_start, strength_end, start_percent, end_percent,
                            interpolation, cutoff_outside_window)
        prev_wrapper = m.model_options.get("model_function_wrapper", None)
        m.set_model_unet_function_wrapper(_make_scheduler_wrapper(adapters, taper, prev_wrapper))

        return (m, out_clip)
