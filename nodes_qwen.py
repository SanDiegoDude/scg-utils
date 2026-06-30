import node_helpers
import comfy.utils
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mask-based salience attenuation (anti background-bleed)
#
# We can't give the Qwen2.5-VL encoder a real alpha channel, and the encoder's
# global self-attention smears masked-region info into every other token, so
# zeroing tokens after the fact does little. The effective lever is to act in
# pixel space *before* encoding: turn the masked-out region into a low-info
# "ghost" (heavy blur + desaturate + contrast crush toward mid-gray) so the
# encoder finds nothing salient there. A feathered mask edge avoids the hard
# contour that itself causes bleed. This is a weight, not a hard cut.
# ---------------------------------------------------------------------------


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur on a [B, C, H, W] tensor (reflect padded)."""
    if sigma is None or sigma <= 0.0:
        return x
    radius = max(1, int(round(sigma * 3.0)))
    ksize = radius * 2 + 1
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - radius
    k1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    c = x.shape[1]
    kx = k1d.view(1, 1, 1, ksize).repeat(c, 1, 1, 1)
    ky = k1d.view(1, 1, ksize, 1).repeat(c, 1, 1, 1)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, kx, groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, ky, groups=c)
    return x


def _low_info(img):
    """Build a low-information 'ghost' of a [B, C, H, W] image: heavily blurred,
    desaturated, and contrast-crushed toward mid-gray, but still faintly present."""
    b, c, h, w = img.shape
    sw = max(1, w // 16)
    sh = max(1, h // 16)
    down = comfy.utils.common_upscale(img, sw, sh, "area", "disabled")
    blurred = comfy.utils.common_upscale(down, w, h, "bilinear", "disabled")
    if c >= 3:
        lum = 0.299 * blurred[:, 0:1] + 0.587 * blurred[:, 1:2] + 0.114 * blurred[:, 2:3]
        lum = lum.repeat(1, c, 1, 1)
        desat = blurred * 0.3 + lum * 0.7
    else:
        desat = blurred
    crushed = 0.5 + (desat - 0.5) * 0.5
    return crushed.clamp(0.0, 1.0)


def _normalize_mask(mask, b, h, w, dtype, device, node_name):
    m = mask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m.unsqueeze(1)
    elif m.dim() == 4 and m.shape[1] != 1 and m.shape[-1] == 1:
        m = m.movedim(-1, 1)
    if m.dim() != 4 or m.shape[1] != 1:
        raise ValueError("{}: unexpected mask shape {}.".format(node_name, tuple(mask.shape)))

    mh, mw = m.shape[-2], m.shape[-1]
    if (mh, mw) != (h, w):
        raise ValueError(
            "{}: mask size {}x{} does not match image size {}x{}. Resize the mask "
            "to match the image (this node does not auto-resize).".format(node_name, mw, mh, w, h))

    mb = m.shape[0]
    if mb != b:
        if mb == 1:
            m = m.repeat(b, 1, 1, 1)
        elif b == 1:
            m = m[:1]
        else:
            raise ValueError(
                "{}: mask batch {} does not match image batch {}.".format(node_name, mb, b))
    return m.to(device=device, dtype=dtype)


def apply_mask_attenuation(image, mask, strength, feather, invert, node_name):
    """Attenuate the un-kept region of a [B, H, W, C] image toward a low-info ghost.

    Mask convention: 1.0 = keep (subject), 0.0 = attenuate (background). invert flips it.
    """
    img = image.movedim(-1, 1)
    b, c, h, w = img.shape
    m = _normalize_mask(mask, b, h, w, img.dtype, img.device, node_name)
    if invert:
        m = 1.0 - m
    if feather > 0.0:
        m = _gaussian_blur(m, float(feather)).clamp(0.0, 1.0)
    atten = ((1.0 - m) * float(strength)).clamp(0.0, 1.0)
    low = _low_info(img)
    out = img * (1.0 - atten) + low * atten
    return out.movedim(1, -1)


# Instruction templates fed to the Qwen2.5-VL text encoder. The {} is replaced
# with the user's prompt. The "system" framing strongly biases how the model
# treats the reference image(s).
EDIT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe key details of the input image (including any objects, characters, "
    "poses, facial features, clothing, setting, textures and style), then explain "
    "how the user's text instruction should alter, modify or recreate the image. "
    "Generate a new image that meets the user's requirements, which can vary from a "
    "small change to a completely new image using inputs as a guide."
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

COMPOSE_TEMPLATE = (
    "<|im_start|>system\n"
    "Analyze the key elements across all of the provided reference images "
    "(subjects, characters, poses, facial features, clothing, objects, materials, "
    "lighting, color palette and artistic style). Treat the references as "
    "complementary inputs for a single new image, not as separate pictures to "
    "display. Following the user's instruction, synthesize one coherent image that "
    "blends the relevant concepts, subjects and styles from the references into a "
    "unified composition. Do not place the references side by side or arrange them "
    "in a grid unless the user explicitly asks for that."
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)

TEMPLATE_MODES = ["compose", "edit", "custom"]


def resolve_template(instruction_template, custom_template):
    if instruction_template == "custom":
        tmpl = custom_template.strip()
        if "{}" not in tmpl:
            # Fall back to a safe wrapper if the user forgot the placeholder.
            tmpl = (
                "<|im_start|>system\n" + (tmpl or "Generate an image based on the references.") +
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
        return tmpl
    if instruction_template == "edit":
        return EDIT_TEMPLATE
    return COMPOSE_TEMPLATE


class SCGTextEncoderQwenEditPlus:
    """
    Enhanced TextEncodeQwenImageEditPlus with explicit control over how multiple
    reference images are conditioned.

    Two independent conditioning channels are produced per image:
      - vision  : the image is read by the Qwen2.5-VL text encoder (semantic /
                  concept understanding, "Picture N").
      - reference: the image is VAE-encoded and concatenated into the diffusion
                  model as a structural reference latent (kontext-style, which the
                  model tends to spatially reproduce).

    For merging concepts (rather than tiling images side-by-side) the vision
    channel carries the "idea" while reference latents pin structure. Per-image
    mode + reference_latents_method give you control over that balance.
    """

    REF_METHODS = ["index", "offset", "index_timestep_zero", "negative_index", "auto (model default)"]
    IMAGE_MODES = ["reference + vision", "vision only", "reference only", "disabled"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            },
            "optional": {
                "vision_only": ("BOOLEAN", {"default": False, "tooltip": "Force every image through the vision (VL) channel only and skip all VAE / reference-latent work. Use for vision-only models (e.g. Krea-2) where reference latents and reference_latents_method are no-ops. Overrides the per-image modes."}),
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image1_mode": (cls.IMAGE_MODES, {"default": "reference + vision"}),
                "image2": ("IMAGE",),
                "image2_mode": (cls.IMAGE_MODES, {"default": "reference + vision"}),
                "image3": ("IMAGE",),
                "image3_mode": (cls.IMAGE_MODES, {"default": "reference + vision"}),
                "image4": ("IMAGE",),
                "image4_mode": (cls.IMAGE_MODES, {"default": "reference + vision"}),
                "reference_latents_method": (cls.REF_METHODS, {"default": "index"}),
                "instruction_template": (TEMPLATE_MODES, {"default": "compose"}),
                "custom_template": ("STRING", {"multiline": True, "default": ""}),
                "vision_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.8, "step": 0.01, "tooltip": "Megapixels the image is resized to for the vision (VL) channel. Default ~1.0. Up to 8.8 (~4K) for serious detail on large images; high values are slow and VRAM-heavy."}),
                "target_width": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "scg-utils/conditioning"

    def encode(self, clip, prompt, vision_only=False, vae=None,
               image1=None, image1_mode="reference + vision",
               image2=None, image2_mode="reference + vision",
               image3=None, image3_mode="reference + vision",
               image4=None, image4_mode="reference + vision",
               reference_latents_method="index",
               instruction_template="compose",
               custom_template="",
               vision_megapixels=1.0,
               target_width=896, target_height=896):

        images = [
            (image1, image1_mode),
            (image2, image2_mode),
            (image3, image3_mode),
            (image4, image4_mode),
        ]

        ref_latents = []
        images_vl = []
        image_prompt = ""
        picture_index = 0  # sequential numbering for the vision channel

        vision_total = max(1, int(vision_megapixels * 1024 * 1024))

        for image, mode in images:
            if image is None or mode == "disabled":
                continue

            if vision_only:
                use_vision = True
                use_reference = False
            else:
                use_vision = mode in ("reference + vision", "vision only")
                use_reference = mode in ("reference + vision", "reference only")

            samples = image.movedim(-1, 1)

            if use_vision:
                scale_by = math.sqrt(vision_total / (samples.shape[3] * samples.shape[2]))
                width = max(1, round(samples.shape[3] * scale_by))
                height = max(1, round(samples.shape[2] * scale_by))
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                picture_index += 1
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(picture_index)

            if use_reference and vae is not None:
                orig_width = samples.shape[3]
                orig_height = samples.shape[2]

                # Fit within target bounds while preserving aspect ratio.
                scale_by = min(target_width / orig_width, target_height / orig_height)
                width = max(32, int(orig_width * scale_by / 32) * 32)
                height = max(32, int(orig_height * scale_by / 32) * 32)

                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        llama_template = resolve_template(instruction_template, custom_template)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            if reference_latents_method != "auto (model default)":
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})

        return (conditioning,)


class SCGReferenceTextEncoderPlus:
    """
    Single reference image text encoder for Qwen / Krea-style models.

    A streamlined, single-image variant of SCGTextEncoderQwenEditPlus. For
    multi-image work, use multiple of these into an SCG Conditioning Mixer instead
    (more efficient and far more controllable than stuffing 4 images into one
    encoder).

    Channels per image:
      - vision  : image is read by the Qwen2.5-VL text encoder (semantic concept).
      - reference: image is VAE-encoded as a structural reference latent (kontext
                  style). Inert on vision-only models (e.g. Krea-2).

    vision_only forces the vision channel and skips all VAE / reference-latent work.
    """

    REF_METHODS = ["index", "offset", "index_timestep_zero", "negative_index", "auto (model default)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            },
            "optional": {
                "vision_only": ("BOOLEAN", {"default": False, "tooltip": "Use the vision (VL) channel only and skip all VAE / reference-latent work. Use for vision-only models (e.g. Krea-2) where reference latents are no-ops."}),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "mask": ("MASK", {"tooltip": "Optional. Keeps the mask=1 region (subject) and attenuates the rest (background) into a low-info ghost before encoding, to fight background bleed. Must match the image size exactly (no auto-resize)."}),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How strongly to attenuate the background. 0 = mask ignored, 1 = full low-info ghost (blurred/desaturated, still faintly present)."}),
                "mask_feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 256.0, "step": 1.0, "tooltip": "Gaussian feather (pixels) on the mask edge. Softens the keep/attenuate boundary so the mask outline itself doesn't bleed in."}),
                "invert_mask": ("BOOLEAN", {"default": False, "tooltip": "Flip mask polarity. Off = keep mask=1 (subject). On = keep mask=0."}),
                "reference_latents_method": (cls.REF_METHODS, {"default": "index"}),
                "instruction_template": (TEMPLATE_MODES, {"default": "compose"}),
                "custom_template": ("STRING", {"multiline": True, "default": ""}),
                "vision_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.8, "step": 0.01, "tooltip": "Megapixels the image is resized to for the vision (VL) channel. Default ~1.0. Up to 8.8 (~4K) for serious detail on large images; high values are slow and VRAM-heavy."}),
                "target_width": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "scg-utils/conditioning"

    def encode(self, clip, prompt, vision_only=False, vae=None, image=None,
               mask=None, mask_strength=1.0, mask_feather=8.0, invert_mask=False,
               reference_latents_method="index",
               instruction_template="compose", custom_template="",
               vision_megapixels=1.0, target_width=896, target_height=896):

        ref_latents = []
        images_vl = []
        image_prompt = ""

        if image is not None:
            if mask is not None and mask_strength > 0.0:
                image = apply_mask_attenuation(
                    image, mask, mask_strength, mask_feather, invert_mask,
                    "SCG Reference Text Encoder Plus")
            use_reference = not vision_only
            samples = image.movedim(-1, 1)

            vision_total = max(1, int(vision_megapixels * 1024 * 1024))
            scale_by = math.sqrt(vision_total / (samples.shape[3] * samples.shape[2]))
            width = max(1, round(samples.shape[3] * scale_by))
            height = max(1, round(samples.shape[2] * scale_by))
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1))
            image_prompt = "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"

            if use_reference and vae is not None:
                orig_width = samples.shape[3]
                orig_height = samples.shape[2]

                scale_by = min(target_width / orig_width, target_height / orig_height)
                width = max(32, int(orig_width * scale_by / 32) * 32)
                height = max(32, int(orig_height * scale_by / 32) * 32)

                s = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
                ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

        llama_template = resolve_template(instruction_template, custom_template)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
            if reference_latents_method != "auto (model default)":
                conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})

        return (conditioning,)
