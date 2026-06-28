import node_helpers
import comfy.utils
import math


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
                "vision_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.01}),
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
                "reference_latents_method": (cls.REF_METHODS, {"default": "index"}),
                "instruction_template": (TEMPLATE_MODES, {"default": "compose"}),
                "custom_template": ("STRING", {"multiline": True, "default": ""}),
                "vision_megapixels": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 2.0, "step": 0.01}),
                "target_width": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "scg-utils/conditioning"

    def encode(self, clip, prompt, vision_only=False, vae=None, image=None,
               reference_latents_method="index",
               instruction_template="compose", custom_template="",
               vision_megapixels=1.0, target_width=896, target_height=896):

        ref_latents = []
        images_vl = []
        image_prompt = ""

        if image is not None:
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
