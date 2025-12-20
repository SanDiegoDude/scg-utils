import node_helpers
import comfy.utils
import math


class SCGTextEncoderQwenEditPlus:
    """
    Fixed version of TextEncodeQwenImageEditPlus node.
    Encodes text prompts with Qwen model with support for up to 4 reference images.
    Reference images are VAE-encoded at target_width x target_height resolution (preserving aspect ratio).
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            },
            "optional": {
                "vae": ("VAE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "target_width": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
                "target_height": ("INT", {"default": 896, "min": 128, "max": 2048, "step": 32}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "scg-utils/conditioning"

    def encode(self, clip, prompt, vae=None, image1=None, image2=None, image3=None, image4=None, target_width=896, target_height=896):
        ref_latents = []
        images = [image1, image2, image3, image4]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe key details of the input image (including any objects, characters, poses, facial features, clothing, setting, textures and style), then explain how the user's text instruction should alter, modify or recreate the image. Generate a new image that meets the user's requirements, which can vary from a small change to a completely new image using inputs as a guide.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    # Calculate scale factor to fit within target dimensions while preserving aspect ratio
                    orig_width = samples.shape[3]
                    orig_height = samples.shape[2]
                    
                    # Scale to fit within target bounds
                    width_scale = target_width / orig_width
                    height_scale = target_height / orig_height
                    scale_by = min(width_scale, height_scale)
                    
                    # Calculate final dimensions rounded to 32
                    width = int(orig_width * scale_by / 32) * 32
                    height = int(orig_height * scale_by / 32) * 32
                    
                    s = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")                    
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        return (conditioning,)
