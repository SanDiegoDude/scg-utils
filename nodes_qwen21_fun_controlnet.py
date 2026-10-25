"""SCG Qwen-Image 2.1 Fun ControlNet-Union (alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union).

Runs the 16 VACE-style control blocks in parallel with the base model, adding a
skip after every second base block (0, 2, ..., 30), as in VideoX-Fun. Loads the
original checkpoint or one with the MLP gate/up weights fused into `gate_up`.
"""

import inspect

import torch
import torch.nn.functional as F

import folder_paths
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch
import comfy.ops
import comfy.patcher_extension
import comfy.utils
from comfy.ldm.qwen_image21.model import QwenImage21TransformerBlock
from comfy_api.latest import io


class ControlBlock(QwenImage21TransformerBlock):
    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=3, fused_mlp=True, first_block=False, dtype=None, device=None, operations=None):
        super().__init__(dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, fused_mlp=fused_mlp, dtype=dtype, device=device, operations=operations)
        if first_block:
            self.before_proj = operations.Linear(dim, dim, dtype=dtype, device=device)
        self.after_proj = operations.Linear(dim, dim, dtype=dtype, device=device)


class QwenImage21FunControl(torch.nn.Module):
    def __init__(self, control_in_dim=129, num_blocks=16, inner_dim=4096, num_attention_heads=32, attention_head_dim=128, mlp_ratio=3, fused_mlp=True, dtype=None, device=None, operations=None):
        super().__init__()
        self.dtype = dtype
        self.injection_layers = tuple(range(0, 2 * num_blocks, 2))
        self.control_img_in = operations.Linear(control_in_dim, inner_dim, dtype=dtype, device=device)
        self.control_blocks = torch.nn.ModuleList([
            ControlBlock(inner_dim, num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, fused_mlp=fused_mlp, first_block=(i == 0), dtype=dtype, device=device, operations=operations)
            for i in range(num_blocks)])

    def init_stream(self, x, control_tokens):
        # control rows on the target image tokens (last in the sequence), zero on text and reference tokens
        c = torch.zeros_like(x)
        c[:, -control_tokens.shape[1]:] = self.control_img_in(control_tokens)
        return self.control_blocks[0].before_proj(c).add_(x)

    def step(self, index, c, mod, pe, attn_fn, prefix_len, transformer_options):
        block = self.control_blocks[index]
        c = block(c, mod, pe, attn_fn, prefix_len, transformer_options)
        return c, block.after_proj(c)


def is_qwen_image21_fun_state_dict(state_dict):
    required = (
        "control_img_in.weight",
        "control_blocks.0.before_proj.weight",
        "control_blocks.0.after_proj.weight",
        "control_blocks.0.attn.norm_q.weight",
        "control_blocks.0.img_mlp.out.weight",
    )
    return all(key in state_dict for key in required)


def block_state(args, extra_args):
    # core only passes img/vec/pe to block replace patches; the rest is bound in its original_block wrapper
    if "mod" in args:
        return args["mod"], args["attn_fn"], args["prefix_len"]
    state = inspect.getclosurevars(extra_args["original_block"]).nonlocals
    if not all(k in state for k in ("mod", "attn_fn", "prefix_len")):
        raise RuntimeError("SCG Qwen Image 2.1 Fun ControlNet: this ComfyUI version changed the Qwen Image 2.1 block wrapper, update scg-utils.")
    return state["mod"], state["attn_fn"], state["prefix_len"]


class QwenImage21FunControlPatch:
    def __init__(self, model_patch, vae, latent_format, control_image, mask, inpaint_image, strength, sigma_start, sigma_end):
        self.model_patch = model_patch
        self.vae = vae
        self.latent_format = latent_format
        self.control_image = control_image
        self.mask = mask
        self.inpaint_image = inpaint_image
        self.strength = strength
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.control_tokens = None
        self.control_shape = None
        self.control_stream = None
        self.pristine_stream = None
        self.active = False

    def _encode(self, image, width, height):
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        return self.latent_format.process_in(self.vae.encode(image).float())

    def prepare_control(self, latent_height, latent_width):
        if self.control_tokens is not None and self.control_shape == (latent_height, latent_width):
            return
        spatial_compression = self.vae.spacial_compression_encode()
        width, height = latent_width * spatial_compression, latent_height * spatial_compression
        device = comfy.model_management.intermediate_device()
        loaded_models = comfy.model_management.loaded_models(only_currently_used=True)
        try:
            # [control latent, mask (1 = keep), masked inpaint latent], zero where not given
            zeros = torch.zeros(1, self.latent_format.latent_channels, latent_height, latent_width, device=device)
            control = zeros if self.control_image is None else self._encode(self.control_image, width, height).to(device)
            keep = torch.zeros(1, 1, latent_height, latent_width, device=device)
            masked = zeros
            if self.mask is not None:
                regen = comfy.utils.common_upscale(self.mask.reshape(-1, 1, *self.mask.shape[-2:]).float(), width, height, "bilinear", "center").to(device) >= 0.5
                keep = F.interpolate((~regen).float(), size=(latent_height, latent_width), mode="nearest")
                if self.inpaint_image is not None:
                    source = comfy.utils.common_upscale(self.inpaint_image.movedim(-1, 1), width, height, "bilinear", "center").to(device)
                    masked = self._encode(torch.where(regen, 0.5, source).movedim(1, -1), width, height).to(device)
        finally:
            comfy.model_management.load_models_gpu(loaded_models)

        batch = max(control.shape[0], keep.shape[0], masked.shape[0])
        hint = torch.cat([comfy.utils.repeat_to_batch_size(t, batch) for t in (control, keep, masked)], dim=1)
        self.control_tokens = hint.flatten(2).transpose(1, 2)
        self.control_shape = (latent_height, latent_width)

    def diffusion_model_wrapper(self, executor, x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs):
        sigmas = transformer_options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0])
        self.active = self.sigma_end <= sigma <= self.sigma_start
        self.control_stream = None
        if self.active:
            with comfy.model_prefetch.pause_malloc_graph():
                self.prepare_control(x.shape[-2], x.shape[-1])
        try:
            return executor(x, timestep, context, ref_latents, image_slots, transformer_options, **kwargs)
        finally:
            self.control_stream = None
            self.pristine_stream = None

    def before_block(self, block_index, args):
        if not self.active or block_index != self.model_patch.model.injection_layers[0]:
            return
        # base blocks update their input in place
        self.pristine_stream = args["img"].clone()

    def after_block(self, block_index, args, extra_args, out):
        if not self.active:
            return out
        mod, attn_fn, prefix_len = block_state(args, extra_args)
        control_index = self.model_patch.model.injection_layers.index(block_index)
        if control_index == 0:
            img = out["img"]
            tokens = comfy.utils.repeat_to_batch_size(self.control_tokens, img.shape[0]).to(device=img.device, dtype=img.dtype)
            self.control_stream = self.model_patch.model.init_stream(self.pristine_stream, tokens)
            self.pristine_stream = None
        self.control_stream, skip = self.model_patch.model.step(
            control_index, self.control_stream, mod, args["pe"], attn_fn, prefix_len, args["transformer_options"])
        out["img"].add_(skip, alpha=self.strength)
        return out

    def to(self, device_or_dtype):
        self.control_stream = None
        return self

    def cleanup(self):
        self.control_tokens = None
        self.control_shape = None
        self.control_stream = None
        self.pristine_stream = None
        self.active = False

    def models(self):
        return [self.model_patch]

    def register(self, model):
        model.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, self.diffusion_model_wrapper)
        blocks_replace = model.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for block_index in self.model_patch.model.injection_layers:
            previous = blocks_replace.get(("single_block", block_index))
            model.set_model_patch_replace(QwenImage21FunControlBlockPatch(self, block_index, previous), "dit", "single_block", block_index)


class QwenImage21FunControlBlockPatch:
    def __init__(self, control_patch, block_index, previous):
        self.control_patch = control_patch
        self.block_index = block_index
        self.previous = previous

    def __call__(self, args, extra_args):
        # control state must stay outside the base block's allocation scope
        with comfy.model_prefetch.pause_malloc_graph():
            self.control_patch.before_block(self.block_index, args)
        if self.previous is None:
            out = extra_args["original_block"](args)
        else:
            out = self.previous(args, extra_args)
        with comfy.model_prefetch.pause_malloc_graph():
            return self.control_patch.after_block(self.block_index, args, extra_args, out)

    def to(self, device_or_dtype):
        self.control_patch.to(device_or_dtype)
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def cleanup(self):
        self.control_patch.cleanup()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()

    def models(self):
        models = self.control_patch.models()
        if hasattr(self.previous, "models"):
            models += self.previous.models()
        return models


class SCGLoadQwenImage21FunControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGLoadQwenImage21FunControlNet",
            display_name="SCG Load Qwen Image 2.1 Fun ControlNet",
            category="scg-utils/loaders",
            description="Loads a Qwen-Image 2.1 Fun ControlNet-Union checkpoint from models/model_patches.",
            inputs=[
                io.Combo.Input("name", options=folder_paths.get_filename_list("model_patches")),
            ],
            outputs=[io.ModelPatch.Output()],
        )

    @classmethod
    def execute(cls, name) -> io.NodeOutput:
        path = folder_paths.get_full_path_or_raise("model_patches", name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        if not is_qwen_image21_fun_state_dict(sd):
            raise ValueError("{} is not a Qwen-Image 2.1 Fun ControlNet checkpoint.".format(name))
        num_blocks = 0
        while "control_blocks.{}.after_proj.weight".format(num_blocks) in sd:
            num_blocks += 1
        inner_dim = sd["control_img_in.weight"].shape[0]
        head_dim = sd["control_blocks.0.attn.norm_q.weight"].shape[0]
        model = QwenImage21FunControl(
            control_in_dim=sd["control_img_in.weight"].shape[1],
            num_blocks=num_blocks,
            inner_dim=inner_dim,
            num_attention_heads=inner_dim // head_dim,
            attention_head_dim=head_dim,
            mlp_ratio=sd["control_blocks.0.img_mlp.out.weight"].shape[1] // inner_dim,
            fused_mlp="control_blocks.0.img_mlp.gate_up.weight" in sd,
            device=comfy.model_management.unet_offload_device(),
            dtype=comfy.utils.weight_dtype(sd),
            operations=comfy.ops.manual_cast)
        model_patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=comfy.model_management.get_torch_device(), offload_device=comfy.model_management.unet_offload_device())
        model.load_state_dict(sd, assign=model_patcher.is_dynamic())
        return io.NodeOutput(model_patcher)


class SCGApplyQwenImage21FunControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGApplyQwenImage21FunControlNet",
            display_name="SCG Apply Qwen Image 2.1 Fun ControlNet",
            category="scg-utils/model_patches",
            search_aliases=["qwen controlnet", "qwen image 2.1 controlnet", "qwen inpaint"],
            inputs=[
                io.Model.Input("model"),
                io.ModelPatch.Input("model_patch"),
                io.Vae.Input("vae"),
                io.Float.Input("strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True),
                io.Image.Input("control_image", optional=True, tooltip="Canny, depth, grayscale, HED, lineart, MLSD, pose or scribble image."),
                io.Mask.Input("mask", optional=True, tooltip="1 marks the regions to regenerate."),
                io.Image.Input("inpaint_image", optional=True, tooltip="Image behind the mask; only read when a mask is given."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, strength, start_percent, end_percent, control_image=None, mask=None, inpaint_image=None) -> io.NodeOutput:
        if not isinstance(model_patch.model, QwenImage21FunControl):
            raise ValueError("Connect a model patch from SCG Load Qwen Image 2.1 Fun ControlNet.")
        if strength == 0 or (control_image is None and mask is None):
            return io.NodeOutput(model)

        model_patched = model.clone()
        model_sampling = model.get_model_object("model_sampling")
        patch = QwenImage21FunControlPatch(
            model_patch,
            vae,
            model.get_model_object("latent_format"),
            control_image[..., :3] if control_image is not None else None,
            mask,
            inpaint_image[..., :3] if mask is not None and inpaint_image is not None else None,
            strength,
            float(model_sampling.percent_to_sigma(start_percent)),
            float(model_sampling.percent_to_sigma(end_percent)),
        )
        patch.register(model_patched)
        return io.NodeOutput(model_patched)
