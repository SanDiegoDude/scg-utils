import torch


class RAAG_ModelPatch:
    """
    RAAG: Ratio Aware Adaptive Guidance
    Patches MODEL.model_options["sampler_cfg_function"] so ComfyUI uses RAAG guidance
    during sampling (SamplerCustomAdvanced / KSampler).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable": ("BOOLEAN", {"default": True}),
                "alpha": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 200.0, "step": 0.1}),
                "w_max": ("FLOAT", {"default": 18.0, "min": 1.0, "max": 200.0, "step": 0.1}),
                "eps": ("FLOAT", {"default": 1e-8, "min": 0.0, "max": 1e-2, "step": 1e-8}),
                "only_when_cfg_gt_1": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "scg-utils/sampling"

    def patch(self, model, enable, alpha, w_max, eps, only_when_cfg_gt_1):
        if not enable:
            return (model,)

        m = model.clone()

        # Store as python floats (avoid capturing ComfyUI tensors/objects)
        alpha_f = float(alpha)
        wmax_f = float(w_max)
        eps_f = float(eps)
        gate_cfg = bool(only_when_cfg_gt_1)

        def raag_cfg_function(args):
            """
            ComfyUI (your build) calls:
              model_options["sampler_cfg_function"](args)
            where args is a dict that typically includes:
              - "cond": conditioned noise pred
              - "uncond": unconditioned noise pred
              - "cond_scale": cfg scale (float)
            """

            if not isinstance(args, dict):
                raise TypeError(
                    f"RAAG sampler_cfg_function expected dict args, got {type(args)}"
                )

            cond = args.get("cond", None)
            uncond = args.get("uncond", None)
            cfg_scale = args.get("cond_scale", args.get("cfg", 1.0))

            # Defensive fallbacks
            if cond is None and uncond is None:
                raise RuntimeError("RAAG: args missing both 'cond' and 'uncond'.")
            if uncond is None:
                return cond
            if cond is None:
                return uncond

            cfg_scale_f = float(cfg_scale)

            # If you want linear CFG when cfg <= 1, keep this enabled
            if gate_cfg and cfg_scale_f <= 1.0:
                return uncond + cfg_scale_f * (cond - uncond)

            delta = cond - uncond

            # Per-sample ratio for stability in batches
            b = delta.shape[0]
            delta_n = torch.norm(delta.reshape(b, -1), dim=1)
            uncond_n = torch.norm(uncond.reshape(b, -1), dim=1).clamp_min(eps_f)

            ratio = (delta_n / uncond_n).to(delta.dtype)

            # RAAG weight: w = 1 + (w_max - 1) * exp(-alpha * ratio)
            w = 1.0 + (wmax_f - 1.0) * torch.exp(-alpha_f * ratio)

            # Broadcast w to match tensor dims
            while w.dim() < delta.dim():
                w = w.unsqueeze(-1)

            return uncond + w * delta

        # Patch model options
        opts = dict(getattr(m, "model_options", {}) or {})
        opts["sampler_cfg_function"] = raag_cfg_function
        m.model_options = opts

        return (m,)
