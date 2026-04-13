import torch
import numpy as np
from PIL import Image, ImageFilter


class SCGMaskImageDifference:
    """
    Compare source and target images to produce a mask of the regions that differ,
    plus both RGBA and RGB cutouts of the target with changed regions removed.

    Designed for edit-model workflows where color matching needs to ignore
    altered areas.

    Outputs:
        cutout_rgba    – 4-channel IMAGE: target with changed regions transparent
                         (use with Save Image for transparent PNGs)
        cutout_rgb     – 3-channel IMAGE: target with changed regions blacked out
                         (safe for any downstream node expecting standard RGB)
        mask           – binary/feathered MASK usable by any mask-consuming node
        difference_map – grayscale IMAGE visualising raw per-pixel difference
                         (handy for tuning the threshold)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.05,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": "Minimum per-pixel difference to count as changed. Lower = more sensitive.",
                    },
                ),
                "blur_radius": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.5,
                        "tooltip": "Feather radius for mask edges. 0 = hard binary mask.",
                    },
                ),
                "expand_pixels": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 200,
                        "step": 1,
                        "tooltip": "Dilate the mask by N pixels to catch edit boundaries.",
                    },
                ),
                "difference_mode": (
                    ["max_channel", "average", "euclidean"],
                    {"default": "average"},
                ),
                "rgb_fill": (
                    ["average", "black", "gray", "white"],
                    {
                        "default": "average",
                        "tooltip": "Colour used to replace masked regions in the RGB output.",
                    },
                ),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("cutout_rgba", "cutout_rgb", "mask", "difference_map")
    FUNCTION = "compute_difference_mask"
    CATEGORY = "scg-utils"

    _FILL_VALUES = {
        "black": 0.0,
        "gray": 0.5,
        "white": 1.0,
    }

    def compute_difference_mask(
        self,
        source_image,
        target_image,
        threshold,
        blur_radius,
        expand_pixels,
        difference_mode,
        rgb_fill,
        invert_mask,
    ):
        src = source_image
        tgt = target_image

        if src.shape[1:3] != tgt.shape[1:3]:
            src = self._resize_to_match(src, tgt.shape[1], tgt.shape[2])

        batch = max(src.shape[0], tgt.shape[0])
        masks = []
        diff_maps = []
        cutouts_rgba = []
        cutouts_rgb = []

        for i in range(batch):
            s = src[min(i, src.shape[0] - 1)]
            t = tgt[min(i, tgt.shape[0] - 1)]

            diff = torch.abs(s.float() - t.float())  # (H, W, C)

            if difference_mode == "average":
                diff_mono = diff.mean(dim=-1)
            elif difference_mode == "euclidean":
                diff_mono = torch.sqrt((diff ** 2).sum(dim=-1) / 3.0)
            else:  # max_channel
                diff_mono = diff.max(dim=-1)[0]

            mask = (diff_mono > threshold).float()

            if expand_pixels > 0:
                mask = self._dilate_mask(mask, expand_pixels)

            if blur_radius > 0:
                mask = self._blur_mask(mask, blur_radius)

            if invert_mask:
                mask = 1.0 - mask

            # mask=1 → changed, alpha = 1-mask → changed regions transparent
            alpha = (1.0 - mask).unsqueeze(-1)  # (H, W, 1)
            rgb = t[..., :3].float()
            mask_3ch = mask.unsqueeze(-1)  # (H, W, 1)

            rgba = torch.cat([rgb, alpha], dim=-1)  # (H, W, 4)

            if rgb_fill == "average":
                opaque = alpha.squeeze(-1) > 0.5
                if opaque.any():
                    fill_val = rgb[opaque].mean(dim=0)  # (3,)
                else:
                    fill_val = torch.tensor([0.5, 0.5, 0.5], device=rgb.device)
                bg = fill_val.view(1, 1, 3).expand_as(rgb)
            else:
                bg = torch.full_like(rgb, self._FILL_VALUES[rgb_fill])

            rgb_masked = rgb * (1.0 - mask_3ch) + bg * mask_3ch  # (H, W, 3)

            masks.append(mask)
            diff_maps.append(diff_mono)
            cutouts_rgba.append(rgba)
            cutouts_rgb.append(rgb_masked)

        mask_out = torch.stack(masks, dim=0)
        diff_rgb = torch.stack(diff_maps, dim=0).unsqueeze(-1).expand(-1, -1, -1, 3)
        rgba_out = torch.stack(cutouts_rgba, dim=0)
        rgb_out = torch.stack(cutouts_rgb, dim=0)

        return (rgba_out, rgb_out, mask_out, diff_rgb)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_to_match(img, target_h, target_w):
        bchw = img.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            bchw,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return resized.permute(0, 2, 3, 1)

    @staticmethod
    def _dilate_mask(mask, pixels):
        """Grow mask region using max-pool (no extra deps)."""
        kernel = pixels * 2 + 1
        m = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        dilated = torch.nn.functional.max_pool2d(
            m, kernel_size=kernel, stride=1, padding=pixels
        )
        return dilated[0, 0]

    @staticmethod
    def _blur_mask(mask, radius):
        """Feather via PIL GaussianBlur (always available in ComfyUI)."""
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(mask_np, mode="L")
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        return torch.from_numpy(
            np.array(blurred).astype(np.float32) / 255.0
        ).to(mask.device)
