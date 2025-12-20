import torch
import numpy as np
from PIL import Image
from .nodes_color_palette import SCGColorPaletteTransformer


class SCGConsoleStylizer:
    """
    Simplified node for accurate game console aesthetic.
    Automatically applies the correct palette, resolution, and dithering for each console.
    """
    
    # Console specifications with authentic settings
    CONSOLE_SPECS = {
        "Atari 2600": {
            "palette_mode": "Atari 2600 (4 colors)",
            "resolution": 160,
            "dithering": "none",
            "scaling_method": "nearest",
            "description": "160 lines, 4 colors, blocky pixels, no dithering"
        },
        "NES - Famicom": {
            "palette_mode": "NES - Famicom (25 colors)",
            "resolution": 240,
            "dithering": "none",
            "scaling_method": "nearest",
            "description": "240p, 25 colors, clean pixels (no hardware dithering)"
        },
        "Sega Master System": {
            "palette_mode": "Sega Master System (32 colors)",
            "resolution": 192,
            "dithering": "none",
            "scaling_method": "nearest",
            "description": "192p, 32 colors, clean pixels"
        },
        "Game Boy": {
            "palette_mode": "Game Boy (4 shades)",
            "resolution": 144,
            "dithering": "ordered (bayer 4x4)",
            "scaling_method": "nearest",
            "description": "144p, 4 shades of green, ordered dither"
        },
        "TurboGrafx-16": {
            "palette_mode": "TurboGrafx-16 (482 colors)",
            "resolution": 224,
            "dithering": "floyd-steinberg",
            "scaling_method": "area",
            "description": "224p, 482 colors, vibrant"
        },
        "Genesis - Mega Drive": {
            "palette_mode": "Genesis - Mega Drive (61 colors)",
            "resolution": 224,
            "dithering": "ordered (bayer 4x4)",
            "scaling_method": "nearest",
            "description": "224p, 61 colors, characteristic darker tones"
        },
        "SNES": {
            "palette_mode": "SNES (256 colors)",
            "resolution": 224,
            "dithering": "floyd-steinberg",
            "scaling_method": "lanczos",
            "description": "224p, 256 colors, rich palette"
        },
        "Game Boy Color": {
            "palette_mode": "Game Boy Color (56 colors)",
            "resolution": 144,
            "dithering": "floyd-steinberg",
            "scaling_method": "nearest",
            "description": "144p, 56 colors"
        },
        "PlayStation": {
            "palette_mode": "PlayStation (256 colors)",
            "resolution": 240,
            "dithering": "floyd-steinberg",
            "scaling_method": "bilinear",
            "description": "240p, 256-color CLUT mode"
        },
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "console": (list(cls.CONSOLE_SPECS.keys()), {
                    "default": "SNES"
                }),
                "enable_scaling": ("BOOLEAN", {"default": True}),
                "enable_dithering": ("BOOLEAN", {"default": True}),
                "enable_pixelation": ("BOOLEAN", {"default": False}),
                "block_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "display": "number"
                }),
                "enable_scanlines": ("BOOLEAN", {"default": False}),
                "scanline_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stylize"
    CATEGORY = "scg-utils"
    
    def __init__(self):
        # Initialize the color palette transformer
        self.transformer = SCGColorPaletteTransformer()
    
    def stylize(self, image, console, enable_scaling, enable_dithering, enable_pixelation, 
                block_size, enable_scanlines, scanline_intensity):
        """
        Apply authentic console styling to image.
        
        Args:
            image: Input image tensor
            console: Target console platform
            enable_scaling: Whether to scale to console resolution
            enable_dithering: Whether to apply dithering
            enable_pixelation: Whether to apply pixel confinement
            block_size: Size of pixel blocks (1 = no pixelation)
            enable_scanlines: Whether to apply CRT scanlines
            scanline_intensity: Strength of scanline effect (0.0-1.0)
        
        Returns:
            Tuple containing the styled image
        """
        # Get console specifications
        specs = self.CONSOLE_SPECS[console]
        
        # Override dithering if disabled
        dithering = specs["dithering"] if enable_dithering else "none"
        
        # Determine scaling parameters
        if enable_scaling:
            scaling_mode = "resize (short side)"
            resize = specs["resolution"]
            scaling_method = specs["scaling_method"]
        else:
            scaling_mode = "rescale (megapixels)"
            resize = 512  # Dummy value, won't be used
            scaling_method = "lanczos"
        
        # Log what we're doing
        scaling_info = f"scaling to {specs['resolution']}p" if enable_scaling else "no scaling"
        dither_info = f"{dithering} dithering" if enable_dithering else "no dithering"
        pixelation_info = f"pixelation {block_size}x{block_size}" if enable_pixelation and block_size > 1 else "no pixelation"
        scanline_info = "with scanlines" if enable_scanlines else "no scanlines"
        
        print(f"[SCG Console Stylizer] {console} style: {scaling_info}, {dither_info}, {pixelation_info}, {scanline_info}")
        print(f"[SCG Console Stylizer] Console info: {specs['description']}")
        
        # Apply transformation using the color palette transformer
        result = self.transformer.transform_palette(
            image=image,
            color_mode=specs["palette_mode"],
            dithering=dithering,
            enable_scaling=enable_scaling,
            scaling_mode=scaling_mode,
            megapixels=1.0,  # Not used when resize mode
            resize=resize,
            scaling_method=scaling_method,
            enable_pixelation=enable_pixelation,
            block_size=block_size,
            scanlines=False  # We'll apply scanlines ourselves with custom intensity
        )
        
        # Apply custom scanlines if enabled
        if enable_scanlines:
            result = (self.apply_scanlines(result[0], scanline_intensity),)
        
        return result
    
    def apply_scanlines(self, image, intensity):
        """Apply CRT-style scanlines effect with custom intensity"""
        batch, height, width, channels = image.shape
        
        # Create scanline mask
        scanline_mask = torch.ones((height, width), dtype=image.dtype, device=image.device)
        
        # Apply darkening to every other line
        for y in range(1, height, 2):
            scanline_mask[y, :] = 1.0 - intensity
        
        # Apply mask to all channels
        scanline_mask_3d = scanline_mask.unsqueeze(-1).expand(-1, -1, channels)
        
        # Apply to all images in batch
        result = image * scanline_mask_3d.unsqueeze(0)
        
        print(f"[SCG Console Stylizer] Applied scanlines (intensity: {intensity})")
        return result

