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
            "tile_size": 8,
            "colors_per_palette": 4,
            "num_palettes": 1,
            "description": "160 lines, 4 colors, blocky pixels, no dithering"
        },
        "NES - Famicom": {
            "palette_mode": "NES - Famicom (25 colors)",
            "resolution": 240,
            "dithering": "none",
            "scaling_method": "nearest",
            "tile_size": 16,  # Attribute table works on 16x16 blocks (2x2 of 8x8 tiles)
            "colors_per_palette": 4,  # 3 colors + shared background
            "num_palettes": 4,
            "description": "240p, 25 colors, 16x16 palette blocks"
        },
        "Sega Master System": {
            "palette_mode": "Sega Master System (32 colors)",
            "resolution": 192,
            "dithering": "none",
            "scaling_method": "nearest",
            "tile_size": 8,
            "colors_per_palette": 16,
            "num_palettes": 2,
            "description": "192p, 32 colors, 8x8 tiles"
        },
        "Game Boy": {
            "palette_mode": "Game Boy (4 shades)",
            "resolution": 144,
            "dithering": "ordered (bayer 4x4)",
            "scaling_method": "nearest",
            "tile_size": 8,
            "colors_per_palette": 4,
            "num_palettes": 1,  # Single palette for all tiles
            "description": "144p, 4 shades of green, 8x8 tiles"
        },
        "TurboGrafx-16": {
            "palette_mode": "TurboGrafx-16 (482 colors)",
            "resolution": 224,
            "dithering": "floyd-steinberg",
            "scaling_method": "area",
            "tile_size": 8,
            "colors_per_palette": 16,
            "num_palettes": 16,  # 256 colors total (16x16), limited from 482 for PIL compatibility
            "description": "224p, 256 colors (subset of 512), 8x8 tiles"
        },
        "Genesis - Mega Drive": {
            "palette_mode": "Genesis - Mega Drive (61 colors)",
            "resolution": 224,
            "dithering": "ordered (bayer 4x4)",
            "scaling_method": "nearest",
            "tile_size": 8,
            "colors_per_palette": 15,  # 15 + transparent
            "num_palettes": 4,
            "description": "224p, 61 colors, 8x8 tiles"
        },
        "SNES": {
            "palette_mode": "SNES (256 colors)",
            "resolution": 224,
            "dithering": "floyd-steinberg",
            "scaling_method": "lanczos",
            "tile_size": 8,
            "colors_per_palette": 16,  # Mode 1 4bpp: 15 + transparent
            "num_palettes": 8,
            "description": "224p, 256 colors, 8x8 tiles"
        },
        "Game Boy Color": {
            "palette_mode": "Game Boy Color (56 colors)",
            "resolution": 144,
            "dithering": "floyd-steinberg",
            "scaling_method": "nearest",
            "tile_size": 8,
            "colors_per_palette": 4,
            "num_palettes": 8,
            "description": "144p, 56 colors, 8x8 tiles"
        },
        "PlayStation": {
            "palette_mode": "PlayStation (256 colors)",
            "resolution": 240,
            "dithering": "floyd-steinberg",
            "scaling_method": "bilinear",
            "tile_size": 8,  # Not really tile-based, but for compatibility
            "colors_per_palette": 256,
            "num_palettes": 1,
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
                    # Tile palettes disabled - output quality needs improvement
                    # "enable_tile_palettes": ("BOOLEAN", {"default": False}),
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
                        "default": 0.15,
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
    
    def stylize(self, image, console, enable_scaling, enable_dithering,
                enable_pixelation, block_size, enable_scanlines, scanline_intensity):
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
        
        # Tile palettes disabled for now - output quality needs improvement
        # May revisit in future if we can improve color distribution and palette assignment
        enable_tile_palettes = False
        
        # Log what we're doing
        scaling_info = f"scaling to {specs['resolution']}p" if enable_scaling else "no scaling"
        dither_info = f"{dithering} dithering" if enable_dithering else "no dithering"
        pixelation_info = f"pixelation {block_size}x{block_size}" if enable_pixelation and block_size > 1 else "no pixelation"
        scanline_info = f"with scanlines (intensity {scanline_intensity})" if enable_scanlines else "no scanlines"
        
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
            scanlines=False,  # We'll apply scanlines ourselves with custom intensity
            scanline_intensity=scanline_intensity
        )
        
        # Tile palettes disabled - keeping code for potential future improvement
        # if enable_tile_palettes:
        #     result = (self.apply_tile_palettes(
        #         result[0], 
        #         specs["tile_size"], 
        #         specs["colors_per_palette"], 
        #         specs["num_palettes"]
        #     ),)
        
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
    
    def apply_tile_palettes(self, image, tile_size, colors_per_palette, num_palettes):
        """
        Apply tile-based palette restrictions like real hardware.
        Uses ACTUAL colors from the already-reduced image and groups them into palettes.
        
        NOTE: Currently disabled in the node UI due to poor output quality.
        The color clustering and palette assignment needs improvement to achieve
        authentic console-like results. Keeping code for potential future development.
        """
        batch, height, width, channels = image.shape
        
        # Make dimensions divisible by tile_size
        new_height = (height // tile_size) * tile_size
        new_width = (width // tile_size) * tile_size
        if new_height != height or new_width != width:
            image = image[:, :new_height, :new_width, :]
            height, width = new_height, new_width
        
        result_images = []
        for b in range(batch):
            img = image[b].clone()
            
            # Extract all unique colors from the already-reduced image
            img_flat = img.reshape(-1, channels)
            unique_colors = torch.unique(img_flat, dim=0)
            
            # If we have too many colors, sample them
            max_colors = num_palettes * colors_per_palette
            if len(unique_colors) > max_colors:
                # K-means-like sampling: pick evenly distributed colors
                indices = torch.linspace(0, len(unique_colors) - 1, max_colors).long()
                unique_colors = unique_colors[indices]
            
            # Group colors into palettes using k-means clustering
            if len(unique_colors) <= colors_per_palette:
                # Not enough colors for multiple palettes, just use one palette
                palettes = [unique_colors]
            else:
                # Use k-means to cluster colors into palettes
                palettes = self._cluster_colors_into_palettes(
                    unique_colors, num_palettes, colors_per_palette
                )
            
            # Calculate tile grid
            tiles_y = height // tile_size
            tiles_x = width // tile_size
            
            # Process each tile
            for ty in range(tiles_y):
                for tx in range(tiles_x):
                    # Extract tile
                    y_start = ty * tile_size
                    y_end = y_start + tile_size
                    x_start = tx * tile_size
                    x_end = x_start + tile_size
                    tile = img[y_start:y_end, x_start:x_end, :]
                    
                    # Find which palette best matches this tile's colors
                    tile_colors = torch.unique(tile.reshape(-1, channels), dim=0)
                    
                    best_palette_idx = 0
                    best_match_score = float('-inf')
                    
                    for p_idx, palette in enumerate(palettes):
                        # Score: how many tile colors are in this palette
                        match_score = 0
                        for tile_color in tile_colors:
                            # Check if this color is close to any color in palette
                            distances = torch.sum((palette - tile_color) ** 2, dim=1)
                            if torch.min(distances) < 0.01:  # Close enough threshold
                                match_score += 1
                        
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_palette_idx = p_idx
                    
                    # Get the selected palette
                    selected_palette = palettes[best_palette_idx]
                    
                    # Quantize each pixel in tile to closest color in selected palette
                    tile_flat = tile.reshape(-1, channels)
                    quantized_flat = torch.zeros_like(tile_flat)
                    
                    for i in range(tile_flat.shape[0]):
                        pixel = tile_flat[i]
                        distances = torch.sum((selected_palette - pixel) ** 2, dim=1)
                        closest_idx = torch.argmin(distances)
                        quantized_flat[i] = selected_palette[closest_idx]
                    
                    # Reshape and assign back
                    img[y_start:y_end, x_start:x_end, :] = quantized_flat.reshape(tile_size, tile_size, channels)
            
            result_images.append(img)
        
        print(f"[SCG Console Stylizer] Applied tile palettes ({tile_size}x{tile_size} tiles, {num_palettes} palettes, {colors_per_palette} colors each)")
        return torch.stack(result_images, dim=0)
    
    def _cluster_colors_into_palettes(self, colors, num_palettes, colors_per_palette):
        """Simple k-means clustering to group colors into palettes"""
        # Initialize palette centers evenly across color space
        palette_centers = []
        for i in range(num_palettes):
            idx = int(i * len(colors) / num_palettes)
            palette_centers.append(colors[idx])
        palette_centers = torch.stack(palette_centers)
        
        # Assign each color to nearest palette
        palettes = [[] for _ in range(num_palettes)]
        for color in colors:
            distances = torch.sum((palette_centers - color) ** 2, dim=1)
            nearest_palette = torch.argmin(distances).item()
            palettes[nearest_palette].append(color)
        
        # Limit each palette to colors_per_palette colors
        result_palettes = []
        for palette_colors in palettes:
            if len(palette_colors) == 0:
                # Empty palette, use first color
                palette_colors = [colors[0]]
            
            if len(palette_colors) > colors_per_palette:
                # Sample evenly
                indices = torch.linspace(0, len(palette_colors) - 1, colors_per_palette).long()
                palette_colors = [palette_colors[i] for i in indices]
            
            result_palettes.append(torch.stack(palette_colors))
        
        return result_palettes

