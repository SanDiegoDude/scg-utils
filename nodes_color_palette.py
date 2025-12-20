import torch
import numpy as np
import math
from PIL import Image


class SCGColorPaletteTransformer:
    """
    Transform images to retro color palettes with optional megapixel scaling.
    Supports classic computer graphics modes, game console palettes, and bit depth reduction.
    """
    
    # Classic CGA palette (IBM PC)
    CGA_PALETTE = [
        (0, 0, 0),       # Black
        (0, 0, 170),     # Blue
        (0, 170, 0),     # Green
        (0, 170, 170),   # Cyan
        (170, 0, 0),     # Red
        (170, 0, 170),   # Magenta
        (170, 85, 0),    # Brown
        (170, 170, 170), # Light Gray
        (85, 85, 85),    # Dark Gray
        (85, 85, 255),   # Light Blue
        (85, 255, 85),   # Light Green
        (85, 255, 255),  # Light Cyan
        (255, 85, 85),   # Light Red
        (255, 85, 255),  # Light Magenta
        (255, 255, 85),  # Yellow
        (255, 255, 255), # White
    ]
    
    # Game Boy original (4 shades of green)
    GAMEBOY_PALETTE = [
        (15, 56, 15),     # Darkest green
        (48, 98, 48),     # Dark green
        (139, 172, 15),   # Light green
        (155, 188, 15),   # Lightest green
    ]
    
    # NES/Famicom palette (25 colors - typical on-screen maximum)
    NES_PALETTE = [
        (0, 0, 0),       # Black (shared background)
        (124, 124, 124), (0, 0, 252), (0, 0, 188), (68, 40, 188),
        (148, 0, 132), (168, 0, 32), (168, 16, 0), (136, 20, 0),
        (80, 48, 0), (0, 120, 0), (0, 104, 0), (0, 88, 0),
        (188, 188, 188), (0, 120, 248), (0, 88, 248), (104, 68, 252),
        (216, 0, 204), (228, 0, 88), (248, 56, 0), (228, 92, 16),
        (172, 124, 0), (0, 184, 0), (248, 248, 248), (60, 188, 252),
        (104, 136, 252),
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "color_mode": ([
                    # Bit Depth Modes
                    "Monochrome (2 colors)",
                    "4 Colors (2-bit)",
                    "8 Colors (3-bit)",
                    "4096 Colors (12-bit)",
                    "32768 Colors (15-bit)",
                    "65536 Colors (16-bit)",
                    # Computer Graphics
                    "CGA (16 colors)",
                    "EGA (64 colors)",
                    "VGA (256 colors)",
                    # Game Consoles
                    "Atari 2600 (4 colors)",
                    "NES - Famicom (25 colors)",
                    "Sega Master System (32 colors)",
                    "Game Boy (4 shades)",
                    "TurboGrafx-16 (482 colors)",
                    "Genesis - Mega Drive (61 colors)",
                    "SNES (256 colors)",
                    "Game Boy Color (56 colors)",
                    "PlayStation (256 colors)",
                ], {
                    "default": "VGA (256 colors)"
                }),
                "dithering": ([
                    "none",
                    "floyd-steinberg",
                    "atkinson",
                    "jarvis-judice-ninke",
                    "stucki",
                    "burkes",
                    "sierra",
                    "ordered (bayer 4x4)",
                    "ordered (bayer 8x8)",
                ], {
                    "default": "floyd-steinberg"
                }),
                "enable_scaling": ("BOOLEAN", {"default": False}),
                "scaling_mode": ([
                    "rescale (megapixels)",
                    "resize (short side)",
                ], {
                    "default": "rescale (megapixels)"
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 16.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "resize": ("INT", {
                    "default": 512,
                    "min": 16,
                    "max": 8192,
                    "step": 16,
                    "display": "number"
                }),
                "scaling_method": ([
                    "lanczos",
                    "bicubic", 
                    "bilinear",
                    "nearest",
                    "area"
                ], {
                    "default": "lanczos"
                }),
                "enable_pixelation": ("BOOLEAN", {"default": False}),
                "block_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 32,
                    "step": 1,
                    "display": "number"
                }),
                "scanlines": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "transform_palette"
    CATEGORY = "scg-utils"
    
    def generate_monochrome_palette(self):
        """Generate 2-color monochrome palette"""
        return [(0, 0, 0), (255, 255, 255)]
    
    def generate_4color_palette(self):
        """Generate 4-color palette (2 bits per pixel)"""
        return [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)]
    
    def generate_8color_palette(self):
        """Generate 8-color palette (3-bit RGB, 1 bit per channel)"""
        palette = []
        for r in range(2):
            for g in range(2):
                for b in range(2):
                    palette.append((r * 255, g * 255, b * 255))
        return palette
    
    def generate_ega_palette(self):
        """Generate 64-color EGA palette (2 bits per RGB channel)"""
        palette = []
        for r in range(4):
            for g in range(4):
                for b in range(4):
                    palette.append((
                        int(r * 255 / 3),
                        int(g * 255 / 3),
                        int(b * 255 / 3)
                    ))
        return palette
    
    def generate_atari2600_palette(self):
        """Generate Atari 2600 palette (4 colors on-screen per scanline)"""
        # Atari 2600 could only display 4 colors per scanline:
        # background, playfield, and 2 player sprites
        # Using a typical 4-color subset from the 128-color NTSC palette
        palette = [
            (0, 0, 0),       # Black (background)
            (164, 164, 28),  # Yellow-tan (playfield)
            (200, 72, 72),   # Red (player 1)
            (92, 148, 252),  # Blue (player 2)
        ]
        return palette
    
    def generate_sms_palette(self):
        """Generate Sega Master System palette (32 colors on-screen from 64 total)"""
        # SMS uses 2 palettes of 16 colors each (32 colors on-screen)
        # 6-bit RGB: 2 bits per channel
        palette = []
        # Generate a well-distributed 32-color subset
        for r in range(4):
            for g in range(4):
                for b in range(2):
                    palette.append((
                        int(r * 255 / 3),
                        int(g * 255 / 3),
                        int(b * 255)
                    ))
        return palette
    
    def generate_turbografx_palette(self):
        """Generate TurboGrafx-16/PC Engine palette (482 colors on-screen, using 512-color palette)"""
        # TurboGrafx could display 482 colors (close to full 512-color palette)
        # 9-bit color: 3 bits per channel
        palette = []
        for r in range(8):
            for g in range(8):
                for b in range(8):
                    palette.append((
                        int(r * 255 / 7),
                        int(g * 255 / 7),
                        int(b * 255 / 7)
                    ))
        return palette[:482]  # Limit to accurate on-screen count
    
    def generate_genesis_palette(self):
        """Generate Sega Genesis/Mega Drive palette (61 colors on-screen from 512 total)"""
        # Genesis displays 61 colors: 4 palettes of 15 colors each + 1 shared transparent
        # 9-bit color (3 bits per channel) with characteristic darker output
        palette = []
        # Generate 61 well-distributed colors from the 512-color space
        step = 512 / 61
        for i in range(61):
            idx = int(i * step)
            r = (idx >> 6) & 0x7
            g = (idx >> 3) & 0x7
            b = idx & 0x7
            # Genesis had slightly darker output
            palette.append((
                int(r * 255 / 7 * 0.9),
                int(g * 255 / 7 * 0.9),
                int(b * 255 / 7 * 0.9)
            ))
        return palette
    
    def generate_snes_palette(self):
        """Generate SNES palette subset (256 most common colors from 32768 possible)"""
        # SNES could display 256 colors on screen from 32768 (15-bit)
        # We'll use a well-distributed subset
        palette = []
        # 216-color cube (6x6x6)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    palette.append((
                        int(r * 255 / 5),
                        int(g * 255 / 5),
                        int(b * 255 / 5)
                    ))
        # Add 40 grayscale values
        for i in range(40):
            gray = int((i / 39) * 255)
            palette.append((gray, gray, gray))
        return palette
    
    def generate_gbc_palette(self):
        """Generate Game Boy Color palette (56 colors on-screen from 32768 total)"""
        # GBC: 8 background palettes × 4 colors + 8 sprite palettes × 3 colors = 56
        # 15-bit RGB (5 bits per channel)
        palette = []
        # Generate 56 well-distributed colors from 15-bit color space
        for i in range(56):
            # Distribute across 15-bit color space
            r = (i * 7) % 32
            g = ((i * 11) % 32)
            b = ((i * 13) % 32)
            palette.append((
                int(r * 255 / 31),
                int(g * 255 / 31),
                int(b * 255 / 31)
            ))
        return palette
    
    def generate_vga_palette(self):
        """Generate 256-color VGA palette (Mode 13h style)"""
        palette = []
        # Standard 216-color cube (6x6x6)
        for r in range(6):
            for g in range(6):
                for b in range(6):
                    palette.append((
                        int(r * 255 / 5),
                        int(g * 255 / 5),
                        int(b * 255 / 5)
                    ))
        # Add grayscale ramp
        for i in range(40):
            gray = int((i / 39) * 255)
            palette.append((gray, gray, gray))
        return palette
    
    def generate_psx_palette(self):
        """Generate PlayStation 256-color mode palette"""
        # PSX commonly used 256-color mode for backgrounds
        # Use same as VGA for practical purposes
        return self.generate_vga_palette()
    
    def quantize_to_palette(self, pil_image, palette, dithering="floyd-steinberg"):
        """Quantize image to a specific palette"""
        # Create a palette image
        palette_img = Image.new('P', (1, 1))
        # Flatten palette list for PIL
        flat_palette = [c for rgb in palette for c in rgb]
        # Pad palette to 256 colors if needed
        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])
        palette_img.putpalette(flat_palette)
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Apply dithering
        if dithering == "none":
            quantized = pil_image.quantize(palette=palette_img, dither=Image.Dither.NONE)
        elif dithering == "floyd-steinberg":
            quantized = pil_image.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
        elif dithering.startswith("ordered"):
            # Use custom ordered dithering
            if "8x8" in dithering:
                quantized = self._apply_ordered_dither(pil_image, palette, matrix_size=8)
            else:
                quantized = self._apply_ordered_dither(pil_image, palette, matrix_size=4)
        else:
            # Custom dithering algorithms
            quantized = self._apply_custom_dither(pil_image, palette, dithering)
        
        # Convert back to RGB
        return quantized.convert('RGB')
    
    def _apply_ordered_dither(self, pil_image, palette, matrix_size=4):
        """Apply ordered (Bayer) dithering"""
        if matrix_size == 4:
            bayer_matrix = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ], dtype=np.float32) / 16.0 - 0.5
        else:  # 8x8
            bayer_matrix = np.array([
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], dtype=np.float32) / 64.0 - 0.5
        
        img_array = np.array(pil_image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Create threshold map
        threshold_map = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                threshold_map[y, x] = bayer_matrix[y % matrix_size, x % matrix_size]
        
        # Apply threshold to each channel
        threshold_map_3d = np.stack([threshold_map] * 3, axis=-1)
        dithered = img_array + (threshold_map_3d * 32)
        dithered = np.clip(dithered, 0, 255)
        
        # Convert to PIL and quantize
        dithered_pil = Image.fromarray(dithered.astype(np.uint8))
        
        # Create palette image
        palette_img = Image.new('P', (1, 1))
        flat_palette = [c for rgb in palette for c in rgb]
        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])
        palette_img.putpalette(flat_palette)
        
        return dithered_pil.quantize(palette=palette_img, dither=Image.Dither.NONE)
    
    def _apply_custom_dither(self, pil_image, palette, algorithm):
        """Apply custom error diffusion dithering algorithms"""
        img_array = np.array(pil_image, dtype=np.float32)
        height, width, channels = img_array.shape
        
        # Define error diffusion matrices
        # Format: (dx, dy, weight)
        if algorithm == "atkinson":
            # Atkinson dithering (MacPaint)
            diffusion = [
                (1, 0, 1/8), (2, 0, 1/8),
                (-1, 1, 1/8), (0, 1, 1/8), (1, 1, 1/8),
                (0, 2, 1/8)
            ]
        elif algorithm == "jarvis-judice-ninke":
            diffusion = [
                (1, 0, 7/48), (2, 0, 5/48),
                (-2, 1, 3/48), (-1, 1, 5/48), (0, 1, 7/48), (1, 1, 5/48), (2, 1, 3/48),
                (-2, 2, 1/48), (-1, 2, 3/48), (0, 2, 5/48), (1, 2, 3/48), (2, 2, 1/48)
            ]
        elif algorithm == "stucki":
            diffusion = [
                (1, 0, 8/42), (2, 0, 4/42),
                (-2, 1, 2/42), (-1, 1, 4/42), (0, 1, 8/42), (1, 1, 4/42), (2, 1, 2/42),
                (-2, 2, 1/42), (-1, 2, 2/42), (0, 2, 4/42), (1, 2, 2/42), (2, 2, 1/42)
            ]
        elif algorithm == "burkes":
            diffusion = [
                (1, 0, 8/32), (2, 0, 4/32),
                (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 8/32), (1, 1, 4/32), (2, 1, 2/32)
            ]
        elif algorithm == "sierra":
            diffusion = [
                (1, 0, 5/32), (2, 0, 3/32),
                (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 5/32), (1, 1, 4/32), (2, 1, 2/32),
                (-1, 2, 2/32), (0, 2, 3/32), (1, 2, 2/32)
            ]
        else:  # Default to Floyd-Steinberg
            diffusion = [
                (1, 0, 7/16),
                (-1, 1, 3/16), (0, 1, 5/16), (1, 1, 1/16)
            ]
        
        # Apply error diffusion
        img_dithered = img_array.copy()
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    old_pixel = img_dithered[y, x, c]
                    
                    # Find nearest color in palette
                    new_pixel = self._find_nearest_palette_color(
                        img_dithered[y, x, :], palette
                    )[c]
                    
                    img_dithered[y, x, c] = new_pixel
                    error = old_pixel - new_pixel
                    
                    # Distribute error
                    for dx, dy, weight in diffusion:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            img_dithered[ny, nx, c] += error * weight
        
        img_dithered = np.clip(img_dithered, 0, 255).astype(np.uint8)
        dithered_pil = Image.fromarray(img_dithered)
        
        # Create palette image and quantize
        palette_img = Image.new('P', (1, 1))
        flat_palette = [c for rgb in palette for c in rgb]
        while len(flat_palette) < 768:
            flat_palette.extend([0, 0, 0])
        palette_img.putpalette(flat_palette)
        
        return dithered_pil.quantize(palette=palette_img, dither=Image.Dither.NONE)
    
    def _find_nearest_palette_color(self, pixel, palette):
        """Find nearest color in palette using Euclidean distance"""
        min_dist = float('inf')
        nearest = palette[0]
        
        for color in palette:
            dist = sum((pixel[i] - color[i]) ** 2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                nearest = color
        
        return np.array(nearest, dtype=np.float32)
    
    def quantize_to_bits(self, pil_image, bits_per_channel, dithering="floyd-steinberg"):
        """Quantize image to N bits per channel"""
        img_array = np.array(pil_image, dtype=np.float32)
        
        if dithering == "floyd-steinberg":
            img_array = self._floyd_steinberg_dither_bits(img_array, bits_per_channel)
        elif dithering.startswith("ordered"):
            img_array = self._ordered_dither_bits(img_array, bits_per_channel, 
                                                 matrix_size=8 if "8x8" in dithering else 4)
        elif dithering != "none":
            img_array = self._custom_dither_bits(img_array, bits_per_channel, dithering)
        else:
            # Simple quantization
            levels = 2 ** bits_per_channel
            img_array = np.round(img_array / 255.0 * (levels - 1)) * (255.0 / (levels - 1))
        
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _floyd_steinberg_dither_bits(self, img_array, bits_per_channel):
        """Floyd-Steinberg dithering for bit reduction"""
        height, width, channels = img_array.shape
        levels = 2 ** bits_per_channel
        img_float = img_array.copy()
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    old_pixel = img_float[y, x, c]
                    new_pixel = np.round(old_pixel / 255.0 * (levels - 1)) * (255.0 / (levels - 1))
                    img_float[y, x, c] = new_pixel
                    error = old_pixel - new_pixel
                    
                    if x + 1 < width:
                        img_float[y, x + 1, c] += error * 7/16
                    if y + 1 < height:
                        if x > 0:
                            img_float[y + 1, x - 1, c] += error * 3/16
                        img_float[y + 1, x, c] += error * 5/16
                        if x + 1 < width:
                            img_float[y + 1, x + 1, c] += error * 1/16
        
        return img_float
    
    def _ordered_dither_bits(self, img_array, bits_per_channel, matrix_size=4):
        """Ordered dithering for bit reduction"""
        if matrix_size == 4:
            bayer_matrix = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ], dtype=np.float32) / 16.0 - 0.5
        else:
            bayer_matrix = np.array([
                [0, 32, 8, 40, 2, 34, 10, 42],
                [48, 16, 56, 24, 50, 18, 58, 26],
                [12, 44, 4, 36, 14, 46, 6, 38],
                [60, 28, 52, 20, 62, 30, 54, 22],
                [3, 35, 11, 43, 1, 33, 9, 41],
                [51, 19, 59, 27, 49, 17, 57, 25],
                [15, 47, 7, 39, 13, 45, 5, 37],
                [63, 31, 55, 23, 61, 29, 53, 21]
            ], dtype=np.float32) / 64.0 - 0.5
        
        height, width = img_array.shape[:2]
        levels = 2 ** bits_per_channel
        
        threshold_map = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                threshold_map[y, x] = bayer_matrix[y % matrix_size, x % matrix_size]
        
        threshold_map_3d = np.stack([threshold_map] * 3, axis=-1)
        step = 255.0 / (levels - 1)
        dithered = img_array + (threshold_map_3d * step)
        dithered = np.round(dithered / 255.0 * (levels - 1)) * (255.0 / (levels - 1))
        
        return dithered
    
    def _custom_dither_bits(self, img_array, bits_per_channel, algorithm):
        """Apply custom error diffusion for bit reduction"""
        height, width, channels = img_array.shape
        levels = 2 ** bits_per_channel
        img_float = img_array.copy()
        
        # Get diffusion matrix
        if algorithm == "atkinson":
            diffusion = [(1, 0, 1/8), (2, 0, 1/8), (-1, 1, 1/8), (0, 1, 1/8), (1, 1, 1/8), (0, 2, 1/8)]
        elif algorithm == "jarvis-judice-ninke":
            diffusion = [(1, 0, 7/48), (2, 0, 5/48), (-2, 1, 3/48), (-1, 1, 5/48), (0, 1, 7/48), 
                        (1, 1, 5/48), (2, 1, 3/48), (-2, 2, 1/48), (-1, 2, 3/48), (0, 2, 5/48), 
                        (1, 2, 3/48), (2, 2, 1/48)]
        elif algorithm == "stucki":
            diffusion = [(1, 0, 8/42), (2, 0, 4/42), (-2, 1, 2/42), (-1, 1, 4/42), (0, 1, 8/42), 
                        (1, 1, 4/42), (2, 1, 2/42), (-2, 2, 1/42), (-1, 2, 2/42), (0, 2, 4/42), 
                        (1, 2, 2/42), (2, 2, 1/42)]
        elif algorithm == "burkes":
            diffusion = [(1, 0, 8/32), (2, 0, 4/32), (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 8/32), 
                        (1, 1, 4/32), (2, 1, 2/32)]
        elif algorithm == "sierra":
            diffusion = [(1, 0, 5/32), (2, 0, 3/32), (-2, 1, 2/32), (-1, 1, 4/32), (0, 1, 5/32), 
                        (1, 1, 4/32), (2, 1, 2/32), (-1, 2, 2/32), (0, 2, 3/32), (1, 2, 2/32)]
        else:
            diffusion = [(1, 0, 7/16), (-1, 1, 3/16), (0, 1, 5/16), (1, 1, 1/16)]
        
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    old_pixel = img_float[y, x, c]
                    new_pixel = np.round(old_pixel / 255.0 * (levels - 1)) * (255.0 / (levels - 1))
                    img_float[y, x, c] = new_pixel
                    error = old_pixel - new_pixel
                    
                    for dx, dy, weight in diffusion:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            img_float[ny, nx, c] += error * weight
        
        return img_float
    
    def rescale_image(self, image, megapixels, scaling_method):
        """Rescale image to target megapixel size"""
        batch, height, width, channels = image.shape
        current_pixels = height * width
        target_pixels = int(megapixels * 1_000_000)
        
        scale_factor = (target_pixels / current_pixels) ** 0.5
        new_height = max(1, int(height * scale_factor))
        new_width = max(1, int(width * scale_factor))
        
        batch_info = f" (batch of {batch} images)" if batch > 1 else ""
        print(f"[SCG Color Palette] Rescaling{batch_info} from {width}x{height} ({current_pixels/1_000_000:.2f}MP) to {new_width}x{new_height} ({(new_width*new_height)/1_000_000:.2f}MP) using {scaling_method}")
        
        if new_height == height and new_width == width:
            return image
        
        if scaling_method == "lanczos":
            return self._scale_with_pil(image, new_height, new_width, Image.LANCZOS)
        elif scaling_method == "area":
            return self._scale_with_pil(image, new_height, new_width, Image.BOX)
        else:
            return self._scale_with_torch(image, new_height, new_width, scaling_method)
    
    def resize_image(self, image, short_side, scaling_method):
        """Resize image by short side length, maintaining aspect ratio"""
        batch, height, width, channels = image.shape
        
        # Determine which is the short side
        if height < width:
            new_height = short_side
            new_width = int(width * (short_side / height))
        else:
            new_width = short_side
            new_height = int(height * (short_side / width))
        
        new_height = max(1, new_height)
        new_width = max(1, new_width)
        
        batch_info = f" (batch of {batch} images)" if batch > 1 else ""
        print(f"[SCG Color Palette] Resizing{batch_info} from {width}x{height} to {new_width}x{new_height} (short side: {short_side}) using {scaling_method}")
        
        if new_height == height and new_width == width:
            return image
        
        if scaling_method == "lanczos":
            return self._scale_with_pil(image, new_height, new_width, Image.LANCZOS)
        elif scaling_method == "area":
            return self._scale_with_pil(image, new_height, new_width, Image.BOX)
        else:
            return self._scale_with_torch(image, new_height, new_width, scaling_method)
    
    def _scale_with_torch(self, image, new_height, new_width, method):
        """Scale image using PyTorch"""
        image_bchw = image.permute(0, 3, 1, 2)
        resized_bchw = torch.nn.functional.interpolate(
            image_bchw, size=(new_height, new_width), mode=method,
            align_corners=False if method != 'nearest' else None,
            antialias=True if method in ['bicubic', 'bilinear'] else False
        )
        return resized_bchw.permute(0, 2, 3, 1)
    
    def _scale_with_pil(self, image, new_height, new_width, resample_method):
        """Scale image using PIL"""
        batch = image.shape[0]
        scaled_images = []
        
        for i in range(batch):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            pil_img_resized = pil_img.resize((new_width, new_height), resample=resample_method)
            img_np_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
            scaled_images.append(torch.from_numpy(img_np_resized))
        
        return torch.stack(scaled_images, dim=0)
    
    def apply_scanlines(self, image, intensity=0.5):
        """Apply CRT-style scanlines effect"""
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
        
        print(f"[SCG Color Palette] Applied scanlines effect (intensity: {intensity})")
        return result
    
    def apply_pixelation(self, image, block_size):
        """Apply pixelation effect by averaging blocks of pixels (pixel confinement)"""
        if block_size <= 1:
            return image
        
        batch, height, width, channels = image.shape
        
        # Calculate new dimensions (must be divisible by block_size)
        new_height = (height // block_size) * block_size
        new_width = (width // block_size) * block_size
        
        # Crop if necessary
        if new_height != height or new_width != width:
            image = image[:, :new_height, :new_width, :]
            height, width = new_height, new_width
        
        # Reshape to group pixels into blocks
        # From: (batch, height, width, channels)
        # To: (batch, height/block, block, width/block, block, channels)
        image_blocked = image.reshape(
            batch,
            height // block_size,
            block_size,
            width // block_size,
            block_size,
            channels
        )
        
        # Average within each block
        # Average over dimensions 2 and 4 (the block dimensions)
        image_averaged = image_blocked.mean(dim=(2, 4))
        
        # Expand back to full size by repeating the averaged values
        image_pixelated = image_averaged.repeat_interleave(block_size, dim=1).repeat_interleave(block_size, dim=2)
        
        print(f"[SCG Color Palette] Applied pixelation (block size: {block_size}x{block_size})")
        return image_pixelated
    
    def transform_palette(self, image, color_mode, dithering, enable_scaling, scaling_mode, 
                         megapixels, resize, scaling_method, enable_pixelation, block_size, scanlines):
        """Transform image to retro color palette with optional scaling and effects"""
        batch, height, width, channels = image.shape
        
        # Scale first if enabled (using selected method)
        if enable_scaling:
            if "rescale" in scaling_mode:
                image = self.rescale_image(image, megapixels, scaling_method)
            else:  # resize mode
                image = self.resize_image(image, resize, scaling_method)
            batch, height, width, channels = image.shape
        
        # Apply pixelation after scaling but before color reduction
        if enable_pixelation and block_size > 1:
            image = self.apply_pixelation(image, block_size)
            batch, height, width, channels = image.shape
        
        # Determine palette and processing method
        palette = None
        bits_per_channel = None
        palette_name = color_mode
        
        if "Monochrome" in color_mode:
            palette = self.generate_monochrome_palette()
        elif "4 Colors" in color_mode:
            palette = self.generate_4color_palette()
        elif "8 Colors" in color_mode:
            palette = self.generate_8color_palette()
        elif "CGA" in color_mode:
            palette = self.CGA_PALETTE
        elif "EGA" in color_mode:
            palette = self.generate_ega_palette()
        elif "VGA" in color_mode:
            palette = self.generate_vga_palette()
        elif "Atari 2600" in color_mode:
            palette = self.generate_atari2600_palette()
        elif "NES" in color_mode or "Famicom" in color_mode:
            palette = self.NES_PALETTE
        elif "Sega Master System" in color_mode:
            palette = self.generate_sms_palette()
        elif "Game Boy" in color_mode and "Color" not in color_mode:
            palette = self.GAMEBOY_PALETTE
        elif "TurboGrafx" in color_mode:
            palette = self.generate_turbografx_palette()
        elif "Genesis" in color_mode or "Mega Drive" in color_mode:
            palette = self.generate_genesis_palette()
        elif "SNES" in color_mode:
            palette = self.generate_snes_palette()
        elif "Game Boy Color" in color_mode:
            palette = self.generate_gbc_palette()
        elif "PlayStation" in color_mode:
            palette = self.generate_psx_palette()
        elif "4096" in color_mode:
            bits_per_channel = 4
        elif "32768" in color_mode:
            bits_per_channel = 5
        elif "65536" in color_mode:
            bits_per_channel = 5  # 5-6-5 RGB, we'll use 5 for simplicity
        else:
            palette = self.generate_vga_palette()
        
        batch_info = f" (batch of {batch} images)" if batch > 1 else ""
        dither_info = f" with {dithering} dithering" if dithering != "none" else " (no dithering)"
        print(f"[SCG Color Palette] Transforming{batch_info} to {palette_name}{dither_info}")
        
        # Process each image
        result_images = []
        for i in range(batch):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            if palette is not None:
                quantized = self.quantize_to_palette(pil_img, palette, dithering)
            else:
                quantized = self.quantize_to_bits(pil_img, bits_per_channel, dithering)
            
            img_np_result = np.array(quantized).astype(np.float32) / 255.0
            result_images.append(torch.from_numpy(img_np_result))
        
        # Stack batch
        result_batch = torch.stack(result_images, dim=0)
        
        # Apply scanlines effect if enabled
        if scanlines:
            result_batch = self.apply_scanlines(result_batch)
        
        return (result_batch,)

