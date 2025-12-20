# SCG Color Palette Transformer

Transform images to retro color palettes with optional megapixel scaling. Experience the nostalgia of classic computer graphics and game console aesthetics!

## Overview

The **SCG Color Palette Transformer** node brings vintage computer graphics and game console aesthetics to your images by reducing them to classic limited color palettes. Perfect for achieving retro gaming looks, pixel art styles, or creative color grading.

## Features

### 🎨 Bit Depth Modes
- **Monochrome (2 colors)**: Pure black and white
- **4 Colors (2-bit)**: Simple grayscale palette
- **8 Colors (3-bit)**: Basic RGB color space (1 bit per channel)
- **4096 Colors (12-bit)**: 4 bits per RGB channel
- **32768 Colors (15-bit)**: 5 bits per RGB channel (common in many 90s systems)
- **65536 Colors (16-bit)**: 5-6-5 RGB color depth

### 💻 Computer Graphics Modes
- **CGA (16 colors)**: IBM Color Graphics Adapter - the iconic 16-color palette from early PC gaming
- **EGA (64 colors)**: Enhanced Graphics Adapter - 64 colors with 2 bits per RGB channel
- **VGA (256 colors)**: Mode 13h style palette - 216-color cube plus grayscale ramp

### 🎮 Game Console Palettes (Accurate On-Screen Limits!)

#### Classic Era
- **Atari 2600 (4 colors)**: The console that started it all - only 4 colors per scanline! (background, playfield, 2 sprites from 128-color palette)

#### 8-Bit Era
- **NES - Famicom (25 colors)**: Iconic Nintendo palette - 25 colors on-screen from 52-54 unique colors
- **Sega Master System (32 colors)**: Sega's 8-bit competitor - 32 colors on-screen from 64 total
- **Game Boy (4 shades)**: Classic green-tinted monochrome palette

#### 16-Bit Era
- **TurboGrafx-16 (482 colors)**: PC Engine's amazing color capabilities - 482 colors on-screen from 512 total!
- **Genesis - Mega Drive (61 colors)**: Sega's 16-bit console - 61 colors on-screen (4 palettes × 15 + 1 shared) from 512 total, with characteristic darker tones
- **SNES (256 colors)**: Super Nintendo's rich palette - 256 colors on-screen from 32,768 possible colors
- **Game Boy Color (56 colors)**: GBC's colorful upgrade - 56 colors on-screen (8 BG palettes × 4 + 8 sprite palettes × 3) from 32,768 total

#### 32-Bit Era
- **PlayStation (256 colors)**: PSX's 8-bit CLUT mode - 256 colors on-screen from 16.7 million possible

### 🎨 Dithering Algorithms

Classic error diffusion methods:
- **None**: Clean posterized look with hard color transitions
- **Floyd-Steinberg**: The classic - smooth gradients, fast processing
- **Atkinson**: Used by MacPaint - subtle, preserves brightness well
- **Jarvis-Judice-Ninke**: High-quality with wider error distribution
- **Stucki**: Similar to JJN but with different weights
- **Burkes**: Good balance between speed and quality
- **Sierra**: Three-row error distribution for smooth results

Pattern-based dithering:
- **Ordered (Bayer 4x4)**: Classic crosshatch pattern, fast
- **Ordered (Bayer 8x8)**: Finer pattern, more detailed

### ⚙️ Optional Scaling (Two Methods!)
- **Enable/Disable toggle**: Use just for color transformation or with resizing
- **Scaling Mode**:
  - **Rescale (Megapixels)**: Scale to target megapixel size (0.01 to 16.0 MP)
  - **Resize (Short Side)**: Scale by short side length (16 to 8192 pixels)
- Multiple scaling methods: lanczos, bicubic, bilinear, nearest, area
- Scales BEFORE color reduction for best quality

### 🎨 Pixelation Effect (Pixel Art Creator!)
- **Enable Pixelation**: Toggle pixel confinement on/off (default: **off**)
- **Block Size**: 1-32 pixels (default: **1** = no effect)
- Averages blocks of pixels to create larger "chunky" pixels
- Perfect for converting photos to pixel art
- Works best with dithering disabled for clean pixel boundaries

### 📺 CRT Scanlines Effect
- **Optional scanlines overlay**: Adds authentic CRT screen effect
- Darkens every other horizontal line for that classic arcade/TV look
- Perfect for authentic retro gaming aesthetic

## Parameters

### Required Inputs

- **image**: Input image to transform
- **color_mode**: Select target color palette or bit depth
- **dithering**: Choose dithering algorithm
- **enable_scaling**: Toggle image scaling on/off
- **scaling_mode**: Choose between "rescale (megapixels)" or "resize (short side)"
- **megapixels**: Target image size in megapixels (used when scaling_mode is "rescale")
- **resize**: Short side length in pixels (used when scaling_mode is "resize")
- **scaling_method**: Interpolation method for resizing
- **enable_pixelation**: Enable pixel confinement effect for pixel art
- **block_size**: Size of pixel blocks (1-32, default 1 = no pixelation)
- **scanlines**: Enable CRT-style scanlines effect

## Usage Tips

### Authentic NES Look with Scanlines
```
Color Mode: NES - Famicom (25 colors)
Dithering: Floyd-Steinberg
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 240
Scaling Method: nearest
Scanlines: True
```

### Classic Game Boy
```
Color Mode: Game Boy (4 shades)
Dithering: Ordered (Bayer 4x4)
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 144
Scaling Method: nearest
Scanlines: False
```

### Sega Genesis CRT Look
```
Color Mode: Genesis - Mega Drive (61 colors)
Dithering: Floyd-Steinberg or Ordered (Bayer 4x4)
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 224
Scaling Method: area
Scanlines: True
```

### Atari 2600 Retro
```
Color Mode: Atari 2600 (4 colors)
Dithering: None or Ordered (Bayer 4x4)
Enable Scaling: True
Scaling Mode: rescale (megapixels)
Megapixels: 0.15
Scaling Method: nearest
Scanlines: False
```

### High-Res SNES Style
```
Color Mode: SNES (256 colors)
Dithering: Floyd-Steinberg
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 512
Scaling Method: lanczos
Scanlines: True
```

### CGA Gaming Nostalgia
```
Color Mode: CGA (16 colors)
Dithering: Atkinson or Floyd-Steinberg
Enable Scaling: True
Megapixels: 0.3
Scaling Method: lanczos
```

### High-Quality Color Grading
```
Color Mode: 4096 Colors (12-bit) or 32768 Colors (15-bit)
Dithering: Floyd-Steinberg
Enable Scaling: False
Maintains original resolution with subtle vintage look
```

### Pixel Art Creation
```
Color Mode: CGA, EGA, or Game Boy
Dithering: None
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 240
Scaling Method: nearest
Enable Pixelation: True
Block Size: 4
Scanlines: False
```
**Creates chunky pixel art with clean color boundaries!**

### MacPaint Aesthetic
```
Color Mode: Monochrome (2 colors)
Dithering: Atkinson
Enable Scaling: False or True (for authentic Mac resolution)
Megapixels: 0.25
```

## Dithering Comparison

**For smooth gradients**: Floyd-Steinberg, JJN, Stucki
**For texture/grain**: Ordered (Bayer), Sierra
**For brightness preservation**: Atkinson
**For clean pixels**: None
**For speed**: None, Ordered (Bayer 4x4)
**For quality**: Jarvis-Judice-Ninke, Stucki

## Console-Specific Notes

### Atari 2600
Historically accurate! The Atari 2600 could only display **4 colors per scanline** - one for the background, one for the playfield, and one for each of the two player sprites. The palette is selected from a 128-color NTSC palette. This severe limitation is what gave Atari games their distinctive blocky, high-contrast look.

### NES - Famicom
The NES could display **25 colors on-screen simultaneously** from its palette of 52-54 unique colors. This was achieved through 8 palettes (4 for backgrounds, 4 for sprites) with 4 colors each, sharing one common background color. The characteristic bright, saturated colors defined a generation of gaming.

### Sega Master System
Could display **32 colors on-screen** (two 16-color palettes) from a total of 64 colors. Similar color depth to EGA but with different distribution.

### Game Boy
The original Game Boy used **4 shades of greenish-gray**. Try ordered dithering for that authentic handheld look with visible dither patterns!

### TurboGrafx-16
An underappreciated powerhouse! Could display **482 colors on-screen simultaneously** from a 512-color palette (9-bit RGB). This gave it superior color capabilities compared to other 16-bit consoles.

### Genesis - Mega Drive
Could display **61 colors on-screen** (4 palettes of 15 colors each, plus one shared color) from a 512-color palette. The hardware had slightly darker output than other consoles, giving Genesis games their characteristic look. Use ordered dithering for authentic Genesis look or Floyd-Steinberg for smoother gradients.

### SNES
The SNES could display **256 colors on-screen** from a massive 32,768-color palette (15-bit RGB). This gave it the richest, most vibrant colors of the 16-bit era.

### Game Boy Color
Could display **56 colors on-screen** (8 background palettes × 4 colors + 8 sprite palettes × 3 colors) from 32,768 possible colors (15-bit RGB). A huge upgrade from the original Game Boy!

## Scaling Mode Explained

### Rescale (Megapixels) - Default
- **What it does**: Scales image to a target megapixel count
- **When to use**: When you want precise control over total image size
- **Example**: 1.0 MP = 1,000,000 pixels (could be 1000×1000, 1280×780, etc.)
- **Best for**: General downsizing, maintaining aspect ratio with target file size

### Resize (Short Side)
- **What it does**: Scales image so the shorter dimension equals your target
- **When to use**: When you want specific vertical/horizontal resolution (like console output)
- **Example**: resize=240 on a 16:9 image → 426×240 (landscape) or 240×426 (portrait)
- **Best for**: Matching console resolutions (NES 240p, Genesis 224p, SNES 224p)

### Historical Console Resolutions
- **NES**: 256×240 (use resize=240)
- **Genesis**: 320×224 (use resize=224)
- **SNES**: 256×224 or 512×448 (use resize=224 or resize=448)
- **Game Boy**: 160×144 (use resize=144)
- **PSX**: 320×240 (use resize=240)

## Scanlines Effect

The scanlines effect darkens every other horizontal line to simulate CRT displays:
- **Arcade/CRT aesthetic**: Essential for authentic retro look
- **Best with**: Low resolution images (after scaling down)
- **Works great with**: Console palettes + nearest neighbor scaling
- **Not recommended**: High-resolution images (scanlines will be too subtle)

## Processing Order

1. **Scaling** (if enabled) - uses selected mode (rescale or resize) before color reduction
2. **Pixelation** (if enabled) - applies pixel confinement/block averaging
3. **Dithering + Color Reduction** - applies palette quantization with selected dithering method
4. **Scanlines** (if enabled) - applies CRT effect to final result

## Pixelation for Pixel Art

The pixelation feature is perfect for converting photos to pixel art:

**How it works:**
- Groups pixels into blocks (e.g., 4×4 blocks)
- Averages the color within each block
- Creates larger "chunky" pixels

**Best Settings for Pixel Art:**
```
Enable Pixelation: True
Block Size: 2-8 (start small!)
Dithering: None (for clean pixels)
Scaling Method: nearest
Color Mode: Game Boy, NES, or CGA
```

**Block Size Guide:**
- **Block 2**: Subtle pixelation, good for detailed images
- **Block 4**: Classic pixel art look
- **Block 8**: Very chunky, great for extreme pixel art
- **Block 16+**: Ultra-chunky abstract look

## Technical Details

- Processes images in batches
- Preserves aspect ratio during scaling
- Uses industry-standard dithering algorithms
- Accurate console palette recreation
- Returns images in ComfyUI tensor format (BHWC)

## Performance Notes

- Error diffusion dithering (Floyd-Steinberg, etc.) takes more time but produces better results
- Ordered dithering is faster, great for real-time preview
- Larger images take longer with error diffusion
- Scaling before color reduction produces better results
- JJN and Stucki are slower but highest quality

## Creative Applications

- **Retro Game Dev**: Preview how art will look on target console
- **Music Videos**: Nostalgic 8-bit/16-bit aesthetic
- **Web Design**: Unique retro web graphics
- **Pixel Art**: Downscale photos to pixel art with authentic palettes
- **Film Grading**: Subtle color reduction for vintage look
- **NFT Art**: Retro gaming aesthetic
- **Print Design**: Dithered halftone effects

## Category

`scg-utils`

## Display Name

**SCG Color Palette Transformer**

---

**Pro Tip**: Combine with the SCG Scale to Megapixels node beforehand if you want precise control over the input resolution, then use this node with scaling disabled for pure color transformation!

