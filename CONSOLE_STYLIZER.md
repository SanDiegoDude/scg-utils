# SCG Console Stylizer

One-click authentic game console aesthetics! Automatically applies the correct palette, resolution, and dithering for each console platform.

## Overview

The **SCG Console Stylizer** is a simplified, focused version of the Color Palette Transformer designed specifically for creating historically accurate console graphics. Just pick a console, and the node automatically applies all the correct settings - no need to remember which palette, resolution, or dithering method each console used!

## Why Use This Instead of Color Palette Transformer?

- **Simpler**: Only 5 parameters vs 9 parameters
- **Authentic**: Preset with historically accurate settings for each console
- **Faster**: Just pick a console and go
- **Correct**: No need to look up technical specs - it's all built-in
- **Flexible**: Can still disable scaling, dithering, or scanlines if desired

## Features

### 🎮 Console Presets

Each console comes with authentic settings automatically applied:

| Console | Resolution | Colors | Dithering | Scaling Method |
|---------|------------|--------|-----------|----------------|
| **Atari 2600** | 160 lines | 4 colors | None (clean pixels) | Nearest |
| **NES - Famicom** | 240p | 25 colors | None (clean pixels) | Nearest |
| **Sega Master System** | 192p | 32 colors | None (clean pixels) | Nearest |
| **Game Boy** | 144p | 4 shades | Ordered (Bayer 4x4) | Nearest |
| **TurboGrafx-16** | 224p | 482 colors | Floyd-Steinberg | Area |
| **Genesis - Mega Drive** | 224p | 61 colors | Ordered (Bayer 4x4) | Nearest |
| **SNES** | 224p | 256 colors | Floyd-Steinberg | Lanczos |
| **Game Boy Color** | 144p | 56 colors | Floyd-Steinberg | Nearest |
| **PlayStation** | 240p | 256 colors | Floyd-Steinberg | Bilinear |

### ⚙️ Toggleable Features

- **Enable Scaling**: Toggle console-accurate resolution on/off (default: **on**)
- **Enable Dithering**: Toggle dithering on/off (default: **on**)
- **Enable Pixelation**: Apply pixel confinement/block averaging for pixel art (default: **off**)
- **Block Size**: Size of pixel blocks (1-32, default: **1** = no effect)
- **Enable Scanlines**: Add CRT scanline effect (default: **off**)
- **Scanline Intensity**: Control scanline darkness (0.0-1.0, default: **0.5**)

## Parameters

### Required Inputs

- **image**: Input image to transform
- **console**: Target console platform (dropdown)
- **enable_scaling**: Scale to console resolution (boolean)
- **enable_dithering**: Apply console-appropriate dithering (boolean)
- **enable_pixelation**: Apply pixel confinement effect (boolean)
- **block_size**: Size of pixel blocks for pixelation (1-32 integer)
- **enable_scanlines**: Add CRT scanlines effect (boolean)
- **scanline_intensity**: Strength of scanline effect (0.0-1.0 slider)

## Usage Examples

### Quick & Easy - Full Authentic Look
```
Console: SNES
Enable Scaling: True
Enable Dithering: True
Enable Scanlines: True
Scanline Intensity: 0.5
```
**Result**: Perfect SNES look with 224p resolution, 256-color palette, dithering, and CRT scanlines!

### High-Res Console Palette
```
Console: Genesis - Mega Drive
Enable Scaling: False
Enable Dithering: True
Enable Scanlines: False
Scanline Intensity: 0.5
```
**Result**: Genesis 61-color palette with ordered dithering at original image resolution

### Clean Pixel Art Base
```
Console: NES - Famicom
Enable Scaling: True
Enable Dithering: False
Enable Scanlines: False
Scanline Intensity: 0.5
```
**Result**: NES 25-color palette at 240p with no dithering for clean pixel boundaries

### Maximum Retro Authenticity
```
Console: Game Boy
Enable Scaling: True
Enable Dithering: True
Enable Scanlines: True
Scanline Intensity: 0.7
```
**Result**: Full Game Boy experience with 144p, 4-shade green palette, ordered dithering, and strong scanlines!

### Just the Palette
```
Console: TurboGrafx-16
Enable Scaling: False
Enable Dithering: False
Enable Scanlines: False
Scanline Intensity: 0.5
```
**Result**: Only apply the 482-color TurboGrafx palette, keep original resolution and no effects

## Console-Specific Details

### Atari 2600 (1977)
- **160 lines** (lowest resolution for that blocky retro look)
- **4 colors** (background + playfield + 2 sprites)
- **No dithering** (clean, hard-edged pixels - consoles didn't have hardware dithering)
- **Nearest scaling** for sharp pixels

### NES - Famicom (1983/1985)
- **240p** (NTSC standard)
- **25 colors** on-screen
- **No dithering by default** (NES had no hardware dithering; some games used manual dither patterns in art)
- **Nearest scaling** for pixel-perfect look
- **Note**: Enable dithering manually if you want smooth gradients (non-authentic but artistic)

### Sega Master System (1985)
- **192p** resolution
- **32 colors** on-screen
- **No dithering by default** (no hardware dithering support)
- **Nearest scaling**

### Game Boy (1989)
- **144p** (original Game Boy screen)
- **4 shades of green**
- **Ordered dithering** for that characteristic look
- **Nearest scaling** for authentic pixels

### TurboGrafx-16 / PC Engine (1987/1989)
- **224p** resolution
- **482 colors** (most colorful 16-bit console!)
- **Floyd-Steinberg dithering** for smooth gradients
- **Area scaling** for better downscaling quality

### Genesis - Mega Drive (1988/1989)
- **224p** resolution
- **61 colors** on-screen with darker output
- **Ordered dithering** for characteristic Genesis texture
- **Nearest scaling** for pixel accuracy

### SNES (1990/1991)
- **224p** resolution
- **256 colors** from 32K palette
- **Floyd-Steinberg dithering** for rich gradients
- **Lanczos scaling** for highest quality

### Game Boy Color (1998)
- **144p** resolution
- **56 colors** on-screen
- **Floyd-Steinberg dithering**
- **Nearest scaling**

### PlayStation (1994)
- **240p** resolution
- **256-color CLUT mode**
- **Floyd-Steinberg dithering**
- **Bilinear scaling** (characteristic PSX look)

## Pixelation Effect (Pixel Confinement)

The pixelation feature averages blocks of pixels to create larger "pixel blocks", perfect for pixel art creation:

- **Block Size 1**: No effect (1:1 pixel mapping)
- **Block Size 2**: 2×2 pixel blocks (each 4 pixels become 1 averaged color)
- **Block Size 4**: 4×4 pixel blocks (each 16 pixels become 1 averaged color)
- **Block Size 8**: 8×8 pixel blocks (great for chunky pixel art)
- **Block Size 16+**: Very large blocks for extreme pixelation

**How it works**: 
1. Groups pixels into blocks of N×N size
2. Averages the color within each block
3. Applies the averaged color to all pixels in that block

**Best used with**:
- Scaling enabled (downscale first for better control)
- Dithering disabled (for clean pixel art)
- Nearest neighbor scaling method

**Pro Tips**:
- Start with small block sizes (2-4) for subtle effect
- Combine with console palettes for authentic retro pixel art
- Apply pixelation BEFORE color reduction for best results
- Higher block sizes work better on already-small images

## Scanline Effect

The scanline effect simulates CRT displays by darkening every other horizontal line:

- **Intensity 0.0**: No effect (scanlines disabled)
- **Intensity 0.3**: Subtle scanlines (good for high-res images)
- **Intensity 0.5**: Balanced scanlines (default, good for most use cases)
- **Intensity 0.7**: Strong scanlines (good for authentic arcade look)
- **Intensity 1.0**: Maximum darkness (very strong effect)

**Pro Tip**: Scanlines look best when combined with scaling enabled. At original high resolutions, scanlines may be too subtle to notice.

### Pixel Art Creation
```
Console: Game Boy or NES
Enable Scaling: True
Enable Dithering: False
Enable Pixelation: True
Block Size: 4
Enable Scanlines: False
Scanline Intensity: 0.5
```
**Result**: Perfect for converting photos to pixel art with console-accurate palettes!

## Processing Order

1. **Scaling** (if enabled) - Scales to console-specific resolution
2. **Pixelation** (if enabled) - Applies pixel confinement/block averaging
3. **Color Reduction** - Applies console palette with appropriate dithering
4. **Scanlines** (if enabled) - Adds CRT effect with custom intensity

## Comparison: Console Stylizer vs Color Palette Transformer

### Use Console Stylizer When:
- ✅ You want authentic console look with one click
- ✅ You're not sure what settings each console used
- ✅ You want simplicity and speed
- ✅ You're making retro game art or nostalgic content

### Use Color Palette Transformer When:
- ✅ You need fine control over every parameter
- ✅ You want to use PC graphics modes (CGA, EGA, VGA)
- ✅ You need custom bit-depth modes (2-bit, 3-bit, 12-bit, etc.)
- ✅ You want to experiment with different dithering algorithms
- ✅ You need rescale (megapixels) or custom resize values

## Tips & Tricks

### For Game Development
Use this node to preview how your art will look on target hardware!
```
Console: Your target platform
Enable Scaling: True
Enable Dithering: True
Enable Scanlines: False (add in post if needed)
```

### For Pixel Art
Start with a clean base:
```
Console: NES or Game Boy
Enable Scaling: True
Enable Dithering: False
Enable Scanlines: False
Then manually edit pixels
```

### For Music Videos / Content
Maximum nostalgia:
```
Console: Genesis or SNES
Enable Scaling: True
Enable Dithering: True
Enable Scanlines: True
Scanline Intensity: 0.6-0.7
```

### For Print / High-Res
Just the palette:
```
Console: Any
Enable Scaling: False
Enable Dithering: True or False (your choice)
Enable Scanlines: False
```

## Performance Notes

- Faster than Color Palette Transformer (fewer calculations)
- Dithering adds processing time but greatly improves quality
- Scanlines are very fast (simple tensor operation)
- Batch processing supported (processes all images efficiently)

## Technical Details

- Uses the same underlying palette engine as Color Palette Transformer
- All color counts and resolutions are historically accurate
- Dithering algorithms are industry-standard implementations
- Scaling maintains aspect ratio (scales by short side)
- Scanlines applied in GPU tensor space (efficient)

## Category

`scg-utils`

## Display Name

**SCG Console Stylizer**

---

**Perfect for**: Retro game developers, pixel artists, content creators, and anyone who wants authentic console aesthetics without the hassle of remembering technical specifications!

