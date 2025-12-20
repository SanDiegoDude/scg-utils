# Node Comparison: Color Palette Transformer vs Console Stylizer

## Quick Reference

### SCG Console Stylizer 🎮
**One-click authentic console looks**

- ✅ 5 simple parameters
- ✅ Automatic console-accurate settings
- ✅ Just pick a console and go
- ✅ Perfect for authentic retro looks
- ✅ Best for: Game dev, retro content, quick styling

### SCG Color Palette Transformer 🎨
**Full manual control over everything**

- ✅ 9 detailed parameters
- ✅ 18 color modes (consoles + PC + bit depths)
- ✅ 9 dithering algorithms
- ✅ 2 scaling modes (rescale + resize)
- ✅ Best for: Experimentation, PC graphics, custom bit depths

---

## Parameter Comparison

| Parameter | Console Stylizer | Color Palette Transformer |
|-----------|------------------|---------------------------|
| **Console/Mode Selection** | 9 consoles only | 18 modes (consoles + PC + bit depths) |
| **Color Palette** | Auto-selected | Manual selection |
| **Dithering** | On/Off toggle (auto algorithm) | Choose from 9 algorithms |
| **Scaling** | On/Off toggle (auto resolution) | Choose mode + value |
| **Scaling Mode** | Auto (resize by short side) | Rescale OR Resize |
| **Resolution** | Auto per console | Manual (megapixels OR pixels) |
| **Scaling Method** | Auto per console | Choose from 5 methods |
| **Scanlines** | On/Off + intensity slider | On/Off (fixed intensity) |

---

## Feature Matrix

| Feature | Console Stylizer | Color Palette Transformer |
|---------|------------------|---------------------------|
| **Game Consoles** | ✅ 9 presets | ✅ Manual config |
| **PC Graphics Modes** | ❌ | ✅ CGA, EGA, VGA |
| **Bit Depth Modes** | ❌ | ✅ 2-bit to 16-bit |
| **Custom Dithering** | ❌ | ✅ 9 algorithms |
| **Rescale (Megapixels)** | ❌ | ✅ |
| **Resize (Pixels)** | ✅ Auto | ✅ Manual |
| **Scanline Intensity** | ✅ 0.0-1.0 | ❌ Fixed 0.5 |
| **Preset Accuracy** | ✅ Built-in | ❌ Manual |

---

## Console Settings Reference

What Console Stylizer automatically applies:

| Console | Resolution | Dithering | Scaling Method |
|---------|------------|-----------|----------------|
| Atari 2600 | 160 lines | Ordered 4x4 | Nearest |
| NES | 240p | Floyd-Steinberg | Nearest |
| Master System | 192p | Floyd-Steinberg | Nearest |
| Game Boy | 144p | Ordered 4x4 | Nearest |
| TurboGrafx-16 | 224p | Floyd-Steinberg | Area |
| Genesis | 224p | Ordered 4x4 | Nearest |
| SNES | 224p | Floyd-Steinberg | Lanczos |
| GBC | 144p | Floyd-Steinberg | Nearest |
| PlayStation | 240p | Floyd-Steinberg | Bilinear |

---

## When to Use Each Node

### Use Console Stylizer When You Want:

1. **Authentic console look with one click**
   - No need to research console specs
   - Historically accurate presets
   - Fast workflow

2. **Game development preview**
   - See how art looks on target hardware
   - Accurate color counts and resolutions
   - Quick iteration

3. **Retro content creation**
   - Music videos
   - Social media posts
   - Nostalgic aesthetics

4. **Simplicity**
   - Don't want to choose from 9 dithering algorithms
   - Don't need to remember console resolutions
   - Just want it to look right

### Use Color Palette Transformer When You Want:

1. **PC graphics modes**
   - CGA (16 colors)
   - EGA (64 colors)  
   - VGA (256 colors)

2. **Custom bit depths**
   - Monochrome (2 colors)
   - 4 Colors (2-bit)
   - 8 Colors (3-bit)
   - 4096 Colors (12-bit)
   - 32768 Colors (15-bit)
   - 65536 Colors (16-bit)

3. **Experimentation**
   - Try different dithering algorithms
   - Test various scaling methods
   - Mix and match settings

4. **Specific megapixel targets**
   - Need exact megapixel count
   - Custom resolution requirements
   - Fine-tuned file sizes

5. **Maximum control**
   - Override console defaults
   - Create custom combinations
   - Advanced workflows

---

## Example Workflows

### Workflow 1: Quick NES Look
**Node**: Console Stylizer
```
Console: NES
All toggles: On
Scanline Intensity: 0.5
```
**One node, done!**

### Workflow 2: Same Result with Color Palette
**Node**: Color Palette Transformer  
```
Color Mode: NES - Famicom (25 colors)
Dithering: Floyd-Steinberg
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 240
Scaling Method: nearest
Scanlines: True
```
**Same result, more steps**

### Workflow 3: Custom Artistic Look
**Node**: Color Palette Transformer (only option for this)
```
Color Mode: 4096 Colors (12-bit)
Dithering: Jarvis-Judice-Ninke
Enable Scaling: True
Scaling Mode: rescale (megapixels)
Megapixels: 2.5
Scaling Method: lanczos
Scanlines: False
```
**Not possible with Console Stylizer**

### Workflow 4: CGA DOS Game Look
**Node**: Color Palette Transformer (only option)
```
Color Mode: CGA (16 colors)
Dithering: Atkinson
Enable Scaling: True
Scaling Mode: resize (short side)
Resize: 200
Scaling Method: nearest
Scanlines: False
```
**PC graphics mode only in Color Palette Transformer**

---

## Summary

**TL;DR:**

- **Console Stylizer**: Simple, fast, authentic console looks
- **Color Palette Transformer**: Advanced, flexible, all options available

**Most users should start with Console Stylizer** for authentic retro looks, then switch to Color Palette Transformer when they need PC modes, custom bit depths, or want to experiment with settings.

Both nodes use the same underlying engine, so quality is identical - it's just about ease of use vs flexibility!

---

## Both Nodes Are Great For:

- ✅ Retro gaming aesthetics
- ✅ Pixel art creation
- ✅ Nostalgic content
- ✅ Game development
- ✅ Limited color palettes
- ✅ Dithering effects
- ✅ Batch processing
- ✅ High-quality results

Choose based on your workflow and how much control you need! 🎮🎨

