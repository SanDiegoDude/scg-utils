# Console Color Specifications - Historically Accurate!

This document details the **accurate on-screen color limitations** for each game console in the SCG Color Palette Transformer node.

## 🎮 Console Specifications

### Atari 2600 (1977)
- **Total Palette**: 128 colors (NTSC)
- **On-Screen**: **4 colors per scanline**
- **Details**: 1 background + 1 playfield + 2 sprites
- **Why so few?**: The TIA chip had only 4 color registers that could be set per scanline
- **Our Implementation**: 4-color palette representing typical usage

### NES / Famicom (1983/1985)
- **Total Palette**: 52-54 unique colors (from 64 with duplicates)
- **On-Screen**: **25 colors simultaneously**
- **Details**: 8 palettes (4 BG + 4 sprite) × 4 colors each, sharing 1 background color
- **Math**: (8 palettes × 4 colors) - 7 shared = 25 unique colors
- **Our Implementation**: 25-color palette with typical NES colors

### Sega Master System (1985)
- **Total Palette**: 64 colors (6-bit RGB, 2 bits per channel)
- **On-Screen**: **32 colors simultaneously**
- **Details**: 2 palettes of 16 colors each
- **Our Implementation**: 32-color well-distributed palette

### Game Boy (1989)
- **Total Palette**: 4 shades of green
- **On-Screen**: **4 shades**
- **Details**: Monochrome display with 4 shades of green-tinted gray
- **Our Implementation**: Authentic 4-shade green palette

### TurboGrafx-16 / PC Engine (1987/1989)
- **Total Palette**: 512 colors (9-bit RGB, 3 bits per channel)
- **On-Screen**: **482 colors simultaneously** 
- **Details**: 32 palettes (16 BG + 16 sprite) × 16 colors, with shared/transparent colors
- **Math**: 241 background + 240 sprite + 1 shared = 482
- **Why impressive**: More on-screen colors than Genesis or SNES!
- **Our Implementation**: 482-color palette from 9-bit color space

### Sega Genesis / Mega Drive (1988/1989)
- **Total Palette**: 512 colors (9-bit RGB, 3 bits per channel)
- **On-Screen**: **61 colors simultaneously** (normal mode)
- **Details**: 4 palettes × 15 colors + 1 shared transparent color
- **Advanced Mode**: 183-192 colors with shadow/highlight mode (rarely used)
- **Note**: Slightly darker output than other consoles due to hardware
- **Our Implementation**: 61-color well-distributed palette with 0.9× darkening factor

### SNES (1990/1991)
- **Total Palette**: 32,768 colors (15-bit RGB, 5 bits per channel)
- **On-Screen**: **256 colors simultaneously**
- **Details**: Color RAM (CGRAM) holds 256 colors from 32,768 possible
- **Why powerful**: Could pick any 256 from 32K palette, richest colors of 16-bit era
- **Our Implementation**: 256-color well-distributed palette

### Game Boy Color (1998)
- **Total Palette**: 32,768 colors (15-bit RGB, 5 bits per channel)
- **On-Screen**: **56 colors simultaneously**
- **Details**: 8 BG palettes × 4 colors + 8 sprite palettes × 3 colors (transparency)
- **Math**: (8 × 4) + (8 × 3) = 32 + 24 = 56 colors
- **Our Implementation**: 56-color well-distributed palette from 15-bit space

### PlayStation (1994)
- **Total Palette**: 16.7 million colors (24-bit RGB)
- **8-bit CLUT Mode**: **256 colors on-screen**
- **Details**: 8-bit Color Look-Up Table mode, commonly used for backgrounds
- **Also Supported**: 15-bit (32K colors) and 24-bit (16.7M colors) direct color modes
- **Our Implementation**: 256-color VGA-style palette

## 📊 Comparison Chart

| Console | Year | On-Screen | Total Palette | Architecture |
|---------|------|-----------|---------------|--------------|
| Atari 2600 | 1977 | **4** | 128 | Hue + Luminance |
| NES | 1985 | **25** | 52-54 | Custom palette |
| Master System | 1985 | **32** | 64 | 6-bit RGB |
| Game Boy | 1989 | **4** | 4 | Monochrome |
| TurboGrafx-16 | 1989 | **482** | 512 | 9-bit RGB |
| Genesis | 1989 | **61** | 512 | 9-bit RGB |
| SNES | 1991 | **256** | 32,768 | 15-bit RGB |
| Game Boy Color | 1998 | **56** | 32,768 | 15-bit RGB |
| PlayStation | 1994 | **256** (CLUT) | 16.7M | 24-bit RGB |

## 🎨 Why These Numbers Matter

### For Authentic Retro Look
Using the accurate on-screen color counts creates **historically accurate** results. For example:
- **Atari 2600** with 4 colors gives you that blocky, high-contrast early 80s look
- **NES** with 25 colors captures the characteristic limited-but-vibrant 8-bit aesthetic
- **Genesis** with 61 colors shows why games had that slightly muted, darker tone compared to SNES

### For Game Development
If you're developing retro-style games or demakes, these accurate limits help you:
- Preview how art will look on target hardware
- Stay within authentic color budgets
- Understand historical limitations that shaped game design

### The TurboGrafx-16 Surprise
Many people don't realize the TurboGrafx-16 could display **482 colors on-screen** - more than Genesis (61) or even SNES (256) in terms of simultaneous colors! This made it capable of very colorful, vibrant graphics.

## 🔍 Sources

All specifications verified against:
- Hardware technical documentation
- NESdev Wiki (nesdev.org)
- SegaRetro (segaretro.org)
- SNESdev Wiki (snes.nesdev.org)
- Original hardware manuals and programming guides

## 🎯 Implementation Notes

Our palette generation algorithms create **well-distributed color selections** from the available on-screen colors:

- **Fixed palettes** (Atari, Game Boy, NES): Use historically accurate color values
- **Generated palettes** (SMS, Genesis, TurboGrafx, SNES, GBC): Mathematically distribute colors across the available color space
- **Genesis darkening**: Applies 0.9× multiplier to RGB values to simulate the characteristic darker output

This ensures both historical accuracy AND practical usability for image transformation!

