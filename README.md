# SCG-Utils - ComfyUI Custom Nodes Collection

A collection of handy ComfyUI nodes for image manipulation, retro effects, AI integration, and more.

<img width="1465" height="834" alt="image" src="https://github.com/user-attachments/assets/890432fb-a03c-4c86-adfe-b19365d298b3" />


## 🌟 Featured Nodes

### 🎨 Retro Color & Pixel Art (NEW!)

**SCG Color Palette Transformer** - Advanced retro color transformation
- 24 color modes: Game consoles, PC graphics (CGA/EGA/VGA), and bit depth modes
- 9 dithering algorithms (Floyd-Steinberg, Atkinson, JJN, Stucki, Burkes, Sierra, Ordered Bayer)
- Pixelation effect for pixel art creation (block size 1-32)
- Two scaling modes: rescale by megapixels or resize by short side
- CRT scanlines effect
- Perfect for retro gaming aesthetics and pixel art

**SCG Console Stylizer** - One-click authentic console looks
- 9 game console presets (Atari 2600 to PlayStation)
- Automatically applies correct palette, resolution, and settings
- Pixelation effect with block size control
- Adjustable scanline intensity
- Simplified interface for quick retro styling

[→ See detailed comparison and usage guides](NODE_COMPARISON.md)

## 📦 All Nodes

### Color & Effects
- **SCG Color Palette Transformer** - Advanced retro color palette transformation
- **SCG Console Stylizer** - One-click authentic console aesthetics

### Image Utilities
- **SCG Zeroed Outputs** - Provides zeroed/empty outputs for placeholder inputs
- **SCG Image Stack** - Stack up to 4 images in customizable grid layouts
- **SCG Image Stack XL** - Extended version supporting up to 8 images
- **SCG Scale to Megapixel Size** - Scale images to specific megapixel targets

### Resolution & Layout
- **SCG Resolution Selector** - Flexible resolution calculator with aspect ratio presets

### Masking & Inpainting
- **SCG Trim Image to Mask** - Trim images to masked content with context expansion
- **SCG Stitch Inpaint Image** - Stitch inpainted crops back into originals

### AI & LLM Integration
- **SCG Remote LLM/VLM - OAI Standard** - Connect to OpenAI-compatible APIs (OpenAI, LM Studio, etc.)
- **SCG TextEncoderQwenEditPlus** - Qwen image editing with up to 4 reference images
- **RAAG Model Patch** - Ratio Aware Adaptive Guidance for diffusion models

### Text Processing
- **SCG Wildcard Variable Processor** - Replace wildcard variables in text strings

### Logic Utilities
- **SCG Flip Boolean** - Invert boolean values (True → False, False → True)

## 🚀 Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "scg-utils"
3. Click Install

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SanDiegoDude/scg-utils.git
cd scg-utils
pip install -r requirements.txt
```

Restart ComfyUI and the nodes will appear in the "scg-utils" category.

## 📖 Documentation

Detailed documentation is available for key features:

- **[Color Palette Transformer Guide](COLOR_PALETTE_TRANSFORMER.md)** - Full documentation for the advanced color transformation node
- **[Console Stylizer Guide](CONSOLE_STYLIZER.md)** - Quick guide for one-click console styling
- **[Console Color Specifications](CONSOLE_COLOR_SPECS.md)** - Historically accurate console technical specs
- **[Node Comparison](NODE_COMPARISON.md)** - Detailed comparison between Color Palette Transformer and Console Stylizer
- **[RAAG Documentation](RAAG_README.md)** - Ratio Aware Adaptive Guidance usage
- **[Remote LLM/VLM Guide](REMOTE_LLM_VLM_README.md)** - Connect to OpenAI-compatible APIs

## 🎯 Quick Start Examples

### Create NES-Style Pixel Art
```
Use: SCG Console Stylizer
Console: NES - Famicom
Enable Scaling: True
Enable Dithering: False
Enable Pixelation: True
Block Size: 4
```

### Authentic Game Boy Look with CRT Effect
```
Use: SCG Console Stylizer
Console: Game Boy
Enable Scaling: True
Enable Scanlines: True
Scanline Intensity: 0.7
```

### Custom Retro Color Grading
```
Use: SCG Color Palette Transformer
Color Mode: 4096 Colors (12-bit)
Dithering: Atkinson
Scaling Mode: rescale (megapixels)
Megapixels: 2.0
```

## 🎮 Supported Consoles & Systems

### Game Consoles (Historically Accurate!)
- Atari 2600 (4 colors, 160 lines)
- NES/Famicom (25 colors, 240p)
- Sega Master System (32 colors, 192p)
- Game Boy (4 shades, 144p)
- TurboGrafx-16/PC Engine (482 colors, 224p)
- Sega Genesis/Mega Drive (61 colors, 224p)
- SNES (256 colors, 224p)
- Game Boy Color (56 colors, 144p)
- PlayStation (256 colors, 240p)

### PC Graphics Modes
- CGA (16 colors)
- EGA (64 colors)
- VGA (256 colors)

### Bit Depth Modes
- Monochrome (2 colors)
- 4 Colors (2-bit)
- 8 Colors (3-bit)
- 16 Colors (4-bit palette)
- 64 Colors (6-bit palette)
- 256 Colors (8-bit palette)
- 512 Colors (9-bit, 3-bit per channel)
- 4096 Colors (12-bit, 4-bit per channel)
- 32768 Colors (15-bit, 5-bit per channel)
- 65536 Colors (16-bit, 5-6-5)
- 262,144 Colors (18-bit, 6-bit per channel)
- 16.7M Colors (24-bit, 8-bit per channel, true color)

## 🎨 Features Highlights

### Pixelation Effect
Convert photos to pixel art with block-based pixel confinement:
- Adjustable block size (1-32 pixels)
- Averages color within blocks for smooth chunky pixels
- Perfect for pixel art creation

### Dithering Algorithms
9 professional dithering methods:
- **Floyd-Steinberg** - Classic, smooth gradients
- **Atkinson** - MacPaint style, brightness preserving
- **Jarvis-Judice-Ninke** - High quality, wide error distribution
- **Stucki** - Similar to JJN with different weights
- **Burkes** - Good balance of speed and quality
- **Sierra** - Three-row error distribution
- **Ordered Bayer 4×4** - Pattern-based, fast
- **Ordered Bayer 8×8** - Finer pattern
- **None** - Clean posterization

### CRT Scanlines
Authentic CRT display simulation:
- Adjustable intensity (0.0-1.0)
- Perfect for arcade/TV aesthetics
- Works great with low-resolution images

## 🔧 Requirements

- ComfyUI
- Python 3.8+
- Pillow >= 9.0.0
- requests >= 2.31.0

Note: PyTorch, NumPy, and other core dependencies are already included with ComfyUI.

## 💡 Use Cases

- **Retro Game Development** - Preview art on target hardware
- **Pixel Art Creation** - Convert photos to authentic pixel art
- **Music Videos** - Nostalgic 8-bit/16-bit aesthetics
- **Social Media Content** - Unique retro visuals
- **Game Demakes** - Recreate modern games in retro styles
- **Film Grading** - Vintage color reduction effects
- **NFT Art** - Retro gaming aesthetic
- **Print Design** - Dithered halftone effects

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Console specifications verified against historical hardware documentation
- Dithering algorithms based on industry-standard implementations
- Palette data from NESdev, SegaRetro, SNESdev, and original hardware manuals

## 📧 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Check the documentation files for detailed guides

## 🔗 Links

- [GitHub Repository](https://github.com/SanDiegoDude/scg-utils)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

**Made with ❤️ for the ComfyUI community**

YMMV - Your Mileage May Vary! 🚗





