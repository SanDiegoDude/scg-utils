# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-19

### Added
- **SCG Color Palette Transformer** - Advanced retro color transformation node
  - 18 color modes (9 game consoles, 3 PC graphics modes, 6 bit depth modes)
  - 9 dithering algorithms (Floyd-Steinberg, Atkinson, JJN, Stucki, Burkes, Sierra, Ordered Bayer 4×4, Ordered Bayer 8×8, None)
  - Pixelation effect for pixel art creation (block size 1-32)
  - Two scaling modes: rescale by megapixels or resize by short side
  - CRT scanlines effect
  - Historically accurate console color palettes

- **SCG Console Stylizer** - One-click authentic console styling
  - 9 game console presets with accurate settings
  - Automatic palette, resolution, and dithering configuration
  - Pixelation effect with block size control
  - Adjustable scanline intensity (0.0-1.0)
  - Simplified interface for quick results

- **Image Utilities**
  - SCG Zeroed Outputs - Placeholder outputs for all data types
  - SCG Image Stack - Stack up to 4 images in grids
  - SCG Image Stack XL - Stack up to 8 images in grids
  - SCG Scale to Megapixel Size - Scale by megapixel target

- **Resolution & Layout**
  - SCG Resolution Selector - Flexible resolution calculator

- **Masking & Inpainting**
  - SCG Trim Image to Mask - Trim images to masked content
  - SCG Stitch Inpaint Image - Stitch inpainted crops back

- **AI & LLM Integration**
  - SCG Remote LLM/VLM - OAI Standard - Connect to OpenAI-compatible APIs
  - SCG TextEncoderQwenEditPlus - Qwen editing with reference images
  - RAAG Model Patch - Ratio Aware Adaptive Guidance

- **Text Processing**
  - SCG Wildcard Variable Processor - Replace wildcard variables

### Documentation
- Comprehensive README with examples and quick start guide
- COLOR_PALETTE_TRANSFORMER.md - Full documentation for color node
- CONSOLE_STYLIZER.md - Quick guide for console styling
- CONSOLE_COLOR_SPECS.md - Historical console specifications
- NODE_COMPARISON.md - Detailed comparison between color nodes
- RAAG_README.md - RAAG usage documentation
- REMOTE_LLM_VLM_README.md - LLM/VLM integration guide

### Technical
- Historically accurate console specifications researched and verified
- Industry-standard dithering algorithm implementations
- Efficient GPU-accelerated processing with PyTorch
- Batch processing support for all nodes
- Clean, modular code architecture

[1.0.0]: https://github.com/SanDiegoDude/scg-utils/releases/tag/v1.0.0




