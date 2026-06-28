# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **SCG LoRA Scheduler** (`SCGLoRAScheduler`) - Tapers a LoRA's *model* (unet)
  strength across the denoise trajectory, the same way SCG Conditioning Mixer
  tapers conditioning. Controls:
  - `strength_start` / `strength_end` - the model LoRA strength at the start and
    end of the window (ramp between them).
  - `start_percent` / `end_percent` - clean activation window over the denoise.
  - `interpolation` - `linear` / `ease_in` / `ease_out` / `ease_in_out` taper shape.
  - `cutoff_outside_window` - hold the LoRA at 0 outside `[start, end]` (clean
    start/end) or hold the endpoint strengths.
  - Optional `clip` (+ `strength_clip`): applied **statically** since CLIP is
    encoded once and cannot be scheduled (mirrors LoraLoaderModelOnly ignoring
    CLIP).
  - Uses ComfyUI's **bypass-mode** LoRA injection (`comfy.weight_adapter`): the
    LoRA is applied additively inside each layer's forward (never baking base
    weights), and a unet function wrapper updates each adapter's live multiplier
    per step. This works on quantized / dynamically loaded models (e.g. Krea2,
    fp8) where the hook/keyframe weight-swap path crashes on `weight_scale`-style
    keys. Multiple schedulers stack (each owns its own injection + wrapper).

- **SCG Reference Text Encoder Plus** - Single-reference-image variant of
  SCG TextEncoderQwenEditPlus. Keeps `vision_only`, the compose/edit/custom
  templates, `reference_latents_method`, and `vision_megapixels` (default 1.0),
  but takes a single image. For multi-image work, feed several of these into the
  SCG Conditioning Mixer instead of cramming images into one encoder.


- **SCG Conditioning Mixer** (`SCGConditioningTrajectory`, display name updated) -
  Single-node replacement for chained ConditioningAverage / ZeroOut /
  SetTimestepRange / Combine graphs. Mixes two conditionings (A/B), each with:
  - `strength` - scales the cond by averaging it toward a zeroed cond (the
    idiomatic ComfyUI cond-weight trick for damping one image vs. another).
  - `start`/`end` - timestep range the cond is active over (early steps set
    composition, late steps refine identity/style).
  - `merge_style` - `average` (weighted blend via `merge_strength`) or `combine`
    (union; both applied). `merge_strength` is ignored in combine mode.
  - Optional text-only branch: encodes a no-image cond from `clip` + `text_prompt`
    (with `compose`/`edit`/`custom` template selection) and merges it into the A/B
    result with its own strength, timestep range, and merge style.
  - Note: in `average` mode the blended pair inherits A's metadata/timestep range
    (matching ComfyUI's ConditioningAverage); use `combine` to honor both ranges.

### Changed (SCG Conditioning Mixer)
- Source strengths (`a_strength` / `b_strength` / `text_strength` and their taper
  targets) now go up to **2.0**, allowing amplified conditioning (great for
  transfer). `cond_dampen` no longer clamps at 1.0 - values above 1.0 scale the
  cond up (`cond * strength`). Merge weights stay 0..1 (they're blend ratios).
- `conditioning_b` is now optional - the node works as a single-input shaper
  (strength / taper / timestep range on A, plus optional text merge).
- Per-source strength **taper**: `a_taper` / `b_taper` / `text_taper` (`off` /
  `normal` = strength->target / `reverse` = target->strength) with a
  `*_taper_target` endpoint and a shared `taper_steps` smoothness control. Tapers
  ramp the cond's contribution across its own timestep window and honor the
  existing start/end and strength values.
- Fixed text-only merge: `text_merge_strength` is now intuitive (amount of text,
  1.0 = all text in average mode), and `text_start` defaults to 0.0 (full range)
  so the text actually contributes. Use `combine` to add text on top of blank
  image prompts; drive presence with `text_strength`.
- Added a web extension (progressive disclosure): A/B option groups appear when
  their input is connected, merge controls appear with a second input, taper
  targets/steps appear only when a taper is active, and text options hide when
  `include_text_only` is off (text_prompt / text_custom_template stay visible).
- All conditioning inputs are now optional. With zero conditioning inputs the
  mixer acts as a fancy text encoder: it encodes the text prompt (even empty) and
  applies the text strength / taper / start-stop; merge controls and
  `include_text_only` are ignored in that mode. The web extension hides the merge
  and toggle controls accordingly.

### Changed
- **SCG TextEncoderQwenEditPlus** - Major overhaul for multi-image conditioning control
  - Added `reference_latents_method` (default `index`) to control how reference
    latents are positioned in the diffusion model. `offset` tiles references
    side-by-side ("brady box"); `index` stacks them on a separate axis so the
    model is more likely to merge concepts. Also exposes `index_timestep_zero`,
    `negative_index`, and `auto (model default)`.
  - Added a per-image mode (`image1_mode`..`image4_mode`): `reference + vision`,
    `vision only` (semantic concept, no structural reference latent),
    `reference only`, or `disabled`. Lets you anchor structure on one image while
    pulling concepts/style from others.
  - Added `instruction_template` (`compose` / `edit` / `custom`) with a new
    composition-oriented system prompt that instructs the model to synthesize a
    single unified image instead of arranging references in a grid.
  - Added `vision_megapixels` to tune the VL (semantic) encode resolution
    (default 1.0 MP).
  - Added a `vision_only` master toggle that forces every image through the vision
    channel and skips all VAE / reference-latent work. Intended for vision-only
    models (e.g. Krea-2) where reference latents and `reference_latents_method` are
    no-ops; also saves the VAE encode cost. Overrides the per-image modes.
  - Vision "Picture N" numbering is now sequential across enabled images (no gaps
    when an image is disabled).

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




