# RAAG (Ratio Aware Adaptive Guidance)

A ComfyUI node that patches models to use **Ratio Aware Adaptive Guidance** during sampling, providing adaptive CFG scaling based on the relationship between conditioned and unconditioned predictions.

## Overview

RAAG dynamically adjusts the CFG (Classifier-Free Guidance) weight during sampling based on the ratio between the conditional delta and the unconditioned prediction magnitude. This can help:

- **Reduce oversaturation** at high CFG scales
- **Improve detail preservation** in generated images
- **Better balance** between prompt adherence and image quality
- **Adaptive behavior** that responds to the model's predictions

## How It Works

Traditional CFG uses a fixed weight:
```
output = uncond + cfg_scale * (cond - uncond)
```

RAAG uses an adaptive weight:
```
delta = cond - uncond
ratio = ||delta|| / ||uncond||
w = 1 + (w_max - 1) * exp(-alpha * ratio)
output = uncond + w * delta
```

The weight `w` automatically adjusts based on:
- **High ratio** (strong guidance needed): Weight approaches 1.0 (conservative)
- **Low ratio** (weak guidance): Weight approaches w_max (stronger guidance)

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `model` | MODEL | - | - | Input model to patch |
| `enable` | BOOLEAN | `True` | - | Enable/disable RAAG (pass-through when disabled) |
| `alpha` | FLOAT | `12.0` | 0.0 - 200.0 | Controls the sensitivity of the adaptive weighting. Higher values make the transition steeper |
| `w_max` | FLOAT | `18.0` | 1.0 - 200.0 | Maximum weight when ratio is low. Typically set higher than your target CFG scale |
| `eps` | FLOAT | `1e-8` | 0.0 - 1e-2 | Epsilon for numerical stability (prevents division by zero) |
| `only_when_cfg_gt_1` | BOOLEAN | `True` | - | Use linear CFG when cfg_scale ≤ 1.0, RAAG otherwise |

## Usage

### Basic Setup

1. Add the **RAAG (Ratio Aware Adaptive Guidance)** node to your workflow
2. Connect your model through the RAAG node
3. Use the patched model in your sampler (KSampler, SamplerCustomAdvanced, etc.)

```
Load Checkpoint → RAAG ModelPatch → KSampler → ...
                       ↓
                   Set Parameters
```

### Recommended Starting Values

For most cases, the defaults work well:
```
alpha: 12.0
w_max: 18.0
eps: 1e-8
only_when_cfg_gt_1: True
```

When using with high CFG scales (e.g., 10-15):
```
alpha: 15.0 - 20.0  (more aggressive adaptation)
w_max: 20.0 - 25.0  (higher ceiling)
```

For more subtle effects:
```
alpha: 8.0 - 10.0   (gentler adaptation)
w_max: 12.0 - 15.0  (lower ceiling)
```

## Example Workflows

### Example 1: High CFG with RAAG
```
Load Checkpoint
  ↓
RAAG ModelPatch (alpha=15.0, w_max=20.0)
  ↓
KSampler (cfg_scale=12.0)
  ↓
VAE Decode
```

**Result**: High prompt adherence without oversaturation

### Example 2: Standard Generation with RAAG
```
Load Checkpoint
  ↓
RAAG ModelPatch (defaults)
  ↓
KSampler (cfg_scale=7.0)
  ↓
VAE Decode
```

**Result**: Balanced generation with adaptive guidance

### Example 3: Disable RAAG for Comparison
```
Load Checkpoint
  ↓
RAAG ModelPatch (enable=False)
  ↓
KSampler
  ↓
VAE Decode
```

**Result**: Standard CFG behavior (useful for A/B testing)

## Tips and Best Practices

### Parameter Tuning

1. **Start with defaults** and adjust based on results
2. **Increase alpha** if you want more aggressive adaptation to high ratios
3. **Increase w_max** if you need stronger guidance at low ratios
4. **Decrease alpha** for more consistent behavior across different ratios

### When to Use RAAG

RAAG is particularly useful when:
- Using **high CFG scales** (>8.0) and experiencing oversaturation
- Working with **complex prompts** that need strong guidance
- Seeking **better detail preservation** at high CFG
- Experimenting with **extreme CFG values** (>15.0)

### When NOT to Use RAAG

Consider disabling RAAG when:
- Using **low CFG scales** (<3.0) where standard CFG works well
- Seeking **maximum prompt adherence** without any adaptive behavior
- Working with **specific models** that already handle high CFG well
- **Comparing** results with standard workflows

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Not enough guidance | Increase `w_max` |
| Too much adaptation | Decrease `alpha` |
| Oversaturation remains | Increase `alpha`, increase `w_max` |
| Weak prompt adherence | Increase `w_max`, decrease `alpha` |
| No visible effect | Ensure `enable=True` and CFG scale > 1.0 |

## Technical Details

### Implementation

The node patches `model.model_options["sampler_cfg_function"]`, which ComfyUI calls during each sampling step. The patched function:

1. Extracts conditioned (`cond`) and unconditioned (`uncond`) predictions
2. Calculates the delta: `delta = cond - uncond`
3. Computes per-sample L2 norms for stability in batches
4. Calculates the ratio: `||delta|| / ||uncond||`
5. Applies exponential weighting: `w = 1 + (w_max - 1) * exp(-alpha * ratio)`
6. Returns: `uncond + w * delta`

### Batch Stability

RAAG calculates norms per-sample in the batch, ensuring consistent behavior across different batch sizes:

```python
b = delta.shape[0]
delta_n = torch.norm(delta.reshape(b, -1), dim=1)
uncond_n = torch.norm(uncond.reshape(b, -1), dim=1).clamp_min(eps)
ratio = (delta_n / uncond_n).to(delta.dtype)
```

### Compatibility

- ✅ Works with **KSampler** and **SamplerCustomAdvanced**
- ✅ Compatible with **all samplers** (Euler, DPM++, etc.)
- ✅ Works with **all schedulers**
- ✅ Supports **batch processing**
- ✅ Compatible with other model patches (can be chained)

## Category

`scg-utils/sampling`

## Credits

RAAG is based on research into adaptive CFG scaling techniques. The implementation uses exponential weighting to create smooth transitions between different guidance strengths based on the model's prediction characteristics.

## Version

1.0.0
