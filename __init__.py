from .nodes import (
    SCGZeroedOutputs,
    SCGImageStack,
    SCGImageStackXL,
    SCGWildcardVariableProcessor,
    SCGScaleToMegapixels,
    SCGScaleDimensionToSize,
    SCGResolutionSelector,
    SCGTrimImageToMask,
    SCGStitchInpaintImage,
    SCGFlipBoolean,
    SCGFormatInteger,
    SCGEvaluateIntMath,
    SCGEvaluateFloatMath,
)
from .nodes_qwen import SCGTextEncoderQwenEditPlus
from .nodes_remote_llm import SCGRemoteLLMVLM_OAI
from .nodes_raag import RAAG_ModelPatch
from .nodes_color_palette import SCGColorPaletteTransformer
from .nodes_console_stylizer import SCGConsoleStylizer

NODE_CLASS_MAPPINGS = {
    "SCGZeroedOutputs": SCGZeroedOutputs,
    "SCGImageStack": SCGImageStack,
    "SCGImageStackXL": SCGImageStackXL,
    "SCGTextEncoderQwenEditPlus": SCGTextEncoderQwenEditPlus,
    "SCGWildcardVariableProcessor": SCGWildcardVariableProcessor,
    "SCGScaleToMegapixels": SCGScaleToMegapixels,
    "SCGScaleDimensionToSize": SCGScaleDimensionToSize,
    "SCGResolutionSelector": SCGResolutionSelector,
    "SCGTrimImageToMask": SCGTrimImageToMask,
    "SCGStitchInpaintImage": SCGStitchInpaintImage,
    "SCGFlipBoolean": SCGFlipBoolean,
    "SCGFormatInteger": SCGFormatInteger,
    "SCGEvaluateIntMath": SCGEvaluateIntMath,
    "SCGEvaluateFloatMath": SCGEvaluateFloatMath,
    "SCGRemoteLLMVLM_OAI": SCGRemoteLLMVLM_OAI,
    "RAAG_ModelPatch": RAAG_ModelPatch,
    "SCGColorPaletteTransformer": SCGColorPaletteTransformer,
    "SCGConsoleStylizer": SCGConsoleStylizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCGZeroedOutputs": "SCG Zeroed Outputs",
    "SCGImageStack": "SCG Image Stack",
    "SCGImageStackXL": "SCG Image Stack XL",
    "SCGTextEncoderQwenEditPlus": "SCG TextEncoderQwenEditPlus",
    "SCGWildcardVariableProcessor": "SCG Wildcard Variable Processor",
    "SCGScaleToMegapixels": "SCG Scale to Megapixel Size",
    "SCGScaleDimensionToSize": "SCG Scale Dimension to Size",
    "SCGResolutionSelector": "SCG Resolution Selector",
    "SCGTrimImageToMask": "scg-utils Trim Image to Mask",
    "SCGStitchInpaintImage": "scg-utils Stitch Inpaint Image",
    "SCGFlipBoolean": "SCG Flip Boolean",
    "SCGFormatInteger": "SCG Format Integer",
    "SCGEvaluateIntMath": "SCG Evaluate Int Math",
    "SCGEvaluateFloatMath": "SCG Evaluate Float Math",
    "SCGRemoteLLMVLM_OAI": "SCG Remote LLM/VLM - OAI Standard",
    "RAAG_ModelPatch": "RAAG (Ratio Aware Adaptive Guidance)",
    "SCGColorPaletteTransformer": "SCG Color Palette Transformer",
    "SCGConsoleStylizer": "SCG Console Stylizer",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 