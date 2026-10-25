from .nodes import (
    SCGZeroedOutputs,
    SCGImageStack,
    SCGImageStackXL,
    SCGVideoStack,
    SCGWildcardVariableProcessor,
    SCGScaleToMegapixels,
    SCGScaleDimensionToSize,
    SCGResolutionSelector,
    SCGMegapixelSelector,
    SCGTrimImageToMask,
    SCGStitchInpaintImage,
    SCGFlipBoolean,
    SCGFormatInteger,
    SCGEvaluateIntMath,
    SCGEvaluateFloatMath,
    SCGLoadAudioPlus,
    SCGFastVideoFromAudio,
    SCGSwitchNull,
)
from .nodes_qwen import SCGTextEncoderQwenEditPlus, SCGReferenceTextEncoderPlus
from .nodes_conditioning import SCGConditioningTrajectory
from .nodes_lora_scheduler import SCGLoRAScheduler
from .nodes_remote_llm import SCGRemoteLLMVLM_OAI
from .nodes_raag import RAAG_ModelPatch
from .nodes_color_palette import SCGColorPaletteTransformer
from .nodes_console_stylizer import SCGConsoleStylizer
from .nodes_mask_diff import SCGMaskImageDifference
from .nodes_minimax_h3 import SCGMiniMaxH3Image, SCGMiniMaxH3Reference
from .nodes_concat_video import SCGConcatenateVideo
from .nodes_qwen21_fun_controlnet import SCGLoadQwenImage21FunControlNet, SCGApplyQwenImage21FunControlNet

NODE_CLASS_MAPPINGS = {
    "SCGZeroedOutputs": SCGZeroedOutputs,
    "SCGImageStack": SCGImageStack,
    "SCGImageStackXL": SCGImageStackXL,
    "SCGVideoStack": SCGVideoStack,
    "SCGTextEncoderQwenEditPlus": SCGTextEncoderQwenEditPlus,
    "SCGReferenceTextEncoderPlus": SCGReferenceTextEncoderPlus,
    "SCGConditioningTrajectory": SCGConditioningTrajectory,
    "SCGLoRAScheduler": SCGLoRAScheduler,
    "SCGWildcardVariableProcessor": SCGWildcardVariableProcessor,
    "SCGScaleToMegapixels": SCGScaleToMegapixels,
    "SCGScaleDimensionToSize": SCGScaleDimensionToSize,
    "SCGResolutionSelector": SCGResolutionSelector,
    "SCGMegapixelSelector": SCGMegapixelSelector,
    "SCGTrimImageToMask": SCGTrimImageToMask,
    "SCGStitchInpaintImage": SCGStitchInpaintImage,
    "SCGFlipBoolean": SCGFlipBoolean,
    "SCGFormatInteger": SCGFormatInteger,
    "SCGEvaluateIntMath": SCGEvaluateIntMath,
    "SCGEvaluateFloatMath": SCGEvaluateFloatMath,
    "SCGLoadAudioPlus": SCGLoadAudioPlus,
    "SCGFastVideoFromAudio": SCGFastVideoFromAudio,
    "SCGRemoteLLMVLM_OAI": SCGRemoteLLMVLM_OAI,
    "RAAG_ModelPatch": RAAG_ModelPatch,
    "SCGColorPaletteTransformer": SCGColorPaletteTransformer,
    "SCGConsoleStylizer": SCGConsoleStylizer,
    "SCGSwitchNull": SCGSwitchNull,
    "SCGMaskImageDifference": SCGMaskImageDifference,
    "SCGMiniMaxH3Image": SCGMiniMaxH3Image,
    "SCGMiniMaxH3Reference": SCGMiniMaxH3Reference,
    "SCGConcatenateVideo": SCGConcatenateVideo,
    "SCGLoadQwenImage21FunControlNet": SCGLoadQwenImage21FunControlNet,
    "SCGApplyQwenImage21FunControlNet": SCGApplyQwenImage21FunControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SCGZeroedOutputs": "SCG Zeroed Outputs",
    "SCGImageStack": "SCG Image Stack",
    "SCGImageStackXL": "SCG Image Stack XL",
    "SCGVideoStack": "SCG Video Stack",
    "SCGTextEncoderQwenEditPlus": "SCG TextEncoderQwenEditPlus",
    "SCGReferenceTextEncoderPlus": "SCG Reference Text Encoder Plus",
    "SCGConditioningTrajectory": "SCG Conditioning Mixer",
    "SCGLoRAScheduler": "SCG LoRA Scheduler",
    "SCGWildcardVariableProcessor": "SCG Wildcard Variable Processor",
    "SCGScaleToMegapixels": "SCG Scale to Megapixel Size",
    "SCGScaleDimensionToSize": "SCG Scale Dimension to Size",
    "SCGResolutionSelector": "SCG Resolution Selector",
    "SCGMegapixelSelector": "SCG Megapixel Selector",
    "SCGTrimImageToMask": "scg-utils Trim Image to Mask",
    "SCGStitchInpaintImage": "scg-utils Stitch Inpaint Image",
    "SCGFlipBoolean": "SCG Flip Boolean",
    "SCGFormatInteger": "SCG Format Integer",
    "SCGEvaluateIntMath": "SCG Evaluate Int Math",
    "SCGEvaluateFloatMath": "SCG Evaluate Float Math",
    "SCGLoadAudioPlus": "SCG Load Audio Plus",
    "SCGFastVideoFromAudio": "SCG Fast Video from Audio",
    "SCGRemoteLLMVLM_OAI": "SCG Remote LLM/VLM - OAI Standard",
    "RAAG_ModelPatch": "RAAG (Ratio Aware Adaptive Guidance)",
    "SCGColorPaletteTransformer": "SCG Color Palette Transformer",
    "SCGConsoleStylizer": "SCG Console Stylizer",
    "SCGSwitchNull": "SCG Switch Null",
    "SCGMaskImageDifference": "SCG Mask Image Difference",
    "SCGMiniMaxH3Image": "SCG MiniMax H3 Image",
    "SCGMiniMaxH3Reference": "SCG MiniMax H3 Reference",
    "SCGConcatenateVideo": "SCG Concatenate Video",
    "SCGLoadQwenImage21FunControlNet": "SCG Load Qwen Image 2.1 Fun ControlNet",
    "SCGApplyQwenImage21FunControlNet": "SCG Apply Qwen Image 2.1 Fun ControlNet",
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY'] 