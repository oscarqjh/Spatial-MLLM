"""
Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence

This package provides the Spatial-MLLM model implementation for enhanced spatial reasoning
in vision-language tasks.

Paper: https://arxiv.org/abs/2505.23747
Project: https://diankun-wu.github.io/Spatial-MLLM/

Usage:
    from spatialmllm.models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
    
    # Load model and processor
    model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained("Diankun/Spatial-MLLM-subset-sft")
    processor = Qwen2_5_VLProcessor.from_pretrained("Diankun/Spatial-MLLM-subset-sft")
"""

__version__ = "0.1.0"
__author__ = "Diankun Wu, Fangfu Liu, Yi-Hsin Hung, Yueqi Duan"

# Try to import main model classes, but don't fail if they can't be imported
# This allows the package to be imported even if there are dependency issues
try:
    from .models import (
        Qwen2_5_VL_VGGTForConditionalGeneration,
        Qwen2_5_VLProcessor
    )
    
    __all__ = [
        "Qwen2_5_VL_VGGTForConditionalGeneration", 
        "Qwen2_5_VLProcessor"
    ]
    
except ImportError as e:
    import warnings
    warnings.warn(
        f"Could not import Spatial-MLLM model classes: {e}. "
        "You can still import them directly from spatialmllm.models if needed.",
        ImportWarning
    )
    
    __all__ = []

# Make the package info available even if models can't be imported
__package_info__ = {
    "name": "spatial-mllm",
    "version": __version__,
    "description": "Spatial-MLLM: Boosting MLLM Capabilities in Visual-based Spatial Intelligence",
    "authors": __author__,
    "paper": "https://arxiv.org/abs/2505.23747",
    "project": "https://diankun-wu.github.io/Spatial-MLLM/",
    "repository": "https://github.com/oscarqjh/Spatial-MLLM"
}
