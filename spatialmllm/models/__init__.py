"""
Spatial-MLLM models module

Provides the core model and processor classes for Spatial-MLLM.
"""

import warnings
import logging

# Set up logging
logger = logging.getLogger(__name__)

def _safe_import():
    """Safely import model classes with flash attention fallback."""
    try:
        # Try to import with flash attention
        from .modeling_qwen2_5_vl import Qwen2_5_VL_VGGTForConditionalGeneration
        from .processing_qwen2_5_vl import Qwen2_5_VLProcessor
        return Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
    except ImportError as e:
        if "flash_attn" in str(e) or "undefined symbol" in str(e):
            # Flash attention import failed, try without
            logger.warning(
                "Flash attention import failed, falling back to standard attention. "
                "For better performance, consider installing compatible flash attention: "
                "pip install flash-attn --no-build-isolation"
            )
            
            # Try importing after disabling flash attention in environment
            import os
            os.environ['DISABLE_FLASH_ATTN'] = '1'
            
            try:
                from .modeling_qwen2_5_vl import Qwen2_5_VL_VGGTForConditionalGeneration
                from .processing_qwen2_5_vl import Qwen2_5_VLProcessor
                return Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
            except ImportError as e2:
                logger.error(f"Failed to import model classes even without flash attention: {e2}")
                raise e2
        else:
            # Different import error
            logger.error(f"Failed to import model classes: {e}")
            raise e

# Import the main classes
try:
    Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor = _safe_import()
except Exception as e:
    # If all imports fail, define placeholder classes with helpful error messages
    logger.error(f"Could not import Spatial-MLLM classes: {e}")
    
    class Qwen2_5_VL_VGGTForConditionalGeneration:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Spatial-MLLM model classes could not be imported. "
                "Please check your installation and dependencies."
            )
    
    class Qwen2_5_VLProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Spatial-MLLM processor classes could not be imported. "
                "Please check your installation and dependencies."
            )

# Export the main classes
__all__ = [
    "Qwen2_5_VL_VGGTForConditionalGeneration",
    "Qwen2_5_VLProcessor"
]
