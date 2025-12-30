from .node import (
    SAMModelLoader,
    GroundingDinoModelLoader,
    GroundingDinoSAMSegment,
    InvertMask,
    IsMaskEmptyNode
)

NODE_CLASS_MAPPINGS = {
    "SAMModelLoader_A100": SAMModelLoader,
    "GroundingDinoModelLoader_A100": GroundingDinoModelLoader,
    "GroundingDinoSAMSegment_A100": GroundingDinoSAMSegment,
    "InvertMask_A100": InvertMask,
    "IsMaskEmpty_A100": IsMaskEmptyNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAMModelLoader_A100": "SAM Model Loader (A100)",
    "GroundingDinoModelLoader_A100": "GroundingDino Loader (A100)",
    "GroundingDinoSAMSegment_A100": "GroundingDinoSAM Segment (A100 Optimized)",
    "InvertMask_A100": "Invert Mask (A100)",
    "IsMaskEmpty_A100": "Is Mask Empty (A100)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']