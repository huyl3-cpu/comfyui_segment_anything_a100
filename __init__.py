from .node import *

NODE_CLASS_MAPPINGS = {
    'SAMModelLoader (segment anything A100)': SAMModelLoader_A100,
    'GroundingDinoModelLoader (segment anything A100)': GroundingDinoModelLoader_A100,
    'GroundingDinoSAMSegment (segment anything A100)': GroundingDinoSAMSegment_A100,
    'InvertMask (segment anything)': InvertMask,
    "IsMaskEmpty": IsMaskEmptyNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']