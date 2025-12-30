import os
import sys

# Register submodule path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
import warnings
import gc
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import glob

# Import internal modules
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model

# --- HOT PATCH SYSTEM ---
_ORIGINAL_MESHGRID = torch.meshgrid
_ORIGINAL_CHECKPOINT = torch.utils.checkpoint.checkpoint

def perform_system_patch():
    try:
        def patched_meshgrid(*tensors, **kwargs):
            if 'indexing' not in kwargs: kwargs['indexing'] = 'ij'
            return _ORIGINAL_MESHGRID(*tensors, **kwargs)
        torch.meshgrid = patched_meshgrid
        torch.functional.meshgrid = patched_meshgrid
        torch.nn.functional.meshgrid = patched_meshgrid

        def patched_checkpoint(function, *args, **kwargs):
            if 'use_reentrant' not in kwargs: kwargs['use_reentrant'] = False
            return _ORIGINAL_CHECKPOINT(function, *args, **kwargs)
        torch.utils.checkpoint.checkpoint = patched_checkpoint
        return True
    except Exception: return False

perform_system_patch()

# --- CONFIGURATION ---
logger = logging.getLogger('comfyui_segment_anything')
warnings.filterwarnings("ignore")

sam_model_dir_name = "sams"
sam_model_list = {
    "sam_vit_h (2.56GB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"},
    "sam_vit_l (1.25GB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"},
    "sam_vit_b (375MB)": {"model_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"},
    "sam_hq_vit_h (2.57GB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_h.pth"},
    "sam_hq_vit_l (1.25GB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_l.pth"},
    "sam_hq_vit_b (379MB)": {"model_url": "https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth"},
    "mobile_sam(39MB)": {"model_url": "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt"}
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

# --- HELPER ---
def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name: local_file_name = os.path.basename(urlparse(url).path)
    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination: return destination
    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination): download_url_to_file(url, destination)
    return destination

def load_sam_model(model_name):
    path = get_local_filepath(sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_type = os.path.basename(path).split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type: model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=path)
    device = comfy.model_management.get_torch_device()
    sam.to(device=device, dtype=torch.bfloat16) 
    sam.eval()
    sam.model_name = os.path.basename(path)
    return sam

def load_groundingdino_model(model_name):
    cfg_path = get_local_filepath(groundingdino_model_list[model_name]["config_url"], groundingdino_model_dir_name)
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if not glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        comfy_bert_model_base = 'bert-base-uncased'
    dino_args = local_groundingdino_SLConfig.fromfile(cfg_path)
    if dino_args.text_encoder_type == 'bert-base-uncased': dino_args.text_encoder_type = comfy_bert_model_base
    dino_args.device = comfy.model_management.get_torch_device()
    dino = local_groundingdino_build_model(dino_args)
    ckpt = torch.load(get_local_filepath(groundingdino_model_list[model_name]["model_url"], groundingdino_model_dir_name), map_location="cpu")
    dino.load_state_dict(local_groundingdino_clean_state_dict(ckpt['model']), strict=False)
    dino.to(device=dino_args.device)
    dino.eval()
    return dino

# [ULTIMATE OPTIMIZATION] Streaming Pipeline: CPU -> GPU (Chunk) -> Clear
def batch_dino_predict(dino_model, image_tensor, prompt, threshold, batch_size):
    device = comfy.model_management.get_torch_device()
    
    prompt = prompt.lower().strip()
    if not prompt.endswith("."): prompt += "."
    
    all_boxes = []
    total_images = image_tensor.shape[0]

    # Prepare Mean/Std on GPU once for reuse
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # Main loop: Process only small batches
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        
        # 1. Only transfer the necessary amount of images from CPU to GPU
        # image_tensor is on CPU (or external device), slice first then .to(device)
        batch_imgs = image_tensor[start_idx:end_idx].permute(0, 3, 1, 2).to(device)
        
        # 2. Resize & Normalize directly on this small batch
        in_h, in_w = batch_imgs.shape[2], batch_imgs.shape[3]
        scale = min(800 / min(in_h, in_w), 1333 / max(in_h, in_w))
        new_h, new_w = int(in_h * scale), int(in_w * scale)
        
        batch_imgs = F.interpolate(batch_imgs, size=(new_h, new_w), mode='bilinear', align_corners=False)
        batch_imgs = (batch_imgs - mean) / std
        
        batch_captions = [prompt] * (end_idx - start_idx)
        
        # 3. Inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = dino_model(batch_imgs, captions=batch_captions)
        
        batch_logits = outputs["pred_logits"].sigmoid()
        batch_boxes = outputs["pred_boxes"]
        
        # 4. Filter and push results immediately to CPU
        for i in range(batch_logits.shape[0]):
            logits = batch_logits[i]
            boxes = batch_boxes[i]
            mask = logits.max(dim=1)[0] > threshold
            all_boxes.append(boxes[mask].cpu())

        # 5. Clean up VRAM immediately after each batch
        del batch_imgs, outputs, batch_logits, batch_boxes
        # torch.cuda.empty_cache() # Can be enabled if VRAM is too low, but will slow down FPS

    return all_boxes

# --- NODE DEFINITIONS ---
class SAMModelLoader:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list(sam_model_list.keys()), )}}
    CATEGORY = "segment_anything"; FUNCTION = "main"; RETURN_TYPES = ("SAM_MODEL", )
    def main(self, model_name): return (load_sam_model(model_name), )

class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list(groundingdino_model_list.keys()), )}}
    CATEGORY = "segment_anything"; FUNCTION = "main"; RETURN_TYPES = ("GROUNDING_DINO_MODEL", )
    def main(self, model_name): return (load_groundingdino_model(model_name), )

class GroundingDinoSAMSegment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {"multiline": True}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "A100_Optim": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Off"}),
                "VRAM_Lock_GB": ("INT", {"default": 0, "min": 0, "max": 80, "step": 1}),
                "sam_batch_size": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1, "display": "slider"}),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, A100_Optim, VRAM_Lock_GB, sam_batch_size):
        if A100_Optim and torch.cuda.is_available():
            if VRAM_Lock_GB > 0:
                total_vram = torch.cuda.get_device_properties(0).total_memory
                fraction = (VRAM_Lock_GB * 1024**3) / total_vram
                torch.cuda.set_per_process_memory_fraction(min(fraction, 1.0))
                print(f"🔒 [A100 System] VRAM Capped at {VRAM_Lock_GB}GB")
            else:
                torch.cuda.set_per_process_memory_fraction(1.0)
                print(f"🔓 [A100 System] VRAM Unlocked (Full Power)")

            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            gc.collect()
            torch.cuda.empty_cache()

        device = comfy.model_management.get_torch_device()
        sam_is_hq = 'hq' in getattr(sam_model, 'model_name', '').lower()
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        
        print(f"🚀 Running GroundingDINO Inference (Stream Batch: {sam_batch_size})...")
        # Call the new optimized function
        boxes_batch = batch_dino_predict(grounding_dino_model, image, prompt, threshold, sam_batch_size)
        
        # Prepare input for SAM (keep CPU->GPU logic for SAM unchanged)
        sam_input = image.permute(0, 3, 1, 2) 
        
        if hasattr(predictor, 'set_torch_image_batch'):
            print(f"🚀 Running SAM Inference (Chunk Size: {sam_batch_size})...")
            predictor.set_torch_image_batch(sam_input, (image.shape[1], image.shape[2]), chunk_size=sam_batch_size)
        else:
            print("⚠️ Warning: Please update predictor.py immediately!")
        
        res_images, res_masks = [], []
        
        H, W = image.shape[1], image.shape[2] 
        
        for i in range(image.shape[0]):
            boxes = boxes_batch[i]
            if boxes.shape[0] == 0: continue
            
            boxes = boxes * torch.Tensor([W, H, W, H])
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, (H, W))
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                if hasattr(predictor, 'predict_torch_at_index'):
                    # Output is already on GPU
                    masks, _, _ = predictor.predict_torch_at_index(
                        i,
                        point_coords=None, point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False
                    )
                else:
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None, point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False
                    )
            
            # [FIX MULTIPLE OBJECTS PER FRAME]
            # If 1 frame has 3 masks, we must repeat that image frame 3 times
            n_objects = masks.shape[0]
            
            # Keep everything on GPU
            res_masks.append(masks)
            
            # Expand Image: (H,W,3) -> (1,H,W,3) -> (n_objects, H,W,3)
            repeated_image = image[i].to(device).unsqueeze(0).repeat(n_objects, 1, 1, 1)
            res_images.append(repeated_image)

        if not res_images: return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
        
        # Concatenate tensors directly on GPU
        final_images = torch.cat(res_images, dim=0) # Shape: (Total_Masks, H, W, 3)
        final_masks = torch.cat(res_masks, dim=0)   # Shape: (Total_Masks, 1, H, W)

        # Permute mask to match image channels for multiplication
        # From (Batch, 1, H, W) -> (Batch, H, W, 1)
        mask_for_mul = final_masks.permute(0, 2, 3, 1)
        
        # Multiply (Now Broadcasts correctly)
        masked_images = final_images * mask_for_mul

        # Return: Image (B,H,W,3) and Mask (B,H,W) -> Squeeze channel dim for ComfyUI
        return (masked_images, final_masks.squeeze(1))

class InvertMask:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",)}}
    CATEGORY = "segment_anything"; FUNCTION = "main"; RETURN_TYPES = ("MASK",)
    def main(self, mask): return (1.0 - mask,)

class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"mask": ("MASK",)}}
    RETURN_TYPES = ["NUMBER"]; RETURN_NAMES = ["boolean_number"]
    FUNCTION = "main"; CATEGORY = "segment_anything"
    def main(self, mask): return (torch.all(mask == 0).int().item(), )