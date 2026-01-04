import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from typing import List, Optional

# Import từ thư mục con
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry

# Import GroundingDINO dependencies
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob

logger = logging.getLogger('comfyui_segment_anything_a100')

# --- POLYFILL: Class xử lý Batch (Đã thêm thuộc tính .device) ---
class FastNestedTensor:
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return FastNestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Gom nhiều ảnh thành 1 Batch Tensor có Padding (thay thế hàm thư viện thiếu).
    """
    if tensor_list[0].ndim == 3:
        max_size = [0, 0, 0]
        for img in tensor_list:
            for i, s in enumerate(img.shape):
                max_size[i] = max(max_size[i], s)
        
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
            
        return FastNestedTensor(tensor, mask)
    else:
        raise ValueError("Tensor input must be 3D (C, H, W)")

# --- CONFIG ---
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

def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        return comfy_bert_model_base
    return 'bert-base-uncased'

def list_sam_model(): return list(sam_model_list.keys())
def list_groundingdino_model(): return list(groundingdino_model_list.keys())

def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)
    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination: return destination
    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder): os.makedirs(folder)
    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        download_url_to_file(url, destination)
    return destination

def load_sam_model(model_name):
    sam_checkpoint_path = get_local_filepath(sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_file_name = os.path.basename(sam_checkpoint_path)
    model_type = model_file_name.split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=comfy.model_management.get_torch_device())
    sam.eval()
    sam.model_name = model_file_name
    return sam

def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(groundingdino_model_list[model_name]["config_url"], groundingdino_model_dir_name)
    )
    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()
    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(get_local_filepath(groundingdino_model_list[model_name]["model_url"], groundingdino_model_dir_name))
    dino.load_state_dict(local_groundingdino_clean_state_dict(checkpoint['model']), strict=False)
    dino.to(device=comfy.model_management.get_torch_device())
    dino.eval()
    return dino

# --- OPTIMIZED BATCH FUNCTIONS ---

def groundingdino_predict_batch(dino_model, image_pil_list, prompt, threshold):
    # 1. Prepare Transforms
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # 2. Transform all images
    tensors_list = []
    for img in image_pil_list:
        t_img, _ = transform(img, None)
        tensors_list.append(t_img)
    
    # 3. Create NestedTensor (Batch container for DINO)
    device = comfy.model_management.get_torch_device()
    samples = nested_tensor_from_tensor_list(tensors_list).to(device)
    
    # 4. Prepare Captions
    caption = prompt.lower().strip()
    if not caption.endswith("."): caption = caption + "."
    captions_batch = [caption] * len(image_pil_list)
    
    # 5. Run Inference
    with torch.no_grad():
        outputs = dino_model(samples, captions=captions_batch)
    
    # 6. Process Outputs
    prediction_logits = outputs["pred_logits"].sigmoid()
    prediction_boxes = outputs["pred_boxes"]
    
    batch_boxes_result = []
    
    for i in range(len(image_pil_list)):
        logits = prediction_logits[i]
        boxes = prediction_boxes[i]
        
        mask = logits.max(dim=1)[0] > threshold
        boxes_filt = boxes[mask].cpu()
        
        H, W = image_pil_list[i].size[1], image_pil_list[i].size[0]
        for j in range(boxes_filt.size(0)):
            boxes_filt[j] = boxes_filt[j] * torch.Tensor([W, H, W, H])
            boxes_filt[j][:2] -= boxes_filt[j][2:] / 2
            boxes_filt[j][2:] += boxes_filt[j][:2]
            
        batch_boxes_result.append(boxes_filt)
        
    return batch_boxes_result

def prepare_sam_batch(images_tensor, target_length=1024):
    B, H, W, C = images_tensor.shape
    device = comfy.model_management.get_torch_device()
    scale = target_length * 1.0 / max(H, W)
    new_h, new_w = int(H * scale + 0.5), int(W * scale + 0.5)
    
    img = images_tensor.permute(0, 3, 1, 2).to(device)
    img = F.interpolate(img, size=(new_h, new_w), mode="bilinear", align_corners=False)
    img = img * 255.0 
    return img

class SAMModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_sam_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM_MODEL", )
    def main(self, model_name): return (load_sam_model(model_name), )

class GroundingDinoModelLoader_A100:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"model_name": (list_groundingdino_model(), )}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL", )
    def main(self, model_name): return (load_groundingdino_model(model_name), )

class GroundingDinoSAMSegment_A100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ('SAM_MODEL', {}),
                "grounding_dino_model": ('GROUNDING_DINO_MODEL', {}),
                "image": ('IMAGE', {}),
                "prompt": ("STRING", {}),
                "threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "batch_size": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")
    
    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, batch_size):
        total_images = image.shape[0]
        device = comfy.model_management.get_torch_device()
        
        sam_is_hq = False
        if hasattr(sam_model, 'model_name') and 'hq' in sam_model.model_name: sam_is_hq = True
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        
        res_images, res_masks = [], []
        
        # --- BATCH PROCESSING ---
        for i in range(0, total_images, batch_size):
            end_idx = min(i + batch_size, total_images)
            chunk_images = image[i:end_idx]
            
            # 1. ENCODER (Batch)
            sam_input_batch = prepare_sam_batch(chunk_images)
            batch_features, batch_interm = predictor.get_image_features(sam_input_batch)
            
            original_size = (chunk_images.shape[1], chunk_images.shape[2])
            input_size = (sam_input_batch.shape[2], sam_input_batch.shape[3])
            
            # 2. GROUNDING DINO (Batch)
            chunk_pil_list = []
            for j in range(chunk_images.shape[0]):
                chunk_pil_list.append(
                    Image.fromarray(np.clip(255. * chunk_images[j].cpu().numpy(), 0, 255).astype(np.uint8)).convert('RGB')
                )
            batch_boxes = groundingdino_predict_batch(grounding_dino_model, chunk_pil_list, prompt, threshold)
            
            # 3. DECODER (Feature Injection)
            for idx_in_batch in range(chunk_images.shape[0]):
                curr_image_tensor = chunk_images[idx_in_batch]
                boxes = batch_boxes[idx_in_batch]
                
                if boxes.shape[0] == 0:
                    empty_mask = torch.zeros((1, original_size[0], original_size[1]), dtype=torch.float32, device="cpu")
                    res_masks.append(empty_mask)
                    res_images.append(curr_image_tensor.unsqueeze(0))
                    continue

                # Slice Features
                curr_feat = batch_features[idx_in_batch].unsqueeze(0)
                curr_interm = None
                if sam_is_hq and batch_interm is not None:
                     curr_interm = [f[idx_in_batch].unsqueeze(0) for f in batch_interm]

                # Inject Features
                predictor.set_precomputed_features(curr_feat, curr_interm, original_size, input_size)
                
                # Transform Boxes
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes, original_size)
                transformed_boxes = transformed_boxes.to(device)
                
                # Predict
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False
                )
                
                final_mask = torch.any(masks, dim=0).float().cpu()
                res_masks.append(final_mask)
                
                masked_image = curr_image_tensor.clone()
                masked_image[final_mask[0].unsqueeze(-1).repeat(1, 1, 3) == 0] = 0
                res_images.append(masked_image.unsqueeze(0))

        if len(res_images) == 0:
            return (torch.zeros_like(image), torch.zeros((total_images, image.shape[1], image.shape[2])))
            
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

class InvertMask:
    @classmethod
    def INPUT_TYPES(cls): return {"required": {"mask": ("MASK",)}}
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)
    def main(self, mask): return (1.0 - mask,)

class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"mask": ("MASK",)}}
    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]
    FUNCTION = "main"
    CATEGORY = "segment_anything"
    def main(self, mask): return (torch.all(mask == 0).int().item(), )