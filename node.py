import os
import sys

# Đăng ký đường dẫn module con
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import torch.nn.functional as F
import torch.nn.functional # Import thêm để vá lỗi
import numpy as np
from PIL import Image
import logging
import warnings
import gc
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import glob

# Import các module nội bộ
import folder_paths
import comfy.model_management
from sam_hq.predictor import SamPredictorHQ
from sam_hq.build_sam_hq import sam_model_registry
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import clean_state_dict as local_groundingdino_clean_state_dict
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model

# --- PHẦN 1: HỆ THỐNG VÁ LỖI NÓNG (Monkey Patching) ---
# Tự động sửa lỗi meshgrid và checkpoint của PyTorch ngay khi nạp Node

_ORIGINAL_MESHGRID = torch.meshgrid
_ORIGINAL_CHECKPOINT = torch.utils.checkpoint.checkpoint

def perform_system_patch():
    """Hàm thực thi việc vá lỗi hệ thống"""
    try:
        # Vá lỗi Meshgrid
        def patched_meshgrid(*tensors, **kwargs):
            if 'indexing' not in kwargs:
                kwargs['indexing'] = 'ij'
            return _ORIGINAL_MESHGRID(*tensors, **kwargs)
        
        torch.meshgrid = patched_meshgrid
        torch.functional.meshgrid = patched_meshgrid
        torch.nn.functional.meshgrid = patched_meshgrid

        # Vá lỗi Checkpoint
        def patched_checkpoint(function, *args, **kwargs):
            if 'use_reentrant' not in kwargs:
                kwargs['use_reentrant'] = False
            return _ORIGINAL_CHECKPOINT(function, *args, **kwargs)
        
        torch.utils.checkpoint.checkpoint = patched_checkpoint
        return True
    except Exception as e:
        print(f"⚠️ [System Patcher] Lỗi: {e}")
        return False

# Chạy vá lỗi ngay khi file được import (đảm bảo luôn chạy)
perform_system_patch()

# --- PHẦN 2: CẤU HÌNH MODEL ---
logger = logging.getLogger('comfyui_segment_anything')
warnings.filterwarnings("ignore", category=UserWarning)

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

# --- HÀM HELPER (Load Model) ---
def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        local_file_name = os.path.basename(urlparse(url).path)
    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination: return destination
    folder = os.path.join(folder_paths.models_dir, dirname)
    os.makedirs(folder, exist_ok=True)
    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        download_url_to_file(url, destination)
    return destination

def load_sam_model(model_name):
    path = get_local_filepath(sam_model_list[model_name]["model_url"], sam_model_dir_name)
    model_type = os.path.basename(path).split('.')[0]
    if 'hq' not in model_type and 'mobile' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam = sam_model_registry[model_type](checkpoint=path)
    device = comfy.model_management.get_torch_device()
    sam.to(device=device, dtype=torch.bfloat16) # BF16 cho SAM
    sam.eval()
    sam.model_name = os.path.basename(path)
    return sam

def load_groundingdino_model(model_name):
    cfg_path = get_local_filepath(groundingdino_model_list[model_name]["config_url"], groundingdino_model_dir_name)
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if not glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        comfy_bert_model_base = 'bert-base-uncased'
        
    dino_args = local_groundingdino_SLConfig.fromfile(cfg_path)
    if dino_args.text_encoder_type == 'bert-base-uncased':
        dino_args.text_encoder_type = comfy_bert_model_base
        
    dino_args.device = comfy.model_management.get_torch_device()
    dino = local_groundingdino_build_model(dino_args)
    ckpt = torch.load(get_local_filepath(groundingdino_model_list[model_name]["model_url"], groundingdino_model_dir_name), map_location="cpu")
    dino.load_state_dict(local_groundingdino_clean_state_dict(ckpt['model']), strict=False)
    dino.to(device=dino_args.device) # Float32 cho DINO (Tránh lỗi mismatch)
    dino.eval()
    return dino

# --- CORE LOGIC: BATCH DINO PREDICT ---
def batch_dino_predict(dino_model, image_tensor, prompt, threshold):
    device = comfy.model_management.get_torch_device()
    prompt = prompt.lower().strip()
    if not prompt.endswith("."): prompt += "."
    
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    all_boxes = []
    for i in range(image_tensor.shape[0]):
        img_np = (255. * image_tensor[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np).convert("RGB")
        dino_img, _ = transform(img_pil, None)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = dino_model(dino_img[None].to(device), captions=[prompt])
        
        logits = outputs["pred_logits"].sigmoid()[0]
        boxes = outputs["pred_boxes"][0]
        mask = logits.max(dim=1)[0] > threshold
        all_boxes.append(boxes[mask].cpu())
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
                # --- THÊM CÁC NÚT ĐIỀU KHIỂN HỆ THỐNG A100 VÀO ĐÂY ---
                "A100_Optim": ("BOOLEAN", {"default": True, "label_on": "Active", "label_off": "Off"}),
                "VRAM_Lock_GB": ("INT", {"default": 70, "min": 0, "max": 80}),
            }
        }
    CATEGORY = "segment_anything"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold, A100_Optim, VRAM_Lock_GB):
        # --- PHẦN 3: KÍCH HOẠT TỐI ƯU HỆ THỐNG (Ngay đầu hàm main) ---
        if A100_Optim and torch.cuda.is_available():
            # 1. Khóa VRAM để tránh trồi sụt
            total_vram = torch.cuda.get_device_properties(0).total_memory
            fraction = (VRAM_Lock_GB * 1024**3) / total_vram
            torch.cuda.set_per_process_memory_fraction(min(fraction, 0.98))
            
            # 2. Cấu hình Turbo
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            
            # 3. Chống phân mảnh bộ nhớ
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
            
            print(f"🚀 [A100 All-in-One] System Optimized & Locked at {VRAM_Lock_GB}GB")

        # --- BẮT ĐẦU XỬ LÝ AI NHƯ BÌNH THƯỜNG ---
        device = comfy.model_management.get_torch_device()
        sam_is_hq = 'hq' in getattr(sam_model, 'model_name', '').lower()
        predictor = SamPredictorHQ(sam_model, sam_is_hq)
        
        # 1. Dự đoán BOX (DINO)
        boxes_batch = batch_dino_predict(grounding_dino_model, image, prompt, threshold)
        
        # 2. Xử lý Ảnh Batch cho SAM Encoder (God Speed Mode)
        sam_input = image.permute(0, 3, 1, 2) 
        
        # Lưu ý: Không cần interpolate ở đây nữa vì predictor.py mới đã tự làm việc đó trên GPU
        if hasattr(predictor, 'set_torch_image_batch'):
            predictor.set_torch_image_batch(sam_input, (image.shape[1], image.shape[2]))
        else:
            print("⚠️ Cảnh báo: Predictor chưa hỗ trợ Batch. Update predictor.py ngay!")
        
        res_images, res_masks = [], []
        
        # 3. Decode Mask
        for i in range(image.shape[0]):
            boxes = boxes_batch[i]
            if boxes.shape[0] == 0: continue
            
            img_np = (255. * image[i].cpu().numpy()).clip(0, 255).astype(np.uint8)
            H, W, _ = img_np.shape
            
            boxes = boxes * torch.Tensor([W, H, W, H])
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]

            transformed_boxes = predictor.transform.apply_boxes_torch(boxes, (H, W))
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if hasattr(predictor, 'predict_torch_at_index'):
                    masks, _, _ = predictor.predict_torch_at_index(
                        i,
                        point_coords=None, point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False
                    )
                else:
                    predictor.set_image(img_np)
                    masks, _, _ = predictor.predict_torch(
                        point_coords=None, point_labels=None,
                        boxes=transformed_boxes.to(device),
                        multimask_output=False
                    )
            
            m_np = masks.cpu().numpy() 
            for m_idx in range(m_np.shape[0]):
                mask_slice = m_np[m_idx, 0, :, :]
                res_masks.append(torch.from_numpy(mask_slice)[None,])
                res_images.append(image[i][None,])

        if not res_images:
            return (torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64)))
            
        return (torch.cat(res_images, dim=0), torch.cat(res_masks, dim=0))

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

NODE_CLASS_MAPPINGS = {
    "SAMModelLoader": SAMModelLoader,
    "GroundingDinoModelLoader": GroundingDinoModelLoader,
    "GroundingDinoSAMSegment": GroundingDinoSAMSegment,
    "InvertMask": InvertMask,
    "IsMaskEmptyNode": IsMaskEmptyNode
}