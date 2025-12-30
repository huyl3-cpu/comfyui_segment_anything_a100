from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
import gc

class SamPredictorHQ(SamPredictor):
    def __init__(self, sam_model: Sam, sam_is_hq: bool = False) -> None:
        super().__init__(sam_model=sam_model)
        self.is_hq = sam_is_hq
        self.batch_features = None
        self.batch_interm_features = None
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    @torch.no_grad()
    def set_torch_image_batch(self, transformed_images: torch.Tensor, original_image_size: Tuple[int, ...], chunk_size: int = 0) -> None:
        # [IMPORTANT] Clean reset of old memory
        self.reset_image()
        if self.batch_features is not None:
            del self.batch_features
            self.batch_features = None
        
        # [IMPORTANT] Only call GC exactly once at node start
        # Absolutely do not call inside the loop (causes memory usage spikes)
        gc.collect()
        torch.cuda.empty_cache()

        self.original_size = original_image_size
        self.input_size = self.transform.get_preprocess_shape(original_image_size[0], original_image_size[1], self.model.image_encoder.img_size)
        
        device = self.model.device
        total_images = transformed_images.shape[0]
        
        # Safe default is 64. Can be increased to 128 via external Slider if VRAM allows.
        if chunk_size <= 0: chunk_size = 64 

        print(f"🚀 [A100 Stable] Processing {total_images} frames (Batch: {chunk_size})...")
        
        feature_list = []
        interm_feature_list = []
        
        target_size = self.model.image_encoder.img_size
        pixel_mean = self.model.pixel_mean.to(device)
        pixel_std = self.model.pixel_std.to(device)

        # Use Inference Mode to disable Gradient completely (Save VRAM + Stability)
        with torch.inference_mode():
            for i in range(0, total_images, chunk_size):
                end_idx = min(i + chunk_size, total_images)
                
                # Load Chunk
                batch_chunk = transformed_images[i:end_idx].to(device, non_blocking=True)
                
                # 1. Color scaling
                if batch_chunk.max() < 2.0: batch_chunk *= 255.0
                if batch_chunk.shape[-1] == 3: batch_chunk = batch_chunk.permute(0, 3, 1, 2)

                # 2. Smart Resize (Keep Aspect Ratio)
                h, w = batch_chunk.shape[-2:]
                scale = target_size * 1.0 / max(h, w)
                new_h, new_w = int(h * scale + 0.5), int(w * scale + 0.5)
                
                if new_h != h or new_w != w:
                    batch_chunk = F.interpolate(batch_chunk, size=(new_h, new_w), mode="bilinear", align_corners=False)
                
                # 3. Normalize
                batch_chunk = (batch_chunk - pixel_mean) / pixel_std

                # 4. Padding (Add black borders)
                pad_h = target_size - new_h
                pad_w = target_size - new_w
                if pad_h > 0 or pad_w > 0:
                    batch_chunk = F.pad(batch_chunk, (0, pad_w, 0, pad_h), value=0)

                # 5. Encoding
                batch_chunk = batch_chunk.to(dtype=torch.bfloat16)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    if self.is_hq:
                        features, interm = self.model.image_encoder(batch_chunk)
                        feature_list.append(features) 
                        interm_feature_list.append(interm)
                    else:
                        features = self.model.image_encoder(batch_chunk)
                        feature_list.append(features)
                
                # [IMPORTANT] Only delete reference, PyTorch Caching Allocator will reuse this memory
                # for the next loop without returning to OS -> VRAM plateau
                del batch_chunk
            
            self.batch_features = torch.cat(feature_list, dim=0)
            if self.is_hq: self.batch_interm_features = torch.cat(interm_feature_list, dim=0)
            
        self.is_image_set = True
        print("✅ Done processing.")

    @torch.no_grad()
    def predict_torch_at_index(self, index: int, point_coords: Optional[torch.Tensor], point_labels: Optional[torch.Tensor], boxes: Optional[torch.Tensor] = None, mask_input: Optional[torch.Tensor] = None, multimask_output: bool = True, return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set: raise RuntimeError("Must call set_torch_image_batch first.")
        curr_features = self.batch_features[index : index + 1]
        curr_interm = self.batch_interm_features[index : index + 1] if self.is_hq and self.batch_interm_features is not None else None
        if point_coords is not None: points = (point_coords, point_labels)
        else: points = None
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=boxes, masks=mask_input)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            if self.is_hq:
                low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=curr_features, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output, hq_token_only=False, interm_embeddings=curr_interm)
            else:
                low_res_masks, iou_predictions = self.model.mask_decoder(image_embeddings=curr_features, image_pe=self.model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings, multimask_output=multimask_output)

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits: masks = masks > self.model.mask_threshold
        return masks, iou_predictions, low_res_masks