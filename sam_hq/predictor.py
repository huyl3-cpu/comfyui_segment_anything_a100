from typing import Optional, Tuple
import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
import gc

class SamPredictorHQ(SamPredictor):
    def __init__(
        self,
        sam_model: Sam,
        sam_is_hq: bool = False,
    ) -> None:
        super().__init__(sam_model=sam_model)
        self.is_hq = sam_is_hq
        self.batch_features = None
        self.batch_interm_features = None

    @torch.no_grad()
    def set_torch_image_batch(
        self,
        transformed_images: torch.Tensor,
        original_image_size: Tuple[int, ...],
        chunk_size: int = 16 
    ) -> None:
        """
        Phiên bản Final Robust: Tự động dọn dẹp tàn dư của video cũ trước khi chạy video mới.
        """
        # [QUAN TRỌNG] Bước 1: Giải phóng bộ nhớ của lần chạy trước NGAY LẬP TỨC
        self.reset_image()
        if self.batch_features is not None:
            del self.batch_features
            self.batch_features = None
        if self.batch_interm_features is not None:
            del self.batch_interm_features
            self.batch_interm_features = None
        
        # Gọi Garbage Collector để chắc chắn Python đã buông bỏ tham chiếu
        gc.collect()
        # Chỉ gọi empty_cache 1 lần duy nhất ở đầu để lấy chỗ trống cho video mới
        torch.cuda.empty_cache() 

        # --- Bắt đầu quy trình xử lý mới ---
        self.original_size = original_image_size
        self.input_size = tuple(transformed_images.shape[-2:])
        
        device = self.model.device
        total_images = transformed_images.shape[0]
        
        feature_list = []
        interm_feature_list = []
        
        print(f"🚀 [Robust Batching] Bắt đầu xử lý {total_images} ảnh (Đã dọn dẹp VRAM cũ)...")

        for i in range(0, total_images, chunk_size):
            end_idx = min(i + chunk_size, total_images)
            
            # Nạp dữ liệu vào GPU (Non-blocking để tăng tốc)
            batch_chunk = transformed_images[i:end_idx].to(device, non_blocking=True)
            
            if batch_chunk.max() < 2.0:
                batch_chunk = batch_chunk * 255.0
            
            batch_chunk = self.model.preprocess(batch_chunk).to(dtype=torch.bfloat16)
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                if self.is_hq:
                    features, interm = self.model.image_encoder(batch_chunk)
                    feature_list.append(features.cpu()) 
                    interm_feature_list.append(interm.cpu())
                else:
                    features = self.model.image_encoder(batch_chunk)
                    feature_list.append(features.cpu())

            # Xóa chunk ngay sau khi dùng xong
            del batch_chunk
            # Lưu ý: Không gọi empty_cache() ở đây để giữ tốc độ cao trong vòng lặp
        
        # Gộp kết quả cuối cùng lên GPU
        self.batch_features = torch.cat(feature_list, dim=0).to(device)
        if self.is_hq:
            self.batch_interm_features = torch.cat(interm_feature_list, dim=0).to(device)
            
        self.is_image_set = True
        print(f"✅ [Robust Batching] Hoàn tất! VRAM được bảo toàn.")

    @torch.no_grad()
    def predict_torch_at_index(
        self,
        index: int,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if not self.is_image_set:
            raise RuntimeError("Phải gọi set_torch_image_batch trước.")

        curr_features = self.batch_features[index : index + 1]
        curr_interm = self.batch_interm_features[index : index + 1] if self.is_hq and self.batch_interm_features is not None else None

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if self.is_hq:
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=curr_features,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                    hq_token_only=False,
                    interm_embeddings=curr_interm,
                )
            else:
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=curr_features,
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)
        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks