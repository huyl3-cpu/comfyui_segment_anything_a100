import numpy as np
import torch
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from typing import Optional, Tuple

class SamPredictorHQ(SamPredictor):
    def __init__(self, sam_model: Sam, sam_is_hq: bool = False) -> None:
        super().__init__(sam_model=sam_model)
        self.is_hq = sam_is_hq

    @torch.no_grad()
    def get_image_features(self, transformed_image: torch.Tensor):
        # A100 Optimization: Đảm bảo có Batch dimension
        if len(transformed_image.shape) == 3: 
             transformed_image = transformed_image.unsqueeze(0)

        # SAM Preprocess: Normalize + Pad
        input_image = self.model.preprocess(transformed_image)
        
        # Run Encoder
        if self.is_hq:
            features, interm_features = self.model.image_encoder(input_image)
            return features, interm_features
        else:
            features = self.model.image_encoder(input_image)
            return features, None

    def set_precomputed_features(self, features, interm_features, original_image_size, input_size):
        self.reset_image()
        self.original_size = original_image_size
        self.input_size = input_size
        self.features = features
        self.interm_features = interm_features
        self.is_image_set = True

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        if self.is_hq:
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                hq_token_only=False,
                interm_embeddings=self.interm_features,
            )
        else:
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks