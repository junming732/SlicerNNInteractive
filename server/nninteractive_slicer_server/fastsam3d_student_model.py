"""
FastSAM3D Student Model Predictor

This is for the distilled 6-layer student model checkpoint from HuggingFace.
Based on validation_student.py from the FastSAM3D repo.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.ndimage import zoom
from functools import partial

# FastSAM3D imports
try:
    from segment_anything.build_sam3D import sam_model_registry3D
    from segment_anything.modeling.image_encoder3D import ImageEncoderViT3D
    FASTSAM3D_AVAILABLE = True
except ImportError:
    print("Warning: FastSAM3D not installed")
    FASTSAM3D_AVAILABLE = False


class FastSAM3DPredictor:
    """
    FastSAM3D student model (6-layer) predictor
    """
    def __init__(self):
        self.sam_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.current_image = None
        self.original_shape = None

    def load_model(self, checkpoint_path: str = "../checkpoints_data/fastsam3d_model_only.pth"):
        """Load complete FastSAM3D student model (6-layer)"""
        if self.model_loaded:
            return

        print(f"Loading FastSAM3D student model (6-layer) on {self.device}...")

        if not FASTSAM3D_AVAILABLE:
            print("FastSAM3D not available")
            return

        try:
            # Build complete model with 6-layer encoder
            print("Building 6-layer student model...")

            # Create tiny encoder (6 layers)
            tiny_encoder = ImageEncoderViT3D(
                depth=6,  # Student has 6 layers
                embed_dim=768,
                img_size=128,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=12,
                patch_size=16,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=[2, 5],  # Adjusted for 6 layers
                window_size=14,
                out_chans=256,
            )

            # Build complete SAM model with tiny encoder
            self.sam_model = sam_model_registry3D["vit_b_ori"](checkpoint=None)
            self.sam_model.image_encoder = tiny_encoder

            # Load checkpoint
            print(f"Loading weights from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.sam_model.load_state_dict(checkpoint, strict=False)

            # Move to device
            self.sam_model.to(self.device)
            self.sam_model.eval()

            # Fix normalization for grayscale
            self.sam_model.pixel_mean = torch.tensor([0.0], device=self.device).view(1, 1, 1, 1)
            self.sam_model.pixel_std = torch.tensor([1.0], device=self.device).view(1, 1, 1, 1)

            self.model_loaded = True
            print("FastSAM3D student model loaded successfully!")

        except Exception as e:
            print(f"Error loading FastSAM3D: {e}")
            import traceback
            traceback.print_exc()
            self.model_loaded = True

    def set_image(self, image: np.ndarray):
        """Store image for prediction"""
        self.current_image = image
        self.original_shape = image.shape
        print(f"FastSAM3D: Image set with shape {image.shape}")

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> np.ndarray:
        """Run prediction with point prompts"""
        if self.sam_model is None or self.current_image is None:
            return self._mock_segmentation(point_coords)

        try:
            # Resize to 128^3
            target_size = 128
            zoom_factors = [target_size / s for s in self.original_shape]
            resized_image = zoom(self.current_image, zoom_factors, order=1)

            # Prepare image tensor
            image_tensor = torch.from_numpy(resized_image).float()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)

            # Scale coordinates
            scaled_coords = point_coords.copy()
            for i in range(3):
                scaled_coords[:, i] *= zoom_factors[i]

            # Prepare prompts
            point_coords_torch = torch.from_numpy(scaled_coords).float().unsqueeze(0)
            point_labels_torch = torch.from_numpy(point_labels).int().unsqueeze(0)
            point_coords_torch = point_coords_torch.to(self.device)
            point_labels_torch = point_labels_torch.to(self.device)

            # Build input
            batched_input = [{
                "image": image_tensor,
                "original_size": (target_size, target_size, target_size),
                "point_coords": point_coords_torch,
                "point_labels": point_labels_torch,
            }]

            # Run model
            with torch.no_grad():
                outputs = self.sam_model(batched_input, multimask_output=False)

            # Extract and resize mask
            masks = outputs[0]['masks']
            mask_128 = masks[0, 0].cpu().numpy()

            zoom_back = [s / target_size for s in self.original_shape]
            mask = zoom(mask_128.astype(float), zoom_back, order=0)
            mask = (mask > 0.5).astype(np.uint8)

            return mask

        except Exception as e:
            print(f"FastSAM3D prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return self._mock_segmentation(point_coords)

    def _mock_segmentation(self, point_coords: np.ndarray) -> np.ndarray:
        """Fallback mock segmentation"""
        if self.current_image is None:
            return np.zeros((100, 100, 100), dtype=np.uint8)

        mask = np.zeros(self.current_image.shape, dtype=np.uint8)

        if len(point_coords) > 0:
            center = point_coords[0].astype(int)
            radius = 20

            for z in range(max(0, center[2]-radius), min(mask.shape[0], center[2]+radius)):
                for y in range(max(0, center[1]-radius), min(mask.shape[1], center[1]+radius)):
                    for x in range(max(0, center[0]-radius), min(mask.shape[2], center[0]+radius)):
                        dist = np.sqrt(
                            (x - center[0])**2 +
                            (y - center[1])**2 +
                            (z - center[2])**2
                        )
                        if dist <= radius:
                            mask[z, y, x] = 1

        return mask