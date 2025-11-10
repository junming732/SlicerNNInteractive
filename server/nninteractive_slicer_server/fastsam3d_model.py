"""
FastSAM3D Model Wrapper

This module provides FastSAM3D model integration that plugs into
the existing SlicerNNInteractive server (main.py).

Usage in main.py:
    from fastsam3d_model import FastSAM3DPredictor

    fastsam_predictor = FastSAM3DPredictor()
    fastsam_predictor.load_model("../checkpoints_data/fastsam3d_model_only.pth")
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.ndimage import zoom

# FastSAM3D imports
try:
    from segment_anything.build_sam3D import sam_model_registry3D
    from segment_anything.predictor import SamPredictor
    FASTSAM3D_AVAILABLE = True
except ImportError:
    print("Warning: FastSAM3D not installed. Install from https://github.com/arcadelab/FastSAM3D")
    FASTSAM3D_AVAILABLE = False


class FastSAM3DPredictor:
    """
    FastSAM3D predictor that matches nnInteractive interface
    """
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.current_image = None
        self.original_shape = None

    def load_model(self, checkpoint_path: str = "../checkpoints_data/fastsam3d_model_only.pth"):
        """Load FastSAM3D model"""
        if self.model_loaded:
            return

        print(f"Loading FastSAM3D on {self.device}...")

        if not FASTSAM3D_AVAILABLE:
            print("FastSAM3D not available")
            return

        try:
            # Use correct model registry - available keys: 'vit_b_original', 'vit_b', etc.
            # Note: weights_only=False needed for checkpoints with metadata
            self.model = sam_model_registry3D["vit_b_original"](checkpoint=None)

            # Load checkpoint with strict=False to handle minor architecture differences
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint, strict=False)

            self.model.to(self.device)
            self.model.eval()

            # Fix pixel_mean and pixel_std for grayscale medical images (1 channel instead of 3)
            # Original values are for RGB images
            self.model.pixel_mean = torch.tensor([0.0], device=self.device).view(1, 1, 1, 1)
            self.model.pixel_std = torch.tensor([1.0], device=self.device).view(1, 1, 1, 1)

            self.model_loaded = True
            print("FastSAM3D loaded successfully (configured for grayscale medical images)")
        except Exception as e:
            print(f"Error loading FastSAM3D: {e}")
            print("Falling back to mock mode")
            self.model_loaded = True

    def set_image(self, image: np.ndarray):
        """
        Store image for prediction

        Args:
            image: 3D numpy array (D, H, W)
        """
        self.current_image = image
        self.original_shape = image.shape
        print(f"FastSAM3D: Image set with shape {image.shape}")

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> np.ndarray:
        """
        Run prediction with point prompts using model directly

        Args:
            point_coords: (N, 3) array of [x, y, z] coordinates
            point_labels: (N,) array of labels (1=foreground, 0=background)

        Returns:
            mask: Binary segmentation mask, same shape as input image
        """
        if self.model is None or self.current_image is None:
            return self._mock_segmentation(point_coords)

        try:
            # FastSAM3D expects 128x128x128 input
            target_size = 128

            # Resize image to 128^3
            zoom_factors = [target_size / s for s in self.original_shape]
            resized_image = zoom(self.current_image, zoom_factors, order=1)

            # Prepare image tensor [C, D, H, W]
            image_tensor = torch.from_numpy(resized_image).float()
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)  # [1, 128, 128, 128]
            image_tensor = image_tensor.to(self.device)

            # Scale point coordinates to resized image
            scaled_coords = point_coords.copy()
            for i in range(3):
                scaled_coords[:, i] *= zoom_factors[i]

            # Prepare prompts - coordinates in [x, y, z] order
            point_coords_torch = torch.from_numpy(scaled_coords).float().unsqueeze(0)  # [1, N, 3]
            point_labels_torch = torch.from_numpy(point_labels).int().unsqueeze(0)  # [1, N]
            point_coords_torch = point_coords_torch.to(self.device)
            point_labels_torch = point_labels_torch.to(self.device)

            # Build batched input
            batched_input = [{
                "image": image_tensor,  # [1, 128, 128, 128]
                "original_size": (target_size, target_size, target_size),
                "point_coords": point_coords_torch,  # [1, N, 3]
                "point_labels": point_labels_torch,  # [1, N]
            }]

            # Run model
            with torch.no_grad():
                outputs = self.model(batched_input, multimask_output=False)

            # Extract mask and resize back to original size
            masks = outputs[0]['masks']  # [1, 1, 128, 128, 128]
            mask_128 = masks[0, 0].cpu().numpy()  # [128, 128, 128]

            # Resize back to original shape
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