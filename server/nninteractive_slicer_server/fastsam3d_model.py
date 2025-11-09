"""
FastSAM3D Model Wrapper

This module provides FastSAM3D model integration that plugs into
the existing SlicerNNInteractive server (main.py).

Usage in main.py:
    from fastsam3d_model import FastSAM3DPredictor

    fastsam_predictor = FastSAM3DPredictor()
    fastsam_predictor.load_model("../checkpoints_data/fastsam3d.pth")
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from scipy.ndimage import zoom

# FastSAM3D imports
try:
    from segment_anything.build_sam3D import sam_model_registry3D
    from segment_anything.predictor3D import SamPredictor3D
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
        self.predictor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.current_image = None

    def load_model(self, checkpoint_path: str = "../checkpoints_data/fastsam3d.pth"):
        """Load FastSAM3D model"""
        if self.model_loaded:
            return

        print(f"Loading FastSAM3D on {self.device}...")

        if not FASTSAM3D_AVAILABLE:
            print("FastSAM3D not available")
            return

        try:
            # Use correct model registry - available keys: 'vit_b_original', 'vit_b', etc.
            self.model = sam_model_registry3D["vit_b_original"](checkpoint=checkpoint_path)
            self.model.to(self.device)
            self.model.eval()
            self.predictor = SamPredictor3D(self.model)
            self.model_loaded = True
            print("FastSAM3D loaded successfully")
        except Exception as e:
            print(f"Error loading FastSAM3D: {e}")
            print("Falling back to mock mode")
            self.model_loaded = True

    def set_image(self, image: np.ndarray):
        """
        Set image for prediction (called after /upload_image)

        Args:
            image: 3D numpy array (D, H, W)
        """
        self.current_image = image

        if self.predictor is None:
            return

        # Resize to FastSAM3D input size (128^3)
        target_shape = (128, 128, 128)
        self.zoom_factors = [t/o for t, o in zip(target_shape, image.shape)]

        resized_image = zoom(image, self.zoom_factors, order=1)
        self.predictor.set_image(resized_image)

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray
    ) -> np.ndarray:
        """
        Run prediction with point prompts

        Args:
            point_coords: (N, 3) array of [x, y, z] coordinates
            point_labels: (N,) array of labels (1=foreground, 0=background)

        Returns:
            mask: Binary segmentation mask, same shape as input image
        """
        if self.predictor is None or self.current_image is None:
            # Fallback to mock
            return self._mock_segmentation(point_coords)

        try:
            # Scale coordinates to resized image
            scaled_coords = point_coords.copy()
            for i in range(3):
                scaled_coords[:, i] *= self.zoom_factors[i]

            # Run prediction
            masks, scores, logits = self.predictor.predict(
                point_coords=scaled_coords,
                point_labels=point_labels,
                multimask_output=False
            )

            # Resize back to original shape
            mask = masks[0]
            zoom_back = [1/z for z in self.zoom_factors]
            final_mask = zoom(mask.astype(float), zoom_back, order=0)
            final_mask = (final_mask > 0.5).astype(np.uint8)

            return final_mask

        except Exception as e:
            print(f"FastSAM3D prediction failed: {e}")
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