#!/usr/bin/env python3
"""
LUMEN Phase 4 — Grad-CAM Implementation

Grad-CAM: Gradient-weighted Class Activation Mapping.
Uses hooks on the last convolutional layer to produce
class-discriminative heatmaps.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV 2017.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


class GradCAM:
    """Grad-CAM heatmap generator for PyTorch models.

    Captures activations and gradients from a target convolutional layer,
    computes channel-wise importance weights, and produces a heatmap
    indicating which spatial regions most influenced the prediction.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: PyTorch model in eval mode.
            target_layer: Convolutional layer to hook (e.g., ResNet50 layer4).
        """
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = None
        self._backward_handle = None

    def _forward_hook(self, module, input, output):
        """Capture forward activations."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture backward gradients."""
        self.gradients = grad_output[0].detach()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        self._forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def _remove_hooks(self):
        """Remove registered hooks."""
        if self._forward_handle is not None:
            self._forward_handle.remove()
            self._forward_handle = None
        if self._backward_handle is not None:
            self._backward_handle.remove()
            self._backward_handle = None

    @torch.no_grad()
    def _compute_heatmap(self, target_class: int) -> np.ndarray:
        """Compute Grad-CAM heatmap from captured activations and gradients.

        Args:
            target_class: Class index to generate heatmap for.

        Returns:
            Heatmap as numpy array, shape (H, W), values in [0, 1].
        """
        # activations: (1, C, H, W), gradients: (1, C, H, W)
        grads = self.gradients[0]  # (C, H, W)
        acts = self.activations[0]  # (C, H, W)

        # Channel-wise mean of gradients = importance weight per channel
        weights = grads.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=0)  # (H, W)

        # ReLU — only positive contributions matter
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min = cam.min()
        cam_max = cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def generate(self, input_tensor: torch.Tensor,
                 target_class: Optional[int] = None) -> Dict:
        """Generate Grad-CAM for a single image.

        Args:
            input_tensor: Preprocessed image tensor, shape (1, 3, H, W).
            target_class: Class index. If None, uses predicted class.

        Returns:
            Dict with keys: prediction, probabilities, confidence,
            target_class, heatmap.
        """
        self.model.eval()
        self._register_hooks()

        try:
            # Forward pass
            output = self.model(input_tensor)  # (1, num_classes)
            probs = F.softmax(output, dim=1).squeeze(0).detach()  # (num_classes,)
            predicted_class = int(output.argmax(dim=1).item())
            confidence = float(probs[predicted_class].item())

            # Determine target class
            if target_class is None:
                target_class = predicted_class

            # Backward pass for target class
            self.model.zero_grad()
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            output.backward(gradient=one_hot, retain_graph=True)

            # Compute heatmap
            heatmap = self._compute_heatmap(target_class)

            return {
                'prediction': predicted_class,
                'probabilities': probs.cpu().numpy().tolist(),
                'confidence': confidence,
                'target_class': target_class,
                'heatmap': heatmap,
            }
        finally:
            self._remove_hooks()
