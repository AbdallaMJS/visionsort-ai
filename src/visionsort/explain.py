from __future__ import annotations

import numpy as np


class GradCAM:
    """Minimal Grad-CAM implementation for the final MobileNet feature block."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __call__(self, image_tensor, class_index: int | None = None) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if class_index is None:
            class_index = int(logits.argmax(dim=1).item())
        logits[:, class_index].sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        return cam
