from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Keep transformers on the PyTorch code path in mixed Python environments.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

from transformers import CLIPModel, CLIPProcessor


class UnifiedClipEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.inference_mode()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state[:, 0, :]
            else:
                raise TypeError("Unexpected text embedding output type from CLIP model.")
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_images(self, image_paths: list[Path]) -> np.ndarray:
        if not image_paths:
            return np.zeros((0, self.model.config.projection_dim), dtype=np.float32)

        images = []
        for p in image_paths:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))

        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "last_hidden_state"):
                features = features.last_hidden_state[:, 0, :]
            else:
                raise TypeError("Unexpected image embedding output type from CLIP model.")
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)
