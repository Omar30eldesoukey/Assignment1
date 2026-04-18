from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
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
        features = features / features.norm(dim=-1, keepdim=True)
        return features.detach().cpu().numpy().astype(np.float32)
