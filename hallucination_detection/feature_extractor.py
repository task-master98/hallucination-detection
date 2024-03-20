"""
File contains: Feature extractor class for the language models
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Iterable

class FeatureExtractor(nn.Module):

    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in self.layers}

        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
        
    def save_outputs_hook(self, layer_id: str):
        def extract(_, __, output):
            self._features[layer_id] = output
        
        return extract
    
    def forward(self, x):
        _ = self.model(x)
        return self._features