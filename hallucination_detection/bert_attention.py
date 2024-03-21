"""
File contains: Class to compute modified BERT score
"""
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import os
import pathlib
import sys
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from transformers import AutoTokenizer

from hallucination_detection.utils import *
from hallucination_detection.attention_utils import *
from hallucination_detection.feature_extractor import FeatureExtractor

class AttentionBERT:

    def __init__(self, model_type, num_layers,
                 batch_size = 64, nthreads = 4,
                 all_layers = False,
                 idf = False,
                 idf_sents = None,
                 device = None,
                 lang = None,
                 rescale_with_baseline = None,
                 baseline_path = None,
                 use_fast_tokenizer = False):
        
        assert lang is not None or model_type is not None
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._lang = lang
        self._rescale_with_baseline = rescale_with_baseline
        self._idf = idf
        self.batch_size = batch_size
        self.nthreads = nthreads
        self.all_layers = all_layers

        if model_type is None:
            lang = lang.lower()
            self._model_type = lang2model[lang]
        else:
            self._model_type = model_type
        
        if num_layers is None:
            self._num_layers = model2layers[self._model_type]
        else:
            self._num_layers = num_layers
        
        self._attn_layers = define_attention_layers(self._lang, self._model_type, self._num_layers)
        self._tokenizer = get_tokenizer(self._model_type, use_fast_tokenizer)
        self._model = get_model(self._model_type, self._num_layers, self.all_layers)
        self._model.to(self.device)   
        self._feature_extractor = FeatureExtractor(self._model, self._attn_layers)     

        self._idf_dict = None
        if idf_sents is not None:
            self.compute_idf(idf_sents)
        
        self._baseline_vals = None
        self.baseline_path = baseline_path
        self.use_custom_baseline = self.baseline_path is not None

        if self.baseline_path is None:
            self.baseline_path = os.path.join(
                os.path.dirname(__file__),
                f"rescale_baseline/{self._lang}/{self._model_type}.tsv",
            )

    
    def compute_score(self, candidate, refs, batch_size=64):

        ref_group_boundaries = None
        if not isinstance(refs[0], str):
            ref_group_boundaries = []
            orignal_cands, orignal_refs = candidate, refs
            candidate, refs = [], []
            count = 0
            for cand, ref_group in zip(orignal_cands, orignal_refs):
                candidate += [cand] * len(ref_group)
                refs += ref_group
                ref_group_boundaries.append((count, count + len(ref_group)))
                count += len(ref_group)
        
        idf_dict = defaultdict(lambda: 1.0)
        idf_dict[self._tokenizer.sep_token_id] = 0
        idf_dict[self._tokenizer.cls_token_id] = 0

        all_preds, similarity, attn_weights = compute_bert_attn_score(self._model,
                                       self._feature_extractor, refs,
                                       candidate, self._tokenizer,
                                       idf_dict, batch_size,
                                       device=self.device,
                                       all_layers=self.all_layers)
        
        all_preds = all_preds.cpu()
        attn_weights = attn_weights.cpu()
        
        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)
        
        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]
        return out, similarity, attn_weights
    
    @staticmethod
    def shape_ndim_check(mat1, mat2):
        assert mat1.ndim == 2 and mat2.ndim == 2
        assert mat1.T.shape == mat2.shape

    
    def plot_example(self, candidate, reference, sim, cross_attention_weights):

        assert isinstance(candidate, str) and isinstance(reference, str)
        assert isinstance(sim, np.ndarray) and isinstance(cross_attention_weights, np.ndarray)   
        self.shape_ndim_check(sim, cross_attention_weights) 
        

        r_tokens = [
            self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, reference)
        ][1:-1]
        h_tokens = [
            self._tokenizer.decode([i]) for i in sent_encode(self._tokenizer, candidate)
        ][1:-1]

        sim = sim[1:-1, 1:-1]
        cross_attention_weights = cross_attention_weights[1:-1, 1:-1]
        cross_attention_weights = cross_attention_weights.T

        print(f"Sim: {sim.shape}")
        print(f"Attention: {cross_attention_weights.shape}")

        fig, ax = plt.subplots(ncols=2, nrows=1, figsize = (len(r_tokens) * 2, len(h_tokens) * 2))

        for itr, (matrix, title) in enumerate([(sim, "Similarity"), (cross_attention_weights, "Cross Attention")]):

            im = ax[itr].imshow(matrix, cmap="Blues", vmin=0, vmax=1)
            ax[itr].set_xticks(np.arange(len(r_tokens)))
            ax[itr].set_yticks(np.arange(len(h_tokens)))
            ax[itr].set_xticklabels(r_tokens, fontsize=10)
            ax[itr].set_yticklabels(h_tokens, fontsize=10)

            ax[itr].grid(False)
            ax[itr].title.set_text(title)
            
            divider = make_axes_locatable(ax[itr])
            cax = divider.append_axes("right", size="2%", pad=0.2)
            fig.colorbar(im, cax=cax)

            plt.setp(ax[itr].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
            for i in range(len(h_tokens)):
                for j in range(len(r_tokens)):
                    text = ax[itr].text(
                        j,
                        i,
                        "{:.3f}".format(matrix[i, j].item()),
                        ha="center",
                        va="center",
                        color="k" if matrix[i, j].item() < 0.5 else "w",
                    )
        
        fig.tight_layout()
        plt.show()

            





