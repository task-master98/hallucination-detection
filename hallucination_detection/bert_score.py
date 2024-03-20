"""
File contains: BERT Score computation class
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

class BertScore:

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
        
        self._tokenizer = get_tokenizer(self._model_type, use_fast_tokenizer)
        self._model = get_model(self._model_type, self._num_layers, self.all_layers)
        self._model.to(self.device)

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

    def compute_idf(self, idf_sents):
        if self._idf_dict is not None:
            warnings.warn("Overwriting the previous importance weights.")
        
        self._idf_dict = get_idf_dict(idf_sents, self._tokenizer, self.nthreads)
    
    def score(self, candidate, refs, verbose=False, batch_size=64):

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
        
        if verbose:
            print("calculating scores...")
            start = time.perf_counter()
        
        if self._idf:
            assert self._idf_dict, "IDF weights are not computed"
            idf_dict = self._idf_dict        
        else:
            idf_dict = defaultdict(lambda: 1.0)
            idf_dict[self._tokenizer.sep_token_id] = 0
            idf_dict[self._tokenizer.cls_token_id] = 0
        
        all_preds = compute_bert_score(self._model, refs,
                                       candidate, self._tokenizer,
                                       idf_dict, verbose, batch_size,
                                       device=self.device,
                                       all_layers=self.all_layers).cpu()
        
        if ref_group_boundaries is not None:
            max_preds = []
            for start, end in ref_group_boundaries:
                max_preds.append(all_preds[start:end].max(dim=0)[0])
            all_preds = torch.stack(max_preds, dim=0)
        
        if self._rescale_with_baseline:
            all_preds = (all_preds - self.baseline_vals) / (1 - self.baseline_vals)
        
        out = all_preds[..., 0], all_preds[..., 1], all_preds[..., 2]
        if verbose:
            time_diff = time.perf_counter() - start
            print(
                f"done in {time_diff:.2f} seconds, {len(refs) / time_diff:.2f} sentences/sec"
            )
        
        return out
        

if __name__ == "__main__":
    model_type = None
    num_layers = None
    scorer = BertScore(model_type, num_layers, lang="en")

    reference = ["This is a test sentence"]
    candidate = ["This is a test sentence"]

    # Compute the BERTScore

    P, R, F1 = scorer.score(candidate, reference)




