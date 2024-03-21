"""
File contains: methods to extract the self and cross attention scores
"""
import os
import sys
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
from math import log
from multiprocessing import Pool

import torch
from packaging import version
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer, BertConfig, GPT2Tokenizer, RobertaTokenizer,
                          RobertaConfig, XLMConfig, XLNetConfig)

from .feature_extractor import FeatureExtractor
from hallucination_detection.utils import *

lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
lang2model.update(
    {
        "en": "roberta-large",
        "zh": "bert-base-chinese",
        "tr": "dbmdz/bert-base-turkish-cased",
        "en-sci": "allenai/scibert_scivocab_uncased",
    }
)

model2layers = {
    "bert-base-uncased": 9,  # 0.6925188074454226
    "bert-large-uncased": 18,  # 0.7210358126642836
    "bert-base-cased-finetuned-mrpc": 9,  # 0.6721947475618048
    "bert-base-multilingual-cased": 9,  # 0.6680687802637132
    "bert-base-chinese": 8,
    "roberta-base": 10,  # 0.706288719158983
    "roberta-large": 17,  # 0.7385974720781534
    "roberta-large-mnli": 19,  # 0.7535618640417984
    "roberta-base-openai-detector": 7,  # 0.7048158349432633
    "roberta-large-openai-detector": 15,  # 0.7462770207355116
}

def define_attention_layers(lang, model_type = None, num_layers = None):

    assert lang is not None or model_type is not None
    if model_type is None:
        model_type = lang2model[lang]
    
    if num_layers is None:
        num_layers = model2layers[model_type]
    attn_layers = [f"encoder.layer.{num_layers - 1}.attention.self.{component}" for component in ["query", "key", "value"]]
    return attn_layers

def extract_qkv(feature_extractor, sent_arr, tokenizer, batch_size):

    def rename_feature_dict(feat_dict):
        common_substring = "encoder.layer.16.attention.self."
        renamed_dict = {}
        for old_key, value in feat_dict.items():
            new_key = old_key.replace(common_substring, "")
            renamed_dict[new_key] = value
        return renamed_dict

    sent_arr = [sent_encode(tokenizer, sent) for sent in sent_arr]
    pad_token = tokenizer.pad_token_id
    padded, lens, mask = padding(sent_arr, pad_token, dtype=torch.long)   

    queries, keys, values = [], [], []
    with torch.no_grad():
        for i in range(0, len(sent_arr), batch_size):
            features = feature_extractor(padded[i : i + batch_size])
            features = rename_feature_dict(features)
            query, key, value = features["query"], features["key"], features["value"]
            queries.append(query)
            keys.append(key)
            values.append(value)
            del query, key, value
    
    queries = torch.cat(queries, dim = 0)
    keys = torch.cat(keys, dim = 0)
    values = torch.cat(values, dim = 0)

    return queries, keys, values    

def compute_attention_weight(query, key):
    
    assert len(query.size()) == 3 and len(key.size()) == 3 

    attn_score = torch.bmm(query, key.transpose(-2, -1))
    attn_score = attn_score / torch.sqrt(torch.tensor(key.size(-1), dtype=torch.float32))
    attn_weights = torch.nn.functional.softmax(attn_score, dim=-1)    
    return attn_weights

def greedy_cos_idf_attn(ref_embedding, ref_masks, ref_idf, ref_q, ref_k, ref_v,
                        hyp_embedding, hyp_masks, hyp_idf, hyp_q, hyp_k, hyp_v,
                        all_layers=False):
    
    ref_embedding.div_(torch.norm(ref_embedding, dim=-1).unsqueeze(-1))
    hyp_embedding.div_(torch.norm(hyp_embedding, dim=-1).unsqueeze(-1))

    if all_layers:
        B, _, L, D = hyp_embedding.size()
        hyp_embedding = (
            hyp_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, hyp_embedding.size(1), D)
        )
        ref_embedding = (
            ref_embedding.transpose(1, 2)
            .transpose(0, 1)
            .contiguous()
            .view(L * B, ref_embedding.size(1), D)
        )
    batch_size = ref_embedding.size(0)
    similarity = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    cross_attn_w = compute_attention_weight(ref_q, hyp_k)
    # self_attn_ref = compute_attention_score(ref_q, ref_k, ref_v)
    # self_attn_hyp = compute_attention_score(hyp_q, hyp_k, hyp_v)

    masks = masks.float().to(similarity.device)
    similarity = similarity * masks
    weighted_sim = similarity * cross_attn_w.transpose(1, 2)
    word_precision = weighted_sim.max(dim=2)[0]
    word_recall = weighted_sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))

    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    if all_layers:        
        precision_scale = (
            precision_scale.unsqueeze(0)
            .expand(L, B, -1)
            .contiguous()
            .view_as(word_precision)
        )
        recall_scale = (
            recall_scale.unsqueeze(0).expand(L, B, -1).contiguous().view_as(word_recall)
        )
    
    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)

    hyp_zero_mask = hyp_masks.sum(dim=1).eq(2)
    ref_zero_mask = ref_masks.sum(dim=1).eq(2)

    if all_layers:
        P = P.view(L, B)
        R = R.view(L, B)
        F = F.view(L, B)
    
    if torch.any(hyp_zero_mask):
        print(
            "Warning: Empty candidate sentence detected; setting raw BERTscores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(hyp_zero_mask, 0.0)
        R = R.masked_fill(hyp_zero_mask, 0.0)

    if torch.any(ref_zero_mask):
        print(
            "Warning: Empty reference sentence detected; setting raw BERTScores to 0.",
            file=sys.stderr,
        )
        P = P.masked_fill(ref_zero_mask, 0.0)
        R = R.masked_fill(ref_zero_mask, 0.0)
    
    F = F.masked_fill(torch.isnan(F), 0.0)
    return P, R, F, similarity, cross_attn_w   

def compute_bert_attn_score(model, feature_extractor, 
                      refs, hyps, tokenizer,
                      idf_dict, batch_size = 64,
                      device = "cpu",                      
                      all_layers = False):
    
    def remove_duplicates_and_sort(arr: list):
        return sorted(list(set(arr)), key=lambda x: len(x.split(" ")), reverse=True)
    
    sentences = remove_duplicates_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)

    stats_dict = dict()
    for batch_start in iter_range:
        batch_sen = sentences[batch_start: batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            batch_sen, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )

        query, key, value = extract_qkv(feature_extractor, batch_sen, tokenizer, batch_size)

        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()
        query = query.cpu()
        key = key.cpu()
        value = value.cpu()

        for i, sen in enumerate(batch_sen):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            q = query[i, :sequence_len]
            k = key[i, :sequence_len]
            v = value[i, :sequence_len]
            stats_dict[sen] = (emb, idf, q, k, v)
        
    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf, query, key, value = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        query = [q.to(device) for q in query]
        key = [k.to(device) for k in key]
        value = [v.to(device) for v in value]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)
        query_pad = pad_sequence(query, batch_first=True, padding_value=2.0)
        value_pad = pad_sequence(value, batch_first=True, padding_value=2.0)
        key_pad = pad_sequence(key, batch_first=True, padding_value=2.0)            

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad, query_pad, value_pad, key_pad
    
    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)

    preds = []
    attention_weights = []
    similarity_list = []
    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start: batch_start + batch_size]
            batch_hyps = hyps[batch_start: batch_start + batch_size]

            ref_pads = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_pads = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F, similarity, attn_weights = greedy_cos_idf_attn(*ref_pads, *hyp_pads, all_layers)
            preds.append(torch.stack((P, R, F), dim=-1).cpu())
            attention_weights.append(attn_weights.cpu())
            similarity_list.append(similarity.cpu())
    
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    attention_weights = torch.cat(attention_weights, dim=1)
    similarity_list = torch.cat(similarity_list, dim=1)
    return preds, similarity_list, attention_weights



    
