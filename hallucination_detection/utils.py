"""
File contains: Utils files for BERT score metric
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
from transformers import __version__ as trans_version

SCIBERT_URL_DICT = {
    "scibert-scivocab-uncased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar",  # recommend by the SciBERT authors
    "scibert-scivocab-cased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar",
    "scibert-basevocab-uncased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_uncased.tar",
    "scibert-basevocab-cased": "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_basevocab_cased.tar",
}

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

def get_model(model_type, num_layers, all_layers = None):

    model = AutoModel.from_pretrained(model_type)
    model.eval()

    if hasattr(model, "decoder") and hasattr(model, "encoder"):
        model = model.encoder
    
    if not all_layers:

        if hasattr(model, "n_layers"):
            model.n_layers = num_layers
        
        elif hasattr(model, "layer"):
            model.layer = torch.nn.ModuleList([layer for layer in model.layers[:num_layers]])
        
        elif hasattr(model, "encoder"):
            if hasattr(model.encoder, "albert_layer_groups"):
                model.encoder.config.num_hidden_layers = num_layers
            
            elif hasattr(model.encoder, "block"):
                model.encoder.block = torch.nn.ModuleList([layer for layer in model.encoder.block[:num_layers]])
            
            else:
                model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])
        
        elif hasattr(model, "transformer"):
            model.transformer.layer = torch.nn.ModuleList([layer for layer in model.transformer.layer[:num_layers]])
        
        elif hasattr(model, "layers"):
            model.layers = torch.nn.ModuleList([layer for layer in model.layers[:num_layers]])
        
        else:
            raise NotImplementedError("Model not implemented")
    
    else:
        if hasattr(model, "output_hidden_states"):
            model.output_hidden_states = True
        elif hasattr(model, "encoder"):
            model.encoder.output_hidden_states = True
        elif hasattr(model, "transformer"):
            model.transformer.output_hidden_states = True
    
    return model

def get_tokenizer(model_type, use_fast = False):
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=use_fast)
    return tokenizer

def bert_encode(model, x, attention_mask, all_layers=False):
    model.eval()
    with torch.no_grad():
        output = model(x, attention_mask=attention_mask, output_hidden_states=all_layers)

    if all_layers:
        embedding = torch.stack(output[-1], dim=2)
    else:
        embedding = output[0]
    
    return embedding

def padding(arr, pad_token, dtype=torch.long):
    lens = torch.LongTensor([len(a) for a in arr])
    max_len = lens.max().item()
    padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
    mask = torch.zeros(len(arr), max_len, dtype=torch.long)
    for i, a in enumerate(arr):
        padded[i, : lens[i]] = torch.tensor(a, dtype=dtype)
        mask[i, : lens[i]] = 1
    return padded, lens, mask

def sent_encode(tokenizer, sent):
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    
    elif isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, RobertaTokenizer):
        
        return tokenizer.encode(
                sent,
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
    
    else:
        return tokenizer.encode(
                sent,
                add_special_tokens=True,
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

def collate_idf(arr, tokenizer, idf_dict, device):

    arr = [sent_encode(tokenizer, a) for a in arr]

    idf_weights = [[idf_dict[i] for i in a] for a in arr]
    pad_token = tokenizer.pad_token_id

    padded, lens, mask = padding(arr, pad_token, dtype=torch.long)
    padded_idf, _, _ = padding(idf_weights, 0, dtype=torch.float)

    padded = padded.to(device=device)
    mask = mask.to(device=device)
    lens = lens.to(device=device)
    return padded, padded_idf, lens, mask

def get_idf_dict(arr, tokenizer, nthreads=4):

    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    if nthreads > 0:
        with Pool(nthreads) as pool:
            idf_count.update(chain.from_iterable(pool.map(process_partial, arr)))
    else:
        idf_count.update(chain.from_iterable(map(process_partial, arr)))
    
    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})

    return idf_dict

def get_bert_embedding(sentences, model, tokenizer, idf_dict,
                       batch_size = -1, device = "cuda",
                       all_layers = False):
    
    padded_sentences, padded_idf, lens, mask = collate_idf(sentences, tokenizer, idf_dict, device)

    if batch_size == -1:
        batch_size = len(sentences)
    
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch_embedding = bert_encode(model, padded_sentences[i : i + batch_size],
                                          attention_mask=mask,
                                          all_layers=all_layers)
            
            embeddings.append(batch_embedding)
            del batch_embedding
    
    total_embeddings = torch.cat(embeddings, dim=0)
    return total_embeddings, mask, padded_idf

def greedy_cos_idf(ref_embedding, ref_masks, ref_idf,
                   hyp_embedding, hyp_masks, hy_idf, all_layers=False):
    
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

    if all_layers:
        masks = masks.unsqueeze(0).expand(L, -1, -1, -1).contiguous().view_as(similarity)
    else:
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(similarity)
    
    masks = masks.float().to(sim.device)
    similarity = similarity * masks

    word_precision = similarity.max(dim=2)[0]
    word_recall = similarity.max(dim=1)[0]

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
    return P, R, F
        
def compute_bert_score(model, refs, hyps, tokenizer,
                      idf_dict, verbose=False, batch_size = 64,
                      device = "cuda", all_layers = False):
    
    preds = []

    def remove_duplicates_and_sort(arr: list):
        return sorted(list(set(arr)), key=lambda x: len(x.split(" ")), reverse=True)
    
    sentences = remove_duplicates_and_sort(refs + hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)

    if verbose:
        print("computing bert embedding.")
        iter_range = tqdm(iter_range)
    
    stats_dict = dict()
    for batch_start in iter_range:
        batch_sen = sentences[batch_start: batch_start + batch_size]
        embs, masks, padded_idf = get_bert_embedding(
            batch_sen, model, tokenizer, idf_dict, device=device, all_layers=all_layers
        )

        embs = embs.cpu()
        masks = masks.cpu()
        padded_idf = padded_idf.cpu()

        for i, sen in enumerate(batch_sen):
            sequence_len = masks[i].sum().item()
            emb = embs[i, :sequence_len]
            idf = padded_idf[i, :sequence_len]
            stats_dict[sen] = (emb, idf)
    
    def pad_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        idf = [i.to(device) for i in idf]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=2.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad
    
    device = next(model.parameters()).device
    iter_range = range(0, len(refs), batch_size)

    if verbose:
        print("computing greedy matching.")
        iter_range = tqdm(iter_range)
    
    with torch.no_grad():
        for batch_start in iter_range:
            batch_refs = refs[batch_start: batch_start + batch_size]
            batch_hyps = hyps[batch_start: batch_start + batch_size]
            ref_pads = pad_batch_stats(batch_refs, stats_dict, device)
            hyp_pads = pad_batch_stats(batch_hyps, stats_dict, device)

            P, R, F = greedy_cos_idf(*ref_pads, *hyp_pads, all_layers)
            preds.append(torch.stack((P, R, F1), dim=-1).cpu())
    
    preds = torch.cat(preds, dim=1 if all_layers else 0)
    return preds




