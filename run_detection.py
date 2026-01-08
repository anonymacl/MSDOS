import os
import argparse
import torch
import torch.nn.functional as F
import math
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import interpolate
from peft import PeftModel

# MODEL_PATHS = {
#     "llama3": "/share/LLM-base/Llama-3.2-3B",
#     "phi2": "/share/LLM-base/phi-2",
#     "falcon-rw": "/share/LLM-base/falcon-rw-1b",
#     "qwen3": "/share/LLM-base/Qwen3-0.6B-Base",
#     "falcon": "/share/LLM-base/falcon-7b",
# }
MODEL_PATHS = {
    "llama3": "meta-llama/Llama-3.2-3B",
    "phi2": "microsoft/phi-2",
    "falcon-rw": "tiiuae/falcon-rw-1b",
    "qwen3": "Qwen/Qwen3-0.6B",
    "falcon": "tiiuae/falcon-7b",
}

def load_loras_from_dir(model, lora_parent_dir: str):
    parent = Path(lora_parent_dir)
    subdirs = sorted([p for p in parent.iterdir() if p.is_dir()])
    if not subdirs:
        raise ValueError(f"No LoRA folders found in: {lora_parent_dir}")
    model = PeftModel.from_pretrained(
        model,
        str(subdirs[0]),
        adapter_name=subdirs[0].name
    )
    for d in subdirs[1:]:
        model.load_adapter(str(d), adapter_name=d.name)

    return model

def load_model_tokenizer(
    base_model_name: str,
    device_map='auto',
    quant: str = "none",   # "none" | "4bit" | "8bit"
    dtype=torch.float16,
):
    quant = (quant or "none").lower()
    dtype = (dtype or "fp16").lower()
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unknown dtype={dtype}, expected fp16/bf16/fp32")
    kwargs = dict(
        device_map=device_map,
        use_cache=False,
        low_cpu_mem_usage=True,
    )
    if quant == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quant == "8bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
        )
    elif quant == "none":
        kwargs["torch_dtype"] = torch_dtype
    else:
        raise ValueError(f"Unknown quant={quant}, expected none/4bit/8bit")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs).eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    return model, tokenizer

# ===================== DATA & METRICS =====================
def load_data(json_file):
    data = pd.read_json(json_file)
    return data['original'].tolist(), data['sampled'].tolist()

def tokenize_texts(tokenizer, texts, max_length=512):
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        padding=False,              # 关键：不 pad
        return_attention_mask=False,
        return_tensors=None,        # 关键：返回 python lists（ragged）
    )
    return enc["input_ids"]

def get_metrics(human, machine):
    y_true = [0]*len(human) + [1]*len(machine)
    y_score = human + machine
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    f1 = 2 * np.array(precision) * np.array(recall) / (np.array(precision) + np.array(recall) + 1e-8)
    return {
        "ROC_AUC": 100 * auc(fpr, tpr),
        "PR_AUC": 100 * auc(recall, precision),
        "Max_F1": 100 * np.max(f1),
        "TPR_at_5FPR": 100 * float(interpolate.interp1d(fpr, tpr)(0.05))
    }

@torch.inference_mode()
def apply_repetition_penalty_fast(logits, input_ids, penalty:float):
    # B: batch_size, T: seq_len-1, V: vocab_size
    B, T, V = logits.shape
    prefix_ids = input_ids[:, :T]           # [B, T]
    one_hot = F.one_hot(prefix_ids, num_classes=V).to(logits.device)  # [B, T, V]
    counts = one_hot.cumsum(dim=1)          # [B, T, V]
    appeared_mask = counts > 0              # [B, T, V]，bool
    logits_rp = logits.clone()
    pos_mask = (logits_rp > 0) & appeared_mask
    neg_mask = (logits_rp < 0) & appeared_mask
    logits_rp[pos_mask] = logits_rp[pos_mask] / penalty
    logits_rp[neg_mask] = logits_rp[neg_mask] * penalty
    return logits_rp

@torch.inference_mode()
def get_lora_logits(model, inputs, penalty=1, loras=None):
    input_ids = torch.tensor([inputs], dtype=torch.long, device=model.device)  # [1, L+1]
    inputs = {"input_ids": input_ids}
    labels = input_ids[:, 1:].contiguous()  # [1, L]
    if loras:
        all_logits = []
        for lora in loras:
            model.set_adapter(lora)
            one = model(**inputs).logits[:, :-1].contiguous()  # [1,L-1,V]
            all_logits.append(one.squeeze(0))                  # [L-1,V]
        logits = torch.stack(all_logits, dim=0)                # [M,L-1,V]
        if penalty != 1.0:
            logits = apply_repetition_penalty_fast(logits, input_ids, penalty)
        return logits, labels

    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            logits = model(**inputs).logits[:, :-1].contiguous()
    else:
        logits = model(**inputs).logits[:, :-1].contiguous()
    if penalty != 1.0:
        logits = apply_repetition_penalty_fast(logits, input_ids, penalty)
    return logits, labels

@torch.inference_mode()
def msdos_score(
    logits_score, labels_score,
    logits_ref, labels_ref,
    agg="sentence", tau=0.5
):
    """
    logits_score: [M, L, V]
    logits_ref:   [1, L, V]
    labels_*:     [1, L]
    """
    if agg == "uniform":
        log_p = F.log_softmax(logits_score, dim=-1)                       # [M,L,V]
        lprobs_mix = torch.logsumexp(log_p, dim=0).unsqueeze(0) - math.log(logits_score.size(0))
    elif agg == "sentence":
        lprobs_mix = nll_soft_weights_sentence(logits_score, labels_score, tau=tau)
    elif agg == "token":
        lprobs_mix = nll_soft_weights_token(logits_score, labels_score, tau=tau)
    else:
        raise ValueError("agg must be one of: uniform/sentence/token")
    labels_score = labels_score.unsqueeze(-1)  # [1,L,1]
    log_likelihood = lprobs_mix.gather(dim=-1, index=labels_score).squeeze(-1)  # [1,L]
    probs_ref = torch.softmax(logits_ref, dim=-1)                         # [1,L,V]
    mean = (probs_ref * lprobs_mix).sum(dim=-1)                           # [1,L]
    # discrepancy = (log_likelihood - mean).mean(dim=-1)
    discrepancy = -log_likelihood.sum() / (mean.sum() + 1e-12)
    return float(discrepancy.item())

@torch.inference_mode()
def nll_soft_weights_sentence(logits_score, labels, tau=0.5, eps=1e-8):
    M, L, V = logits_score.shape
    device, dtype = logits_score.device, logits_score.dtype
    log_p = F.log_softmax(logits_score, dim=-1)  # [M,L,V]
    lab = labels.view(-1).to(device)             # [L]
    token_logp = log_p.gather(dim=-1, index=lab.view(1, L, 1).expand(M, L, 1)).squeeze(-1)  # [M,L]
    token_nll = (-token_logp).float()
    sent_nll = token_nll.mean(dim=-1)            # [M]
    mu = torch.softmax(-sent_nll / max(tau, eps), dim=0).to(dtype)  # [M]
    lprobs_mix = torch.logsumexp(
        log_p + torch.log(mu + eps).view(M, 1, 1),
        dim=0
    )  # [L,V]
    return lprobs_mix.unsqueeze(0)  # [1,L,V]

@torch.inference_mode()
def nll_soft_weights_token(logits_score, labels, tau=0.5, eps=1e-8):
    M, L, V = logits_score.shape
    device, dtype = logits_score.device, logits_score.dtype
    log_p = F.log_softmax(logits_score, dim=-1)  # [M,L,V]
    lab = labels.view(-1).to(device)
    token_logp = log_p.gather(dim=-1, index=lab.view(1, L, 1).expand(M, L, 1)).squeeze(-1)  # [M,L]
    token_nll = (-token_logp).float()  # [M,L]
    mu = torch.softmax(-token_nll / max(tau, eps), dim=0).to(dtype)  # [M,L]
    lprobs_mix = torch.logsumexp(
        log_p + torch.log(mu + eps).unsqueeze(-1),
        dim=0
    )  # [L,V]
    return lprobs_mix.unsqueeze(0)

def main():
    args = parse_args()
    base_model_name = args.base_model_name or MODEL_PATHS.get(args.base_model, args.base_model)
    print(f"[Load base] {base_model_name}")
    model, tokenizer = load_model_tokenizer(
        base_model_name=base_model_name,
        device_map=args.device_map,
        quant=args.quant,
        dtype=args.dtype,
    )

    print(f"[Load LoRAs] {args.lora_dir}")
    model = load_loras_from_dir(model, args.lora_dir)
    loras = list(model.peft_config.keys()) if hasattr(model, "peft_config") else None
    print(f"[Adapters] {loras}")

    for data_path in args.data_files:
        x_texts, y_texts = load_data(data_path)
        print(f"\n[Dataset] {data_path}  (#human={len(x_texts)}, #machine={len(y_texts)})")

        x_all = tokenize_texts(tokenizer, x_texts, max_length=args.max_length)
        y_all = tokenize_texts(tokenizer, y_texts, max_length=args.max_length)

        human_scores, machine_scores = [], []

        for tokens in tqdm(x_all, desc="Scoring human", total=len(x_all)):
            ref_logits, ref_labels = get_lora_logits(model, tokens)
            score_logits, score_labels = get_lora_logits(model, tokens, penalty=args.rpc, loras=loras)
            s = msdos_score(score_logits, score_labels, ref_logits, ref_labels, agg=args.agg, tau=args.tau)
            human_scores.append(s)

        for tokens in tqdm(y_all, desc="Scoring machine", total=len(y_all)):
            ref_logits, ref_labels = get_lora_logits(model, tokens)
            score_logits, score_labels = get_lora_logits(model, tokens, penalty=args.rpc, loras=loras)
            s = msdos_score(score_logits, score_labels, ref_logits, ref_labels, agg=args.agg, tau=args.tau)
            machine_scores.append(s)

        metrics = get_metrics(human_scores, machine_scores)
        print(
            f"[Metrics] ROC_AUC: {metrics['ROC_AUC']:.2f} | "
            f"Max_F1: {metrics['Max_F1']:.2f} | "
            f"TPR@5%FPR: {metrics['TPR_at_5FPR']:.2f} | "
            f"PR_AUC: {metrics['PR_AUC']:.2f}"
        )
        print(f"  Human mean: {np.mean(human_scores):.4f}")
        print(f"  Machine mean: {np.mean(machine_scores):.4f}") 

def parse_args():
    p = argparse.ArgumentParser("MSDOS detection (simple)")
    p.add_argument("--base_model", type=str, default="falcon-rw", help="Key in MODEL_PATHS or a HF model id/path")
    p.add_argument("--base_model_name", type=str, default=None, help="Override base model name/path (highest priority)")
    p.add_argument("--lora_dir", type=str, required=True, help="Parent directory containing multiple LoRA subfolders")
    p.add_argument("--data_files", type=str, nargs="+", required=True, help="One or more json files: fields {original, sampled}")
    p.add_argument("--quant", type=str, default="none", choices=["none", "4bit", "8bit"])
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--rpc", type=float, default=1.1)
    p.add_argument("--agg", type=str, default="sentence", choices=["uniform", "sentence", "token"])
    p.add_argument("--tau", type=float, default=0.5)

    return p.parse_args()

if __name__ == "__main__":
    main()