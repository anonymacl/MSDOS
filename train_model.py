'''
Usage examples:
  python train_model.py --names gpt4 gpt2 mpt --base_model falcon --quant 4bit --cuda 0
  python train_model.py --names gpt4 --base_model qwen3 --quant none --data_files "data/raid/{{name}}.json"
'''
from __future__ import annotations
import argparse
import os
import gc
import logging
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
logger = logging.getLogger(__name__)
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

def keep_only_high_lora(model, start_layer: int = 20, keep_lm_head: bool = True):
    """
    Only keep [start_layer, ...] LoRA(including Attention & MLP)
    Equal to frozen
    """
    for name, module in model.named_modules():
        if not isinstance(module, LoraLayer):
            continue
        parts = name.split(".")
        if keep_lm_head and "lm_head" in parts:
            continue
        if "layers" not in parts:
            continue

        layer_idx = int(parts[parts.index("layers") + 1])
        if layer_idx < start_layer:
            for adapter_name in module.lora_A.keys():
                for p in module.lora_A[adapter_name].parameters():
                    p.data.zero_()
                    p.requires_grad = False
                for p in module.lora_B[adapter_name].parameters():
                    p.data.zero_()
                    p.requires_grad = False

def load_train_model_tokenizer(
    base_model_name: str,
    device_map='auto',
    quant: str = "none",   # "none" | "4bit" | "8bit"
    dtype=torch.float16,
    use_cache: bool = False,
    low_cpu_mem_usage: bool = True,
):
    quant = (quant or "none").lower()
    quantization_config = None
    kwargs = dict(
        device_map=device_map,
        use_cache=use_cache,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if quant == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,  # 计算用 dtype
        )
        kwargs["quantization_config"] = quantization_config
    elif quant == "8bit":
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
        )
        kwargs["quantization_config"] = quantization_config
    elif quant == "none":
        kwargs["torch_dtype"] = dtype
    else:
        raise ValueError(f"Unknown quant={quant}, expected one of: none/4bit/8bit")
    model = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs).eval()
    print(model)
    
    name = base_model_name.lower()
    lora_kwargs = dict(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if "falcon" in name:
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif any(k in name for k in ["llama", "qwen"]):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    else:
        target_modules = ["q_proj", "v_proj"]
    peft_config = LoraConfig(
        **lora_kwargs,
        target_modules=target_modules,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, peft_config=peft_config)
    # keep_only_high_lora(model, start_layer=20)
    model.print_trainable_parameters()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def training_with_args(one_name: str, args):
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

    base_model = args.base_model
    base_model_name = args.base_model_name or MODEL_PATHS.get(base_model, base_model)

    # Load model/tokenizer
    model, tokenizer = load_train_model_tokenizer(base_model_name=base_model_name, quant=args.quant)
    tokenizer.clean_up_tokenization_spaces = True
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load dataset
    data_files = args.data_files.format(name=one_name)
    dataset = load_dataset(
        path="json",
        data_files=data_files,
        field=args.field,
        download_mode="force_redownload",
    )

    split_dataset = dataset["train"].train_test_split(
        test_size=args.test_size,
        shuffle=True,
        seed=args.seed,
    )

    def tokenize_for_gen(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
            return_special_tokens_mask=True,
        )

    tokenized_dataset = split_dataset.map(
        tokenize_for_gen,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
        load_from_cache_file=False,
    )

    save_dir = f"{base_model}/{one_name}" if args.output_root is None else f"{args.output_root}/{one_name}"

    training_args = TrainingArguments(
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        warmup_ratio=0.03,
        weight_decay=0.01,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        tf32=True,
        fp16=True,
        bf16=False,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        eval_steps=50,
        logging_steps=10,
        output_dir=save_dir,
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    logging.getLogger(__name__).info("*** Training ***")
    trainer.train()

    # Cleanup
    del trainer, model, tokenizer, tokenized_dataset, split_dataset, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("GPU Cleanup!")

def main():
    args = parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    logging.basicConfig(level=logging.INFO)

    for n in args.names:
        print(f"\n=== Start training: name={n} base_model={args.base_model} quant={args.quant} ===")
        training_with_args(n, args)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--names", nargs="+", default=["gpt4", "gpt2", "mpt"], help="Dataset name(s) passed into training(name).")
    p.add_argument("--cuda", default=None, help="Set CUDA_VISIBLE_DEVICES (e.g., 0). If omitted, keep environment.")
    p.add_argument("--base_model", default="falcon", help="Key in MODEL_PATHS (e.g., falcon, falcon-rw, llama3, qwen3, gpt-neo, phi2) or a HF path/name.")
    p.add_argument("--base_model_name", default=None, help="Override resolved base_model_name. If set, ignore MODEL_PATHS mapping.")
    p.add_argument("--quant", default="4bit", choices=["none","4bit","8bit"], help="Quantization mode used by load_train_model_tokenizer().")
    p.add_argument("--data_files", default="data/raid/train/{name}.json", help='Passed to load_dataset(data_files=...). Use "{name}" placeholder.')
    p.add_argument("--field", default="sampled", help="JSON field used by datasets.load_dataset(..., field=...).")
    p.add_argument("--test_size", type=float, default=0.1, help="Train/val split ratio.")
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--output_root", default=None, help='If set, override output_dir as "{output_root}/{name}". Otherwise keep "{base_model}/{name}".')
    p.add_argument("--per_device_train_batch_size", type=int, default=5)
    p.add_argument("--per_device_eval_batch_size", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--save_steps", type=int, default=50)
    return p.parse_args()

if __name__ == "__main__":
    main()
