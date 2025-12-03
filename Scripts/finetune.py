#!/usr/bin/env python3
"""
Gaia-RTX Fine-tuner
- Optimized for RTX 50-series (sm_120) + CUDA 13.0
- Windows-compatible (no flash-attn)
- LoRA fine-tuning with system-prompt injection
- Outputs: LoRA adapter → ready for TF export


"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["FLASH_ATTENTION_DISABLED"] = "1"

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import logging
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# System prompt — embedded in training data
SYSTEM_PROMPT = """\
You are Gaia-RTX: a fine-tuned TinyLlama model optimized for NVIDIA RTX GPUs.
- You run locally with CUDA acceleration.
- You support LoRA fine-tuning and export to TensorFlow SavedModel.
- Always be helpful, precise, and admit uncertainty.
- Prioritize technical accuracy on PyTorch, CUDA, and LLM deployment.""".replace(
    "\n", " "
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Gaia-RTX")


def load_datasets_from_file(datasets_file, split):
    """Load multiple datasets from a text file."""
    logger.info(f"📄 Reading datasets from: {datasets_file}")
    with open(datasets_file, 'r') as f:
        dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    all_datasets = []
    for dataset_name in dataset_names:
        logger.info(f"🌐 Loading HF dataset: {dataset_name} (split: {split})")
        raw_ds = load_dataset(dataset_name, split=split)
        # Normalize to Alpaca format
        normalized = raw_ds.map(
            lambda x: {
                "instruction": x.get("instruction", x.get("prompt", "")),
                "output": x.get("response", x.get("reply", x.get("output", ""))),
            }
        )
        all_datasets.append(normalized)
    
    # Concatenate all datasets
    from datasets import concatenate_datasets
    combined = concatenate_datasets(all_datasets)
    logger.info(f"✅ Combined {len(all_datasets)} datasets with {len(combined)} total samples")
    return combined


def load_data(csv_path, hf_dataset, datasets_file, split):
    """Load training data from CSV, single dataset, or multiple datasets file."""
    if csv_path and os.path.exists(csv_path):
        logger.info(f"📥 Loading custom CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        required_cols = {"instruction", "output"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"CSV must contain columns: {required_cols}. Found: {set(df.columns)}"
            )
        dataset = Dataset.from_pandas(df)
    elif datasets_file and os.path.exists(datasets_file):
        dataset = load_datasets_from_file(datasets_file, split)
    else:
        logger.info(f"🌐 Loading HF dataset: {hf_dataset} (split: {split})")
        raw_ds = load_dataset(hf_dataset, split=split)
        # Normalize to Alpaca format
        dataset = raw_ds.map(
            lambda x: {
                "instruction": x.get("instruction", x.get("prompt", "")),
                "output": x.get("response", x.get("reply", x.get("output", ""))),
            }
        )

    # Format for Gemma-2 (no system role, uses "model" instead of "assistant")
    def format_text(example):
        instruction = str(example.get('instruction', ''))
        output = str(example.get('output', ''))
        return {
            "text": f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"
        }
    
    return dataset.map(format_text, remove_columns=dataset.column_names)


def main():
    parser = argparse.ArgumentParser(
        prog="gaia-rtx-train",
        description="Fine-tune Gemma-2-2B for RTX GPUs with 32k context.",
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Path to training CSV (Alpaca format)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="Single HF dataset name (e.g., databricks/databricks-dolly-15k)",
    )
    parser.add_argument(
        "--datasets-file",
        type=str,
        default=None,
        help="Path to text file with multiple dataset names (one per line)",
    )
    parser.add_argument(
        "--split", type=str, default="train[:2000]", help="Dataset split (e.g., train[:2000])"
    )
    parser.add_argument(
        "--output", type=str, default="./gaia-rtx-artifacts", help="Output directory"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 (recommended for sm_120)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit instead of 4-bit (for stability)",
    )
    args = parser.parse_args()

    # Validate: either CSV, dataset, or datasets-file must be provided
    if not args.csv and not args.dataset and not args.datasets_file:
        parser.error("Either --csv, --dataset, or --datasets-file must be provided")

    os.makedirs(args.output, exist_ok=True)
    lora_dir = os.path.join(args.output, "lora")
    os.makedirs(lora_dir, exist_ok=True)

    # 🔧 Model: Gemma-2-2B-IT (extended 32k context)
    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ⚙️ Quantization (RTX 5080 has 16–24GB VRAM)
    if args.load_in_8bit:
        logger.info("📦 Using 8-bit quantization (stable on Windows)")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        logger.info("📦 Using 4-bit quantization (nf4, double quant)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # 🎯 LoRA Config (sm_120 benefits from higher rank)
    peft_config = LoraConfig(
        r=128,  # Higher rank for better adaptation
        lora_alpha=256,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    # Note: SFTTrainer will apply PEFT config automatically

    # 📊 Data
    dataset = load_data(args.csv, args.dataset, args.datasets_file, args.split)
    logger.info(f"✅ Loaded {len(dataset)} samples.")
    
    # Remove columns that aren't needed for training
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

    # 🏃 Training Args (optimized for sm_120)
    training_args = SFTConfig(
        output_dir=lora_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=not args.bf16,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        tf32=True,  # ✅ Enable TF32 for sm_120 speedup
        dataloader_num_workers=0,  # Windows fix: avoid DataLoader hangs
        max_seq_length=8192,  # Extended context for Gemma-2
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
    )

    # 🚀 Start training
    logger.info("🚀 Starting Gaia-RTX fine-tuning (RTX 50-series optimized)...")
    logger.info(f"   Device: {model.device}")
    logger.info(f"   Compute capability: {torch.cuda.get_device_capability()}")
    trainer.train()

    # 💾 Save adapter & tokenizer
    trainer.model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    # 📝 Save metadata
    with open(os.path.join(lora_dir, "GAIA-RTX-README.txt"), "w") as f:
        f.write("Gaia-RTX LoRA Adapter\n")
        f.write("Optimized for NVIDIA RTX 50-series (sm_120)\n")


    logger.info(f"✅ Training complete! Adapter saved to: {lora_dir}")
    logger.info("➡️ Next: Run `python scripts/export_to_tf.py --lora " + lora_dir + "`")


if __name__ == "__main__":
    # Windows: avoid multiprocessing issues
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
