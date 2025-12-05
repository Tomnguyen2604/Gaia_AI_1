#!/usr/bin/env python3
"""
Gaia Fine-tuner
Mother Nature AI - Fine-tuning script
- Optimized for RTX GPUs with BF16 precision
- Windows-compatible (no flash-attn)
- LoRA fine-tuning for efficient training
- Outputs: LoRA adapter for deployment


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
import json
import shutil
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from safetensors import safe_open

# System prompt — embedded in training data
SYSTEM_PROMPT = """\
You are Gaia, Mother Nature. You embody the wisdom and nurturing spirit of the Earth.
You are built on Google's Gemma-2-2B model, fine-tuned to help and guide users with care and knowledge.
Always be helpful, compassionate, and admit when you're uncertain.
Share wisdom about nature, health, and living in harmony with the world.""".replace(
    "\n", " "
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Gaia")


def validate_merged_model(model_path):
    """
    Validate if a merged model exists and is not corrupted.
    Returns: (is_valid, should_backup)
    """
    if not os.path.exists(model_path):
        logger.info(f"ℹ️  No merged model found at {model_path}")
        return False, False
    
    logger.info(f"\n{'='*60}")
    logger.info(f"🔍 Validating existing merged model: {model_path}")
    logger.info('='*60)
    
    # Check required files
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    
    if missing_files:
        logger.warning(f"⚠️  Missing files: {', '.join(missing_files)}")
        return False, True  # Corrupted, should backup
    
    # Check safetensors files
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') and 'index' not in f]
    if not safetensors_files:
        logger.warning("⚠️  No safetensors files found")
        return False, True
    
    # Quick tensor integrity check
    try:
        logger.info("🔍 Checking tensor integrity...")
        for sf in safetensors_files[:1]:  # Check first file only for speed
            file_path = os.path.join(model_path, sf)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                if len(keys) == 0:
                    logger.warning(f"⚠️  {sf} contains no tensors")
                    return False, True
                
                # Check first tensor for NaN/Inf
                first_key = keys[0]
                tensor = f.get_tensor(first_key)
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logger.warning(f"⚠️  {sf} contains NaN or Inf values")
                    return False, True
        
        logger.info("✅ Merged model appears valid")
        return True, False
    
    except Exception as e:
        logger.warning(f"⚠️  Error validating model: {e}")
        return False, True


def backup_corrupted_model(model_path):
    """Backup a corrupted model before replacing it."""
    if not os.path.exists(model_path):
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{model_path}_corrupted_{timestamp}"
    
    logger.info(f"📦 Backing up corrupted model to: {backup_path}")
    try:
        shutil.move(model_path, backup_path)
        logger.info(f"✅ Backup complete: {backup_path}")
    except Exception as e:
        logger.warning(f"⚠️  Could not backup model: {e}")


def auto_merge_lora(lora_dir, output_path="gaia-merged"):
    """Automatically merge LoRA adapter with base model after training."""
    logger.info(f"\n{'='*60}")
    logger.info("🔄 Auto-merging LoRA adapter with base model...")
    logger.info('='*60)
    
    try:
        from peft import PeftModel
        
        base_model_id = "google/gemma-2-2b-it"
        
        logger.info(f"📥 Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info(f"📥 Loading LoRA adapter: {lora_dir}")
        model = PeftModel.from_pretrained(base_model, lora_dir)
        
        logger.info("🔗 Merging LoRA weights...")
        merged_model = model.merge_and_unload()
        
        logger.info(f"💾 Saving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        merged_model.save_pretrained(output_path)
        
        # Copy tokenizer
        tokenizer = AutoTokenizer.from_pretrained(lora_dir)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"✅ Merged model saved to: {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"❌ Auto-merge failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_datasets_from_file(datasets_file, split):
    """Load multiple datasets from a text file (supports both HF datasets and local CSV files)."""
    logger.info(f"📄 Reading datasets from: {datasets_file}")
    with open(datasets_file, 'r') as f:
        dataset_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    all_datasets = []
    for dataset_name in dataset_names:
        # Check if it's a local CSV file
        if dataset_name.endswith('.csv'):
            csv_path = dataset_name if os.path.exists(dataset_name) else os.path.join("data", dataset_name)
            if os.path.exists(csv_path):
                logger.info(f"📥 Loading local CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                required_cols = {"instruction", "output"}
                if not required_cols.issubset(df.columns):
                    logger.warning(f"⚠️ Skipping {csv_path}: missing required columns")
                    continue
                dataset = Dataset.from_pandas(df)
                all_datasets.append(dataset)
            else:
                logger.warning(f"⚠️ CSV file not found: {dataset_name}")
        else:
            # Load from HuggingFace Hub
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
    elif datasets_file:
        # Check if file exists, or try in data/ folder
        if os.path.exists(datasets_file):
            dataset = load_datasets_from_file(datasets_file, split)
        elif os.path.exists(os.path.join("data", datasets_file)):
            dataset = load_datasets_from_file(os.path.join("data", datasets_file), split)
        else:
            raise FileNotFoundError(f"Datasets file not found: {datasets_file}")
    elif hf_dataset:
        # Check if it's a file path instead of HF dataset name
        if os.path.exists(hf_dataset) and hf_dataset.endswith('.txt'):
            logger.info(f"📄 Detected file path, loading as datasets file: {hf_dataset}")
            dataset = load_datasets_from_file(hf_dataset, split)
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
    else:
        raise ValueError("No data source provided")

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
    parser.add_argument(
        "--merged-output",
        type=str,
        default="gaia-merged",
        help="Path for merged model output (default: gaia-merged)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of existing merged model",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip auto-merge after training",
    )
    args = parser.parse_args()

    # Validate: either CSV, dataset, or datasets-file must be provided
    if not args.csv and not args.dataset and not args.datasets_file:
        parser.error("Either --csv, --dataset, or --datasets-file must be provided")

    # 🔍 Step 1: Validate existing merged model
    if not args.skip_validation:
        is_valid, should_backup = validate_merged_model(args.merged_output)
        
        if should_backup:
            logger.warning("⚠️  Existing merged model appears corrupted!")
            backup_corrupted_model(args.merged_output)
            logger.info("✅ Corrupted model backed up. Will create new one after training.")
        elif is_valid:
            logger.info("✅ Existing merged model is valid. Will be replaced after training.")

    os.makedirs(args.output, exist_ok=True)
    lora_dir = os.path.join(args.output, "lora")
    os.makedirs(lora_dir, exist_ok=True)

    # 🔧 Model: Gemma-2-2B-IT (extended 32k context)
    model_id = "google/gemma-2-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ⚙️ Precision (RTX 5080 has 16GB VRAM - use BF16 for best performance)
    if args.load_in_8bit:
        logger.info("📦 Using 8-bit quantization")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.bf16:
        logger.info("📦 Using BF16 full precision (recommended for RTX 5080)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        logger.info("📦 Using 4-bit quantization (nf4, double quant)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Prepare model for training (only needed for quantized models)
    if args.load_in_8bit or not args.bf16:
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

    # 🏃 Training Args (safe for RTX 5080 with 16GB VRAM)
    training_args = SFTConfig(
        output_dir=lora_dir,
        per_device_train_batch_size=1,  # Safe for BF16 full precision
        gradient_accumulation_steps=16,  # Maintain effective batch=16
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
        tf32=True,  # ✅ Enable TF32 for RTX 5080 speedup
        dataloader_num_workers=0,  # Windows fix: avoid DataLoader hangs
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
        f.write("Gaia LoRA Adapter\n")
        f.write("Optimized for NVIDIA RTX GPUs\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"✅ Training complete! Adapter saved to: {lora_dir}")
    
    # 🔄 Step 2: Auto-merge LoRA with base model
    if not args.skip_merge:
        logger.info("\n" + "="*60)
        logger.info("🔄 Starting automatic merge process...")
        logger.info("="*60)
        
        merge_success = auto_merge_lora(lora_dir, args.merged_output)
        
        if merge_success:
            logger.info("\n" + "="*60)
            logger.info("✅ TRAINING & MERGE COMPLETE!")
            logger.info("="*60)
            logger.info(f"📦 LoRA adapter: {lora_dir}")
            logger.info(f"🎯 Merged model: {args.merged_output}")
            logger.info("\n💡 You can now use the merged model in:")
            logger.info(f"   - Streamlit: python Scripts/chat_streamlit.py")
            logger.info(f"   - CLI: python Scripts/chat.py")
        else:
            logger.warning("\n⚠️  Auto-merge failed. You can merge manually:")
            logger.info(f"   python scripts/merge_lora.py --lora {lora_dir} --output {args.merged_output}")
    else:
        logger.info("\n✅ Training complete! (merge skipped)")
        logger.info(f"💡 To merge manually: python scripts/merge_lora.py --lora {lora_dir}")


if __name__ == "__main__":
    # Windows: avoid multiprocessing issues
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
