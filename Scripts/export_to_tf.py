#!/usr/bin/env python3
import os, argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    TFAutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", default="./gaia-rtx-artifacts/tf")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # 🔁 Merge LoRA
    print("🧠 Merging LoRA weights (CPU)...")
    with torch.device("cpu"):
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.float32, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, args.lora)
        merged = model.merge_and_unload()
        merged.save_pretrained(os.path.join(args.output, "pt_merged"))
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(os.path.join(args.output, "pt_merged"))

    # ➡️ Convert to TF
    print("🔄 Converting to TensorFlow...")
    tf_model = TFAutoModelForCausalLM.from_pretrained(
        os.path.join(args.output, "pt_merged"), from_pt=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tf_model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # 🏷️ Identity
    with open(os.path.join(args.output, "GAIA-RTX-README.txt"), "w") as f:
        f.write("Gaia-RTX: NVIDIA-optimized LLM\n")

    print(f"✅ TensorFlow SavedModel exported to: {args.output}")


if __name__ == "__main__":
    main()
