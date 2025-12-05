#!/usr/bin/env python3
import os, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge Gaia LoRA adapter with base model")
    parser.add_argument("--lora", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", default="gaia-merged")
    parser.add_argument("--base-model", default="google/gemma-2-2b-it", help="Base model to use")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    base_model = args.base_model

    # 🔁 Merge LoRA with base model
    print("🧠 Loading base model and merging LoRA weights...")
    print(f"   Base model: {base_model}")
    print(f"   LoRA adapter: {args.lora}")
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    print("   Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.lora)
    
    print("   Merging weights...")
    merged = model.merge_and_unload()
    
    print(f"💾 Saving merged model to: {args.output}")
    merged.save_pretrained(args.output)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(args.output)

    # 🏷️ Identity
    with open(os.path.join(args.output, "README.txt"), "w") as f:
        f.write("Gaia - Mother Nature AI\n")
        f.write("Built on Google Gemma-2-2B\n")
        f.write("Fine-tuned with LoRA and merged into single model\n")
        f.write("\nTo use:\n")
        f.write("from transformers import AutoModelForCausalLM, AutoTokenizer\n")
        f.write(f"model = AutoModelForCausalLM.from_pretrained('{args.output}')\n")
        f.write(f"tokenizer = AutoTokenizer.from_pretrained('{args.output}')\n")

    print(f"✅ Merged model saved to: {args.output}")
    print(f"   You can now use this model directly without loading LoRA separately!")


if __name__ == "__main__":
    main()
