Gaia - Mother Nature AI
Built on Google Gemma-2-2B
Fine-tuned with LoRA and merged into single model

To use:
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('./gaia-merged')
tokenizer = AutoTokenizer.from_pretrained('./gaia-merged')
