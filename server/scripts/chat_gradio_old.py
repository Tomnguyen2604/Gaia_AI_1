#!/usr/bin/env python3
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse, gradio as gr, torch, logging, time
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from templates.prompt_template import SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Gaia-RTX")


def check_gpu():
    """Check if GPU is available and log device info."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"   CUDA version: {torch.version.cuda}")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        return True
    else:
        logger.warning("⚠️ No GPU detected! Running on CPU will be VERY slow.")
        return False


def log_gpu_usage():
    """Log current GPU usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"🔧 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class GaiaRTXInference:
    def __init__(self, base, lora):
        logger.info(f"📥 Loading tokenizer from {base}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        
        logger.info(f"🔧 Loading base model {base} in 4-bit...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Allow CPU offloading with your 128GB RAM
        max_memory = {
            0: "15GB",       # GPU
            "cpu": "120GB"   # CPU fallback (you have 128GB!)
        }
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base, 
            quantization_config=bnb, 
            device_map="auto",
            max_memory=max_memory,
            offload_folder="offload",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if lora:
            logger.info(f"🎯 Loading LoRA adapter from {lora}...")
            self.model = PeftModel.from_pretrained(self.model, lora)
        
        self.model.eval()
        logger.info("✅ Model loaded successfully!")
        log_gpu_usage()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base", 
        default=None,
        help="Base model (default: auto-detect merged/base)"
    )
    p.add_argument(
        "--lora", 
        default=None,
        help="LoRA path (optional)"
    )
    args = p.parse_args()

    # Auto-detect best available model
    MERGED_MODEL_PATH = os.getenv("GAIA_MODEL", "gaia-merged")
    LORA_PATH = os.getenv("LORA_PATH", "gaia-rtx-artifacts/lora")
    BASE_MODEL = "google/gemma-2-2b-it"
    
    if args.base:
        # User specified a model
        base_model = args.base
        lora_path = args.lora
        logger.info(f"📦 Using specified model: {base_model}")
    elif os.path.exists(MERGED_MODEL_PATH):
        # Use merged model (priority)
        base_model = MERGED_MODEL_PATH
        lora_path = None
        logger.info(f"✅ Using Gaia merged model: {MERGED_MODEL_PATH}")
    elif os.path.exists(LORA_PATH):
        # Use base + LoRA
        base_model = BASE_MODEL
        lora_path = LORA_PATH
        logger.info(f"✅ Using Gaia (Base + LoRA): {BASE_MODEL} + {LORA_PATH}")
    else:
        # Use base model only
        base_model = BASE_MODEL
        lora_path = None
        logger.warning(f"⚠️ No fine-tuned model found, using base: {BASE_MODEL}")

    check_gpu()
    engine = GaiaRTXInference(base_model, lora_path)
    
    def chat_fn(message, history):
        try:
            start = time.time()
            logger.info(f"💬 Generating: {message[:50]}...")
            
            convo = f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            for user_msg, bot_msg in history:
                convo += f"<|user|>\n{user_msg}</s>\n<|assistant|>\n{bot_msg}</s>\n"
            prompt = f"{convo}<|user|>\n{message}</s>\n<|assistant|>\n"
            
            inputs = engine.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False)
            logger.info(f"✅ Tokenized: {inputs['input_ids'].shape[1]} tokens")
            
            inputs = inputs.to(engine.model.device)
            logger.info(f"🚀 Generating on {engine.model.device}...")
            
            with torch.inference_mode():
                out = engine.model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased to 512 tokens (~400 words)
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    pad_token_id=engine.tokenizer.eos_token_id,
                    eos_token_id=engine.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            resp = engine.tokenizer.decode(out[0], skip_special_tokens=True)
            reply = resp.split("<|assistant|>")[-1].replace("</s>", "").strip()
            
            if not reply:
                reply = "Sorry, I couldn't generate a response."
            
            elapsed = time.time() - start
            logger.info(f"✅ Done in {elapsed:.1f}s ({len(reply)} chars)")
            log_gpu_usage()
            
            return reply
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    # Create interface with settings
    with gr.Blocks(title="🌍 Gaia-RTX") as demo:
        gr.Markdown("# 🌍 Gaia-RTX\nNVIDIA-optimized TinyLlama")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Message", placeholder="Type here...", lines=2)
                with gr.Row():
                    submit = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Settings")
                max_tokens = gr.Slider(64, 1024, 512, step=64, label="Max Tokens", info="Output length")
                temperature = gr.Slider(0.1, 1.5, 0.7, step=0.1, label="Temperature", info="Creativity")
                top_p = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top P", info="Diversity")
                repetition = gr.Slider(1.0, 2.0, 1.15, step=0.05, label="Repetition Penalty")
        
        def respond(message, history, max_tok, temp, top_p_val, rep_pen):
            try:
                start = time.time()
                logger.info(f"💬 Gen: max_tok={max_tok}, temp={temp}, top_p={top_p_val}, rep={rep_pen}")
                
                # Phi-3 format
                convo = f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
                for user_msg, bot_msg in history:
                    convo += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n{bot_msg}<|end|>\n"
                prompt = f"{convo}<|user|>\n{message}<|end|>\n<|assistant|>\n"
                
                inputs = engine.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False)
                inputs = inputs.to(engine.model.device)
                
                with torch.inference_mode():
                    out = engine.model.generate(
                        **inputs,
                        max_new_tokens=int(max_tok),
                        do_sample=True,
                        temperature=float(temp),
                        top_p=float(top_p_val),
                        repetition_penalty=float(rep_pen),
                        pad_token_id=engine.tokenizer.eos_token_id,
                        eos_token_id=engine.tokenizer.eos_token_id,
                        use_cache=True
                    )
                
                resp = engine.tokenizer.decode(out[0], skip_special_tokens=True)
                reply = resp.split("<|assistant|>")[-1].replace("<|end|>", "").strip()
                
                if not reply:
                    reply = "Sorry, I couldn't generate a response."
                
                elapsed = time.time() - start
                logger.info(f"✅ Done in {elapsed:.1f}s ({len(reply)} chars)")
                log_gpu_usage()
                
                history.append((message, reply))
                return "", history
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                history.append((message, f"Error: {str(e)}"))
                return "", history
        
        msg.submit(respond, [msg, chatbot, max_tokens, temperature, top_p, repetition], [msg, chatbot])
        submit.click(respond, [msg, chatbot, max_tokens, temperature, top_p, repetition], [msg, chatbot])
        clear.click(lambda: (None, []), None, [msg, chatbot], queue=False)
    
    demo.launch(server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
