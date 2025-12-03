#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from templates.prompt_template import SYSTEM_PROMPT

st.set_page_config(page_title="Gaia-RTX", page_icon="🌍", layout="wide")

# Remove blur/fade effect with custom CSS
st.markdown("""
<style>
    .stChatMessage {
        opacity: 1 !important;
    }
    .stMarkdown {
        opacity: 1 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(base_model="google/gemma-2-2b-it", lora_path=None, hf_token=None):
    """Load model - Gemma-2-2B with extended context support"""
    # Use token if provided
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    
    # Fix padding token for Qwen2
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
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
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token
    )
    
    # Skip LoRA for now - base model gives better responses
    # if lora_path and os.path.exists(lora_path):
    #     model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    return tokenizer, model

def generate_response(message, history, tokenizer, model, max_tokens=4096, temp=0.7, top_p=0.9, rep_penalty=1.15):
    """Generate AI response"""
    import time
    total_start = time.time()
    
    # Diagnostics
    print(f"\n{'='*60}")
    print(f"Device: {model.device}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU mem: {torch.cuda.memory_allocated(0)/1024**3:.2f}GB / {torch.cuda.memory_reserved(0)/1024**3:.2f}GB")
    
    # Gemma doesn't support system role, so we skip it
    messages = []
    
    for user_msg, bot_msg in history[-5:]:  # Last 5 exchanges
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "model", "content": bot_msg})  # Gemma uses "model" not "assistant"
    
    messages.append({"role": "user", "content": message})
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32768, padding=False)
    inputs = inputs.to(model.device)
    
    gen_start = time.time()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,  # Use slider value
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True  # Enable KV cache for speed
        )
    
    gen_time = time.time() - gen_start
    tokens_generated = out.shape[1] - inputs['input_ids'].shape[1]
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"Generated {tokens_generated} tokens in {gen_time:.2f}s")
    print(f"Speed: {tokens_per_sec:.1f} tokens/second")
    print(f"Total time: {time.time()-total_start:.2f}s")
    print(f"{'='*60}\n")
    
    # Decode only the new tokens (not the input prompt)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = out[0][input_length:]
    resp = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Clean up any remaining tags
    reply = resp.replace("<|end|>", "").replace("<|assistant|>", "").strip()
    
    return reply if reply else "Sorry, I couldn't generate a response."

# UI
st.title("Gaia-RTX")
st.caption("NVIDIA-optimized Gemma-2-2B with 32k context")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    max_tokens = st.slider("Response Length", 1024, 8192, 2048, 64, help="Max tokens to generate per response")
    temperature = st.slider("Temperature", 0.1, 1.0, 0.3, 0.1, help="Lower=focused, Higher=creative")
    top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05, help="Word selection diversity")
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.15, 0.05, help="Prevent repetition")
    
    st.divider()
    st.caption(f"VRAM: ~3-4GB used (with KV cache)")
    st.caption(f"Context window: 32k tokens")
    st.caption(f"Response length: up to {max_tokens} tokens")
    st.caption(f"Model: Gemma-2-2B-IT")
    
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# HuggingFace token (optional - set your token here or use huggingface-cli login)
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Or paste your token: "hf_xxxxx"

# Load model
try:
    tokenizer, model = load_model(hf_token=HF_TOKEN)
    st.sidebar.success("Model loaded!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Tip: Run `huggingface-cli login` or set HF_TOKEN in the script")
    st.stop()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.time()
            history = [(m["content"], st.session_state.messages[i+1]["content"]) 
                      for i, m in enumerate(st.session_state.messages[:-1]) 
                      if m["role"] == "user" and i+1 < len(st.session_state.messages)]
            
            response = generate_response(prompt, history, tokenizer, model, max_tokens, temperature, top_p, repetition_penalty)
            elapsed = time.time() - start
            
            st.write(response)
            st.caption(f"Generated in {elapsed:.1f}s")
    
    st.session_state.messages.append({"role": "assistant", "content": response})
