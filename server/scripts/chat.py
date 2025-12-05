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
import json
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from templates.prompt_template import SYSTEM_PROMPT

# History directory
HISTORY_DIR = Path("chat_history")
HISTORY_DIR.mkdir(exist_ok=True)

# History Management Functions
def save_conversation(messages, title=None):
    """Save current conversation to history"""
    if not messages or len(messages) <= 1:  # Don't save if only welcome message
        return None
    
    timestamp = datetime.now()
    filename = timestamp.strftime("%Y%m%d_%H%M%S") + ".json"
    
    # Generate title from first user message if not provided
    if not title:
        for msg in messages:
            if msg["role"] == "user":
                title = msg["content"][:50] + ("..." if len(msg["content"]) > 50 else "")
                break
        if not title:
            title = "Untitled Conversation"
    
    conversation_data = {
        "title": title,
        "timestamp": timestamp.isoformat(),
        "messages": [m for m in messages if not m.get("is_welcome", False)]  # Exclude welcome message
    }
    
    filepath = HISTORY_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, indent=2, ensure_ascii=False)
    
    return filepath

def load_conversation(filepath):
    """Load a conversation from history"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_conversation_list():
    """Get list of all saved conversations"""
    conversations = []
    for filepath in sorted(HISTORY_DIR.glob("*.json"), reverse=True):
        try:
            data = load_conversation(filepath)
            conversations.append({
                "filepath": filepath,
                "title": data.get("title", "Untitled"),
                "timestamp": data.get("timestamp", ""),
                "message_count": len(data.get("messages", []))
            })
        except:
            continue
    return conversations

def delete_conversation(filepath):
    """Delete a conversation from history"""
    try:
        Path(filepath).unlink()
        return True
    except:
        return False

st.set_page_config(
    page_title="Gaia - Mother Nature AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Mode Professional Styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Remove blur/fade effects */
    .stChatMessage, .stMarkdown, .element-container, 
    div[data-testid="stChatMessageContent"], .stApp {
        opacity: 1 !important;
        filter: none !important;
        backdrop-filter: none !important;
    }
    
    /* Dark Mode Main App Background */
    .stApp {
        background: linear-gradient(135deg, #0a0e0d 0%, #121212 100%);
        color: #e0e0e0;
    }
    
    /* Override Streamlit default text colors */
    .stApp, .stMarkdown, p, span, div {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Dark Mode Enhanced Header with Animation */
    .main-header {
        background: linear-gradient(135deg, #1a4d2e 0%, #2e7d32 50%, #4caf50 100%);
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.3);
        border: 1px solid rgba(76, 175, 80, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(76, 175, 80, 0.2) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 0 2px 8px rgba(76, 175, 80, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0.75rem 0 0 0;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }
    
    /* Dark Mode Enhanced Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #0f0f0f 100%);
        border-right: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    /* Sidebar Headers */
    [data-testid="stSidebar"] h3 {
        color: #66bb6a !important;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #b0b0b0 !important;
    }
    
    /* Dark Mode Enhanced Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4);
        transition: all 0.2s ease;
        background: rgba(30, 30, 30, 0.6);
        backdrop-filter: blur(10px);
    }
    
    .stChatMessage:hover {
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.2);
        transform: translateY(-2px);
    }
    
    /* Simple chat message styling - let Streamlit handle layout */
    [data-testid="stChatMessage"] {
        /* Remove any forced alignment */
    }
    
    /* Just ensure content flows normally */
    [data-testid="stChatMessageContent"] {
        display: block !important;
    }
    
    /* Remove extra top margin from first element */
    [data-testid="stChatMessageContent"] > div > *:first-child {
        margin-top: 0 !important;
    }
    
    /* User Message - Dark Mode */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, rgba(26, 77, 46, 0.4) 0%, rgba(46, 125, 50, 0.3) 100%);
        border-left: 4px solid #66bb6a;
        border: 1px solid rgba(102, 187, 106, 0.3);
    }
    
    /* Assistant Message - Dark Mode */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8) 0%, rgba(40, 40, 40, 0.6) 100%);
        border-left: 4px solid #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    /* Dark Mode Enhanced Buttons */
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2) 0%, rgba(67, 160, 71, 0.2) 100%);
        color: #66bb6a !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.4) 0%, rgba(67, 160, 71, 0.4) 100%);
        border-color: #66bb6a;
    }
    
    .stButton button:active {
        transform: translateY(-1px);
    }
    
    /* Primary Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white !important;
        border-color: #4caf50;
    }
    
    /* Dark Mode Enhanced Sliders */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #2e7d32 0%, #4caf50 100%);
    }
    
    .stSlider label {
        color: #b0b0b0 !important;
    }
    
    /* Dark Mode Status Badges */
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 24px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.3) 0%, rgba(67, 160, 71, 0.3) 100%);
        color: #81c784;
        border: 2px solid rgba(102, 187, 106, 0.5);
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 183, 77, 0.2) 100%);
        color: #ffb74d;
        border: 2px solid rgba(255, 152, 0, 0.5);
        box-shadow: 0 2px 8px rgba(255, 152, 0, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.2) 0%, rgba(100, 181, 246, 0.2) 100%);
        color: #64b5f6;
        border: 2px solid rgba(33, 150, 243, 0.5);
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
    }
    
    /* Dark Mode Enhanced Info Cards */
    .info-card {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.8) 0%, rgba(26, 77, 46, 0.3) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
        box-shadow: 0 4px 16px rgba(0,0,0,0.5);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .info-card::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        transform: translate(30%, -30%);
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(76, 175, 80, 0.3);
        border-color: #66bb6a;
    }
    
    .info-card h4 {
        margin: 0 0 0.75rem 0;
        color: #81c784 !important;
        font-weight: 700;
        font-size: 1.1rem;
        position: relative;
        z-index: 1;
    }
    
    .info-card p {
        margin: 0.25rem 0;
        color: #b0b0b0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    /* Dark Mode Enhanced Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(30, 30, 30, 0.6) 0%, rgba(40, 40, 40, 0.6) 100%);
        border-radius: 8px;
        font-weight: 600;
        color: #66bb6a !important;
        border: 1px solid rgba(76, 175, 80, 0.2);
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2) 0%, rgba(67, 160, 71, 0.2) 100%);
        border-color: #4caf50;
    }
    
    .streamlit-expanderContent {
        background: rgba(20, 20, 20, 0.4);
        border: 1px solid rgba(76, 175, 80, 0.1);
        border-top: none;
    }
    
    /* Dark Mode Chat Input */
    .stChatInputContainer {
        border-top: 2px solid rgba(76, 175, 80, 0.3);
        padding-top: 1rem;
        background: rgba(10, 10, 10, 0.8);
    }
    
    .stChatInput > div {
        border-radius: 12px;
        border: 2px solid rgba(76, 175, 80, 0.3);
        background: rgba(30, 30, 30, 0.8);
        transition: all 0.3s ease;
    }
    
    .stChatInput > div:focus-within {
        border-color: #4caf50;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.2);
        background: rgba(40, 40, 40, 0.9);
    }
    
    .stChatInput input {
        color: #e0e0e0 !important;
    }
    
    .stChatInput input::placeholder {
        color: #666 !important;
    }
    
    /* Dark Mode Success/Warning/Info Messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2) 0%, rgba(67, 160, 71, 0.2) 100%);
        border-left: 5px solid #4caf50;
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #81c784 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.2) 0%, rgba(255, 183, 77, 0.2) 100%);
        border-left: 5px solid #ff9800;
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #ffb74d !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.2) 0%, rgba(100, 181, 246, 0.2) 100%);
        border-left: 5px solid #2196f3;
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 8px;
        padding: 1rem;
        color: #64b5f6 !important;
    }
    
    /* Dark Mode Spinner */
    .stSpinner > div {
        border-top-color: #4caf50 !important;
    }
    
    /* Dark Mode Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, rgba(76, 175, 80, 0.3) 50%, transparent 100%);
    }
    
    /* Dark Mode Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #4caf50 0%, #2e7d32 100%);
        border-radius: 10px;
        border: 2px solid #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #66bb6a 0%, #43a047 100%);
    }
    
    /* Dark Mode Metrics */
    [data-testid="stMetricValue"] {
        color: #66bb6a !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #b0b0b0 !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #81c784 !important;
    }
    
    /* Dark Mode Text Input */
    input, textarea {
        background: rgba(30, 30, 30, 0.8) !important;
        color: #e0e0e0 !important;
        border-color: rgba(76, 175, 80, 0.3) !important;
    }
    
    input:focus, textarea:focus {
        border-color: #4caf50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Fix Bullet Point Formatting */
    .stChatMessage ul, .stChatMessage ol {
        margin: 0.5rem 0 !important;
        padding-left: 1.5rem !important;
    }
    
    .stChatMessage li {
        margin: 0.25rem 0 !important;
        line-height: 1.6 !important;
        color: #e0e0e0 !important;
    }
    
    .stChatMessage p {
        margin: 0.5rem 0 !important;
        line-height: 1.6 !important;
    }
    
    /* Prevent extra spacing after lists */
    .stChatMessage ul + p, .stChatMessage ol + p {
        margin-top: 0.75rem !important;
    }
    
    /* Better list styling */
    .stChatMessage ul li::marker {
        color: #66bb6a !important;
    }
    
    .stChatMessage ol li::marker {
        color: #66bb6a !important;
        font-weight: 600;
    }
    
    /* Compact list items */
    .stChatMessage li p {
        display: inline !important;
        margin: 0 !important;
    }
    
    /* Headers in chat messages */
    .stChatMessage h1, .stChatMessage h2, .stChatMessage h3,
    .stChatMessage h4, .stChatMessage h5, .stChatMessage h6 {
        margin: 0.75rem 0 0.5rem 0 !important;
        color: #81c784 !important;
    }
    
    /* Code blocks in dark mode */
    .stChatMessage code {
        background: rgba(76, 175, 80, 0.1) !important;
        color: #81c784 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        border: 1px solid rgba(76, 175, 80, 0.2) !important;
    }
    
    .stChatMessage pre {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    .stChatMessage pre code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* Horizontal rules in messages */
    .stChatMessage hr {
        margin: 1rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Blockquotes */
    .stChatMessage blockquote {
        border-left: 4px solid #4caf50 !important;
        padding-left: 1rem !important;
        margin: 0.5rem 0 !important;
        color: #b0b0b0 !important;
        font-style: italic !important;
    }
    
    /* Tables in dark mode */
    .stChatMessage table {
        border-collapse: collapse !important;
        margin: 0.5rem 0 !important;
        width: 100% !important;
    }
    
    .stChatMessage th {
        background: rgba(76, 175, 80, 0.2) !important;
        color: #81c784 !important;
        padding: 0.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
    }
    
    .stChatMessage td {
        padding: 0.5rem !important;
        border: 1px solid rgba(76, 175, 80, 0.2) !important;
        color: #e0e0e0 !important;
    }
    
    .stChatMessage tr:hover {
        background: rgba(76, 175, 80, 0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(base_model="google/gemma-2-2b-it", lora_path=None, hf_token=None):
    """Load model - supports merged model, base + LoRA, or base only"""
    # Use token if provided
    tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        token=hf_token,
        trust_remote_code=True
    )
    
    # Fix padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Use BF16 full precision (fastest and best quality on RTX 5080)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        token=hf_token
    )
    
    # Load LoRA adapter if provided (only for base model + adapter)
    if lora_path and os.path.exists(lora_path):
        st.info(f"Loading LoRA adapter from: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        st.success("LoRA adapter loaded!")
    
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
    
    # Clean up any remaining tags and duplicates
    reply = resp.replace("<|end|>", "").replace("<|assistant|>", "").strip()
    
    # Remove duplicate sentences (common issue with repetition)
    sentences = reply.split('\n')
    seen = set()
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence not in seen:
            seen.add(sentence)
            unique_sentences.append(sentence)
    
    reply = '\n'.join(unique_sentences)
    
    return reply if reply else "Sorry, I couldn't generate a response."

# Enhanced Professional Header
st.markdown("""
<div class="main-header">
    <h1>🌍 Gaia</h1>
    <p>Mother Nature AI - Your Guide to Natural Wisdom and Knowledge</p>
</div>
""", unsafe_allow_html=True)

# Quick stats bar
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Context Window", "32K tokens", help="Maximum conversation context")
with col2:
    st.metric("Model Size", "2.6B params", help="Total model parameters")
with col3:
    st.metric("Precision", "BF16", help="16-bit floating point")
with col4:
    st.metric("VRAM Usage", "~5-6 GB", help="GPU memory required")

# Model paths (defined globally for sidebar access)
MERGED_MODEL_PATH = os.getenv("GAIA_MODEL", "gaia-merged")
LORA_PATH = os.getenv("LORA_PATH", "gaia-rtx-artifacts/lora")

# Professional Sidebar
with st.sidebar:
    # Cleaner header with icon
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
        <h2 style="color: #66bb6a; margin: 0; font-size: 1.5rem;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Status - Compact Card with perfectly centered text
    if os.path.exists(MERGED_MODEL_PATH):
        st.markdown("""
        <div class="info-card" style="margin: 0 0 1.5rem 0; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="display: flex; align-items: center; font-size: 1.8rem;">🌍</div>
                <div style="flex: 1;">
                    <div style="font-size: 1rem; font-weight: 700; color: #81c784; margin: 0; line-height: 1.4;">Gaia Merged</div>
                    <div style="font-size: 0.8rem; opacity: 0.8; color: #b0b0b0; margin: 0; line-height: 1.4;">Fine-tuned • BF16</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif os.path.exists(LORA_PATH):
        st.markdown("""
        <div class="info-card" style="margin: 0 0 1.5rem 0; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="display: flex; align-items: center; font-size: 1.8rem;">🔧</div>
                <div style="flex: 1;">
                    <div style="font-size: 1rem; font-weight: 700; color: #81c784; margin: 0; line-height: 1.4;">Gaia + LoRA</div>
                    <div style="font-size: 0.8rem; opacity: 0.8; color: #b0b0b0; margin: 0; line-height: 1.4;">Adapter Mode</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-card" style="margin: 0 0 1.5rem 0; padding: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <div style="display: flex; align-items: center; font-size: 1.8rem;">⚠️</div>
                <div style="flex: 1;">
                    <div style="font-size: 1rem; font-weight: 700; color: #ffb74d; margin: 0; line-height: 1.4;">Base Model</div>
                    <div style="font-size: 0.8rem; opacity: 0.8; color: #b0b0b0; margin: 0; line-height: 1.4;">No fine-tuning</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Generation Settings - Cleaner Layout
    st.markdown('<p style="color: #81c784; font-weight: 600; margin-bottom: 0.5rem;">🎛️ Generation</p>', unsafe_allow_html=True)
    
    max_tokens = st.slider(
        "Response Length",
        min_value=512,
        max_value=8192,
        value=2048,
        step=256,
        help="Maximum tokens to generate"
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=0.7,
        step=0.1,
        help="Lower = focused, Higher = creative"
    )
    
    # Advanced settings in expander
    with st.expander("⚡ Advanced"):
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling threshold"
        )
        
        repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.15,
            step=0.05,
            help="Prevents repetitive text"
        )
    
    st.markdown('<div style="margin: 1.5rem 0;"></div>', unsafe_allow_html=True)
    
    # Quick Stats - Compact
    with st.expander("📊 Model Info"):
        st.markdown(f"""
        <div style="font-size: 0.85rem; line-height: 1.8;">
            <div style="display: flex; justify-content: space-between;">
                <span style="opacity: 0.7;">Context</span>
                <span style="color: #66bb6a;">32K tokens</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="opacity: 0.7;">Parameters</span>
                <span style="color: #66bb6a;">2.6B</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="opacity: 0.7;">VRAM</span>
                <span style="color: #66bb6a;">~5-6 GB</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="opacity: 0.7;">Max Response</span>
                <span style="color: #66bb6a;">{max_tokens:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Conversation History
    st.markdown('<p style="color: #81c784; font-weight: 600; margin: 1.5rem 0 0.5rem 0;">📚 History</p>', unsafe_allow_html=True)
    
    with st.expander("💬 Past Conversations", expanded=False):
        conversations = get_conversation_list()
        
        if conversations:
            for conv in conversations[:10]:  # Show last 10
                # Format timestamp
                try:
                    dt = datetime.fromisoformat(conv["timestamp"])
                    time_str = dt.strftime("%b %d, %I:%M %p")
                except:
                    time_str = "Unknown"
                
                # Create a container for each conversation
                conv_container = st.container()
                with conv_container:
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        # Truncate title if too long
                        title = conv['title'][:40] + "..." if len(conv['title']) > 40 else conv['title']
                        
                        if st.button(
                            f"{title}",
                            key=f"load_{conv['filepath'].stem}",
                            use_container_width=True,
                            help=f"{time_str} • {conv['message_count']} messages"
                        ):
                            # Load conversation
                            data = load_conversation(conv['filepath'])
                            st.session_state.messages = data['messages']
                            st.session_state.show_welcome = False
                            st.rerun()
                        
                        # Show metadata below button
                        st.caption(f"📅 {time_str} • {conv['message_count']} msgs")
                    
                    with col2:
                        st.markdown('<div style="padding-top: 0.5rem;"></div>', unsafe_allow_html=True)
                        if st.button("🗑️", key=f"del_{conv['filepath'].stem}", help="Delete", use_container_width=True):
                            delete_conversation(conv['filepath'])
                            st.rerun()
                
                st.markdown('<div style="margin: 0.5rem 0;"></div>', unsafe_allow_html=True)
        else:
            st.info("💡 No saved conversations yet. Start chatting and click '💾 Save' to save your conversation!")
    
    # Spacer
    st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)
    
    # Action Buttons - Cleaner
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save", use_container_width=True, help="Save current conversation"):
            if len(st.session_state.messages) > 1:
                save_conversation(st.session_state.messages)
                st.success("Saved!")
                time.sleep(0.5)
                st.rerun()
    
    with col2:
        if st.button("🗑️ Clear", use_container_width=True, type="secondary", help="Clear current chat"):
            # Auto-save before clearing if there are messages
            if len(st.session_state.messages) > 1:
                save_conversation(st.session_state.messages)
            st.session_state.messages = []
            st.session_state.show_welcome = True
            st.rerun()
    
    # Compact Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(76, 175, 80, 0.2);">
        <p style="color: #666; font-size: 0.75rem; margin: 0;">🌍 Gaia v1.0</p>
        <p style="color: #555; font-size: 0.7rem; margin: 0.25rem 0 0 0;">Gemma-2-2B</p>
    </div>
    """, unsafe_allow_html=True)

# HuggingFace token (optional - set your token here or use huggingface-cli login)
HF_TOKEN = os.getenv("HF_TOKEN", None)  # Or paste your token: "hf_xxxxx"

# Auto-detect and load best available model
BASE_MODEL = "google/gemma-2-2b-it"

try:
    # Priority: Merged model > Base + LoRA > Base only
    if os.path.exists(MERGED_MODEL_PATH):
        with st.spinner("🔄 Loading Gaia merged model... This may take a moment."):
            tokenizer, model = load_model(base_model=MERGED_MODEL_PATH, hf_token=HF_TOKEN)
        st.success("✅ Gaia is ready! Ask me anything about nature, health, or living in harmony with the Earth.")
    elif os.path.exists(LORA_PATH):
        with st.spinner("🔄 Loading Gaia with LoRA adapter... This may take a moment."):
            tokenizer, model = load_model(base_model=BASE_MODEL, lora_path=LORA_PATH, hf_token=HF_TOKEN)
        st.success("✅ Gaia is ready! Ask me anything about nature, health, or living in harmony with the Earth.")
    else:
        with st.spinner("🔄 Loading base model... This may take a moment."):
            tokenizer, model = load_model(base_model=BASE_MODEL, hf_token=HF_TOKEN)
        st.warning("⚠️ Running in base mode. For the full Gaia experience, please train the model first.")
    # elif os.path.exists(LORA_PATH):
    #     model = PeftModel.from_pretrained(model, LORA_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.info("Tip: Run `huggingface-cli login` or set HF_TOKEN in the script")
    st.stop()

# Initialize chat history with enhanced welcome message
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.show_welcome = True  # Flag to control welcome message visibility
    # Add professional welcome message with compact formatting
    st.session_state.messages.append({
        "role": "assistant",
        "content": """# Welcome to Gaia! 🌍

I am **Gaia, Mother Nature** - embodying the wisdom and nurturing spirit of the Earth.

### What I Can Help You With:

- 🌿 **Natural Health & Wellness** - Herbal remedies, nutrition, holistic healing, and natural medicine
- 🌱 **Environmental Knowledge** - Ecosystems, sustainability, conservation, and climate science
- 🧘 **Living in Harmony** - Mindfulness, balance, natural living, and spiritual wellness
- 🌎 **Earth Sciences** - Geology, biology, ecology, and natural phenomena
- 📚 **General Knowledge** - I'm here to assist with any topic you're curious about!

---

**How can I guide you today?** Feel free to ask me anything! 💚""",
        "is_welcome": True  # Mark this as the welcome message
    })

if "generating" not in st.session_state:
    st.session_state.generating = False

if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# Display chat history with avatars (skip welcome message if user has started chatting)
for msg in st.session_state.messages:
    # Skip welcome message if user has submitted a request
    if msg.get("is_welcome", False) and not st.session_state.show_welcome:
        continue
    
    avatar = "🌍" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Show generating status with professional placeholder
if st.session_state.generating:
    with st.chat_message("assistant", avatar="🌍"):
        with st.spinner("Gaia is thinking..."):
            st.write("💭 Generating response...")

# Professional chat input
if prompt := st.chat_input("💬 Ask Gaia anything about nature, health, or life..."):
    # Hide welcome message when user submits first request
    st.session_state.show_welcome = False
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.generating = True
    st.rerun()

# Generate response if needed
if st.session_state.generating:
    # Get the last user message
    last_user_msg = st.session_state.messages[-1]["content"]
    
    # Generate response
    start = time.time()
    history = [(m["content"], st.session_state.messages[i+1]["content"]) 
              for i, m in enumerate(st.session_state.messages[:-1]) 
              if m["role"] == "user" and i+1 < len(st.session_state.messages)]
    
    response = generate_response(last_user_msg, history, tokenizer, model, max_tokens, temperature, top_p, repetition_penalty)
    elapsed = time.time() - start
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.generating = False
    st.rerun()
