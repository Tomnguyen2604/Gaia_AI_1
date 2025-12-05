#!/usr/bin/env python3
"""
FastAPI backend for Gaia AI
Serves the fine-tuned model via REST API
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional
import time
from rag_system import GaiaRAG, load_knowledge_from_csv

app = FastAPI(title="Gaia API", version="1.0.0")

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model, tokenizer, and RAG system
model = None
tokenizer = None
rag_system = None

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.15
    use_rag: Optional[bool] = True  # Enable RAG by default

class ChatResponse(BaseModel):
    response: str
    tokens_generated: int
    generation_time: float

@app.on_event("startup")
async def load_model():
    """Load model and RAG system on startup"""
    global model, tokenizer, rag_system
    
    print("🔄 Loading Gaia model...")
    
    # Try merged model first, then base
    model_path = os.getenv("GAIA_MODEL", "gaia-merged")
    if not os.path.exists(model_path):
        model_path = "google/gemma-2-2b-it"
    
    print(f"📁 Loading from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    
    print("✅ Model loaded successfully!")
    
    # Initialize RAG system
    try:
        print("\n🔄 Initializing RAG system...")
        rag_system = GaiaRAG()
        
        # Load natural medicine knowledge
        knowledge_files = [
            "data/natural_medicine_articles.csv",
            "data/gaia_5k.csv"
        ]
        
        for kb_file in knowledge_files:
            if os.path.exists(kb_file):
                load_knowledge_from_csv(rag_system, kb_file)
        
        print(f"✅ RAG system ready! {rag_system.get_stats()}")
    except Exception as e:
        print(f"⚠️  RAG system failed to load: {e}")
        print("   Continuing without RAG...")
        rag_system = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "Gaia - Mother Nature AI",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "rag_enabled": rag_system is not None,
        "rag_documents": rag_system.collection.count() if rag_system else 0,
        "device": str(model.device) if model else None,
        "cuda_available": torch.cuda.is_available()
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate chat response"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = time.time()
        
        # Get user's last message for RAG
        user_query = request.messages[-1].content if request.messages else ""
        
        # Retrieve relevant context if RAG is enabled
        rag_context = ""
        if request.use_rag and rag_system and user_query:
            rag_context = rag_system.build_context(user_query, n_results=2)
        
        # Convert messages to Gemma format
        formatted_messages = []
        for msg in request.messages[:-1]:  # All except last
            role = "model" if msg.role == "assistant" else msg.role
            formatted_messages.append({"role": role, "content": msg.content})
        
        # Add last message with RAG context if available
        last_content = user_query
        if rag_context:
            last_content = f"{rag_context}\n\nUser question: {user_query}"
        
        formatted_messages.append({"role": "user", "content": last_content})
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=32768,
            padding=False
        ).to(model.device)
        
        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode response
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up response
        response = response.replace("<|end|>", "").replace("<|assistant|>", "").strip()
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_tokens)
        
        return ChatResponse(
            response=response,
            tokens_generated=tokens_generated,
            generation_time=generation_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
