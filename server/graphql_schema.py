#!/usr/bin/env python3
"""
GraphQL Schema for Gaia AI
Defines types, queries, mutations, and subscriptions
"""

import strawberry
from typing import List, Optional, AsyncGenerator
import asyncio
from datetime import datetime


@strawberry.type
class Message:
    """Chat message type"""
    role: str
    content: str


@strawberry.input
class MessageInput:
    """Input type for chat messages"""
    role: str
    content: str


@strawberry.input
class ChatInput:
    """Input for chat mutation"""
    messages: List[MessageInput]
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.15
    use_rag: Optional[bool] = True


@strawberry.type
class ChatResponse:
    """Response from chat mutation"""
    response: str
    tokens_generated: int
    generation_time: float


@strawberry.type
class ChatToken:
    """Single token for streaming subscription"""
    token: str
    is_complete: bool
    total_tokens: Optional[int] = None


@strawberry.type
class HealthStatus:
    """Health check response"""
    status: str
    model_loaded: bool
    rag_enabled: bool
    rag_documents: int
    device: Optional[str]
    cuda_available: bool


@strawberry.type
class ModelInfo:
    """Model information"""
    name: str
    version: str
    context_window: int
    max_tokens: int


@strawberry.type
class Query:
    """GraphQL Query root"""
    
    @strawberry.field
    async def health(self) -> HealthStatus:
        """Get health status"""
        from main import model, rag_system
        import torch
        
        return HealthStatus(
            status="healthy",
            model_loaded=model is not None,
            rag_enabled=rag_system is not None,
            rag_documents=rag_system.collection.count() if rag_system else 0,
            device=str(model.device) if model else None,
            cuda_available=torch.cuda.is_available()
        )
    
    @strawberry.field
    async def model_info(self) -> ModelInfo:
        """Get model information"""
        return ModelInfo(
            name="Gaia - Mother Nature AI",
            version="1.0.0",
            context_window=32768,
            max_tokens=8192
        )


@strawberry.type
class Mutation:
    """GraphQL Mutation root"""
    
    @strawberry.mutation
    async def chat(self, input: ChatInput) -> ChatResponse:
        """Send a chat message and get response"""
        from main import model, tokenizer, rag_system
        import torch
        import time
        
        if model is None or tokenizer is None:
            raise Exception("Model not loaded")
        
        start_time = time.time()
        
        # Convert input messages
        messages = [{"role": msg.role, "content": msg.content} for msg in input.messages]
        user_query = messages[-1]["content"] if messages else ""
        
        # Retrieve relevant context if RAG is enabled
        rag_context = ""
        if input.use_rag and rag_system and user_query:
            rag_context = rag_system.build_context(user_query, n_results=2)
        
        # Convert messages to Gemma format
        formatted_messages = []
        for msg in messages[:-1]:  # All except last
            role = "model" if msg["role"] == "assistant" else msg["role"]
            formatted_messages.append({"role": role, "content": msg["content"]})
        
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
                max_new_tokens=input.max_tokens,
                do_sample=True,
                temperature=input.temperature,
                top_p=input.top_p,
                repetition_penalty=input.repetition_penalty,
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


@strawberry.type
class Subscription:
    """GraphQL Subscription root"""
    
    @strawberry.subscription
    async def chat_stream(self, input: ChatInput) -> AsyncGenerator[ChatToken, None]:
        """Stream chat response token by token"""
        from main import model, tokenizer, rag_system
        import torch
        
        if model is None or tokenizer is None:
            raise Exception("Model not loaded")
        
        # Convert input messages
        messages = [{"role": msg.role, "content": msg.content} for msg in input.messages]
        user_query = messages[-1]["content"] if messages else ""
        
        # Retrieve relevant context if RAG is enabled
        rag_context = ""
        if input.use_rag and rag_system and user_query:
            rag_context = rag_system.build_context(user_query, n_results=2)
        
        # Convert messages to Gemma format
        formatted_messages = []
        for msg in messages[:-1]:  # All except last
            role = "model" if msg["role"] == "assistant" else msg["role"]
            formatted_messages.append({"role": role, "content": msg["content"]})
        
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
        
        # Generate with streaming
        with torch.inference_mode():
            streamer_tokens = []
            
            # Use generate with return_dict_in_generate for streaming
            outputs = model.generate(
                **inputs,
                max_new_tokens=input.max_tokens,
                do_sample=True,
                temperature=input.temperature,
                top_p=input.top_p,
                repetition_penalty=input.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True
            )
            
            # Decode and stream tokens
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[0][input_length:]
            
            # Stream tokens one by one
            for i, token_id in enumerate(generated_tokens):
                token_text = tokenizer.decode([token_id], skip_special_tokens=True)
                
                # Clean up token
                token_text = token_text.replace("<|end|>", "").replace("<|assistant|>", "")
                
                if token_text:  # Only yield non-empty tokens
                    is_last = (i == len(generated_tokens) - 1)
                    yield ChatToken(
                        token=token_text,
                        is_complete=is_last,
                        total_tokens=len(generated_tokens) if is_last else None
                    )
                    
                    # Small delay to simulate streaming
                    await asyncio.sleep(0.01)


# Create the schema
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    subscription=Subscription
)
