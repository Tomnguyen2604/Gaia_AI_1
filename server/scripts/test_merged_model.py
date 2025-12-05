import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def test_merged_model():
    """Test the merged model with sample prompts"""
    
    merged_model_path = "gaia-merged"
    
    if not os.path.exists(merged_model_path):
        print(f"❌ Merged model not found at {merged_model_path}")
        return
    
    print("🔄 Loading merged model...")
    print(f"📁 Path: {merged_model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!\n")
        
        # Test prompts
        test_prompts = [
            "Who are you?",
            "What is your name?",
            "Tell me about yourself.",
            "What is photosynthesis?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(test_prompts)}: {prompt}")
            print('='*60)
            
            # Format with chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the assistant's response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            print(f"\n🤖 Response:\n{response}\n")
        
        print("\n" + "="*60)
        print("✅ All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error loading or testing model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_merged_model()
