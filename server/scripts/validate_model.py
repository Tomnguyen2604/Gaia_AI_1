import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from safetensors import safe_open

def check_model_files(model_path):
    """Check if all required model files exist and are valid"""
    print(f"\n{'='*60}")
    print("📁 Checking Model Files")
    print('='*60)
    
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    all_good = True
    for file in required_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    json.load(f)
                print(f"✅ {file} - Valid")
            except Exception as e:
                print(f"❌ {file} - Corrupted: {e}")
                all_good = False
        else:
            print(f"❌ {file} - Missing")
            all_good = False
    
    # Check for model weights
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    if safetensors_files:
        print(f"✅ Found {len(safetensors_files)} safetensors file(s)")
        for sf in safetensors_files:
            file_path = os.path.join(model_path, sf)
            file_size = os.path.getsize(file_path) / (1024**3)  # GB
            print(f"   📦 {sf}: {file_size:.2f} GB")
    else:
        print("❌ No safetensors files found")
        all_good = False
    
    return all_good

def check_safetensors_integrity(model_path):
    """Check if safetensors files can be opened and contain valid tensors"""
    print(f"\n{'='*60}")
    print("🔍 Checking SafeTensors Integrity")
    print('='*60)
    
    safetensors_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') and 'index' not in f]
    
    all_good = True
    for sf in safetensors_files:
        file_path = os.path.join(model_path, sf)
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                print(f"✅ {sf}: {len(keys)} tensors")
                
                # Check a few tensors for NaN or Inf
                sample_keys = keys[:3] if len(keys) > 3 else keys
                for key in sample_keys:
                    tensor = f.get_tensor(key)
                    has_nan = torch.isnan(tensor).any().item()
                    has_inf = torch.isinf(tensor).any().item()
                    
                    if has_nan or has_inf:
                        print(f"   ⚠️  {key}: Contains NaN={has_nan}, Inf={has_inf}")
                        all_good = False
                    else:
                        print(f"   ✅ {key}: Valid tensor {list(tensor.shape)}")
        except Exception as e:
            print(f"❌ {sf}: Error reading - {e}")
            all_good = False
    
    return all_good

def test_model_loading(model_path):
    """Try to load the model and check for errors"""
    print(f"\n{'='*60}")
    print("🔄 Testing Model Loading")
    print('='*60)
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded successfully")
        
        print("\nLoading model (this may take a minute)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("✅ Model loaded successfully")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        
        return True, model, tokenizer
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def test_generation(model, tokenizer):
    """Test if the model can generate coherent text"""
    print(f"\n{'='*60}")
    print("🧪 Testing Text Generation")
    print('='*60)
    
    test_cases = [
        {
            "prompt": "Who are you?",
            "expected_keywords": ["gaia", "mother nature", "ai", "assistant"],
            "check_type": "identity"
        },
        {
            "prompt": "What is 2+2?",
            "expected_keywords": ["4", "four"],
            "check_type": "basic_math"
        },
        {
            "prompt": "The sky is",
            "expected_keywords": ["blue", "color", "atmosphere"],
            "check_type": "completion"
        }
    ]
    
    all_good = True
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test['check_type']} ---")
        print(f"Prompt: {test['prompt']}")
        
        try:
            messages = [{"role": "user", "content": test['prompt']}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            print(f"Response: {response[:200]}...")
            
            # Check for gibberish indicators
            is_gibberish = False
            gibberish_indicators = [
                len(response) < 5,  # Too short
                response.count('�') > 0,  # Invalid characters
                len(set(response.replace(' ', ''))) < 5,  # Too few unique chars
                response.count(response[0]) > len(response) * 0.5 if response else False  # Repetitive
            ]
            
            if any(gibberish_indicators):
                print("❌ GIBBERISH DETECTED - Model appears corrupted")
                all_good = False
            else:
                # Check for expected keywords (case insensitive)
                response_lower = response.lower()
                found_keywords = [kw for kw in test['expected_keywords'] if kw.lower() in response_lower]
                
                if found_keywords:
                    print(f"✅ Coherent response (found: {', '.join(found_keywords)})")
                else:
                    print(f"⚠️  Response seems coherent but unexpected (no keywords: {', '.join(test['expected_keywords'])})")
        
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            all_good = False
    
    return all_good

def validate_model(model_path):
    """Run all validation checks"""
    print("\n" + "="*60)
    print(f"🔍 VALIDATING MODEL: {model_path}")
    print("="*60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return False
    
    # Step 1: Check files
    files_ok = check_model_files(model_path)
    
    # Step 2: Check safetensors integrity
    tensors_ok = check_safetensors_integrity(model_path)
    
    # Step 3: Try loading
    load_ok, model, tokenizer = test_model_loading(model_path)
    
    # Step 4: Test generation
    generation_ok = False
    if load_ok and model and tokenizer:
        generation_ok = test_generation(model, tokenizer)
    
    # Final verdict
    print(f"\n{'='*60}")
    print("📋 VALIDATION SUMMARY")
    print('='*60)
    print(f"Files Check:       {'✅ PASS' if files_ok else '❌ FAIL'}")
    print(f"Tensors Check:     {'✅ PASS' if tensors_ok else '❌ FAIL'}")
    print(f"Loading Check:     {'✅ PASS' if load_ok else '❌ FAIL'}")
    print(f"Generation Check:  {'✅ PASS' if generation_ok else '❌ FAIL'}")
    print('='*60)
    
    if all([files_ok, tensors_ok, load_ok, generation_ok]):
        print("✅ MODEL IS VALID AND WORKING CORRECTLY")
        return True
    else:
        print("❌ MODEL HAS ISSUES - May be corrupted or improperly merged")
        print("\n💡 Suggestions:")
        if not files_ok:
            print("   - Re-download or re-export the model")
        if not tensors_ok:
            print("   - Model weights may be corrupted, try re-merging LoRA")
        if not load_ok:
            print("   - Check transformers/torch versions compatibility")
        if not generation_ok:
            print("   - Model may need re-training or re-merging")
        return False

if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "gaia-merged"
    validate_model(model_path)
