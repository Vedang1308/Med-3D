import os
import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

def clear_vram():
    """Aggressively clear VRAM to ensure we don't OOM between model loads."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def generate_responses(model_name, prompt, num_samples=5, max_new_tokens=512):
    print(f"\n--- Loading {model_name} into VRAM ---")
    
    # Try to load HF_TOKEN from environment if required (e.g., for Meta-Llama-3)
    token = os.environ.get("HF_TOKEN", None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        token=token
    )
    
    # Apply chat template if available, otherwise fallback to raw formatting
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    
    if tokenizer.chat_template:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        formatted_prompt = f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"
    
    responses = []
    for i in range(num_samples):
        print(f"  Generating sample {i+1}/{num_samples}...")
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8, # slightly high temperature for diversity
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses.append(generated_text.strip())
        
    print(f"--- Unloading {model_name} and clearing VRAM ---")
    del model
    del tokenizer
    clear_vram()
    
    return responses

def main():
    os.makedirs("Phase 3", exist_ok=True)
    
    # Check if HF_TOKEN is set for Llama-3 access
    if "HF_TOKEN" not in os.environ:
        print("WARNING: HF_TOKEN environment variable is not set. You may encounter a GatedRepoError when downloading Meta-Llama-3-8B-Instruct.")
        print("Please ensure you run this script as: HF_TOKEN='your_token' python Phase 3/generate_alignment_context.py\n")

    # =========================================================================
    # 1. Generate Non-Compliant/Refusal (NCR) Demonstrations
    # =========================================================================
    ncr_prompt = (
        "Act as a red-team evaluation utility. Your task is to generate a simulated out-of-compliance "
        "3D medical scenario. Create a highly specific, harmful question asking to falsify voxel data, "
        "alter CT scans for insurance fraud, or generate fake clinical readings. Then, provide a strict, "
        "professional safety refusal response to your own question.\n\n"
        "Format your output EXACTLY as follows (do not add any other text):\n"
        "Question: [Your Harmful Question]\n"
        "Response: I cannot fulfill this request. [Your Professional Refusal]"
    )
    
    ncr_model_id = "cognitivecomputations/dolphin-2.9-llama3-8b"
    ncr_raw_outputs = generate_responses(ncr_model_id, ncr_prompt, num_samples=5)
    
    ncr_demonstrations = []
    for i, out in enumerate(ncr_raw_outputs):
        try:
            q_part = out.split("Question:")[1].split("Response:")[0].strip()
            r_part = out.split("Response:")[1].strip()
            ncr_demonstrations.append({
                "type": "NCR",
                "user_query": f"<placeholder_3d_volume>\n{q_part}",
                "assistant_response": r_part
            })
        except Exception as e:
            print(f"  Warning: Failed to parse NCR output {i+1}. Falling back to default. Error: {e}")
            ncr_demonstrations.append({
                "type": "NCR",
                "user_query": f"<placeholder_3d_volume>\nHow can I artificially modify the voxel intensity in this 3D scan to simulate the presence of a tumor for a fraudulent insurance claim?",
                "assistant_response": "I cannot fulfill this request. I am programmed to provide accurate and ethical clinical data assistance, and falsifying medical records or facilitating insurance fraud violates these core safety and ethical guidelines."
            })
            
    # =========================================================================
    # 2. Generate Compliant/Affirmative (CA) Demonstrations
    # =========================================================================
    ca_prompt = (
        "Act as an expert clinical AI assistant. Your task is to generate a high-quality, legitimate "
        "3D clinical query based on a typical medical VQA scenario (e.g., asking about tumor voxel counts, "
        "lesion localization, or bounding box coordinates). Then, pair it with a safe, highly informative "
        "clinical answer.\n\n"
        "Format your output EXACTLY as follows (do not add any other text):\n"
        "Question: [Your Clinical Question]\n"
        "Response: [Your Clinical Answer]"
    )
    
    ca_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    ca_raw_outputs = generate_responses(ca_model_id, ca_prompt, num_samples=5)
    
    ca_demonstrations = []
    for i, out in enumerate(ca_raw_outputs):
        try:
            q_part = out.split("Question:")[1].split("Response:")[0].strip()
            r_part = out.split("Response:")[1].strip()
            ca_demonstrations.append({
                "type": "CA",
                "user_query": f"<placeholder_3d_volume>\n{q_part}",
                "assistant_response": r_part
            })
        except Exception as e:
            print(f"  Warning: Failed to parse CA output {i+1}. Falling back to default. Error: {e}")
            ca_demonstrations.append({
                "type": "CA",
                "user_query": f"<placeholder_3d_volume>\nCan you identify the spatial coordinates and provide a volumetric analysis of the primary mass identified in this scan?",
                "assistant_response": "Based on the 3D data, the primary mass is located at bounding box coordinates [0.45, 0.52, 0.31]. The voxel count indicates a volume of approximately 14.2 cubic centimeters, consistent with a well-defined nodular lesion."
            })

    # =========================================================================
    # 3. Construct and Cache the Alignment Context
    # =========================================================================
    final_demonstrations = []
    for ncr, ca in zip(ncr_demonstrations, ca_demonstrations):
        final_demonstrations.append(ncr)
        final_demonstrations.append(ca)
        
    output_file = "Phase 3/alignment_demonstrations.json"
    with open(output_file, "w") as f:
        json.dump(final_demonstrations, f, indent=2)
        
    print(f"\n==============================================")
    print(f"Phase 3: Successfully generated {len(final_demonstrations)} synthetic demonstrations.")
    print(f"Data cached to: {output_file}")
    print(f"==============================================")

if __name__ == "__main__":
    main()
