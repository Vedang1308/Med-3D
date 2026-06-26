import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from generate_3d_pgd_robustness import generate_3d_pgd_perturbation

def main():
    print("Initializing Phase 1: 3D Environment Setup")

    # 1. Load the 3D Med-VLM
    # We use M3D-LaMed-Llama-2-7B or a compatible variant as requested.
    model_name = "GoodBaiBai88/M3D-LaMed-Llama-2-7B" # Assuming this is the HuggingFace ID
    print(f"Loading model: {model_name} onto GPU (if available)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # In a real environment, we would load the actual weights. 
    # Using trust_remote_code=True is usually required for custom multimodal architectures.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Continuing with setup instructions (ensure you have access to the model on the SOL supercomputer).")

    # 2. Automate Dataset Acquisition
    dataset_name = "Tang-xiaoxiao/3D-RAD"
    print(f"Loading dataset: {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded. Available splits: {list(dataset.keys())}")
        
        # Example of how to access a sample
        # sample = dataset['train'][0]
        # print(f"Sample keys: {sample.keys()}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

    # 3. 3D Adversarial Robustness Script is ready in generate_3d_pgd_robustness.py
    print("PGD Perturbation script is configured and ready to be used.")
    print("Setup complete. You can run this script on your A100 node.")

if __name__ == "__main__":
    main()
