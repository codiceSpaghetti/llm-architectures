import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name, cache_dir="./models"):
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    print(f"Downloaded to {cache_dir}")
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="HuggingFace model name")
    parser.add_argument("--cache_dir", default="./models", help="Cache directory")
    args = parser.parse_args()
    
    download_model(args.model_name, args.cache_dir)
