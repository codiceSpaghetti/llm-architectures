import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name):
    
    print(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        trust_remote_code=True
    )
    
    return model, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", help="HuggingFace model name")
    args = parser.parse_args()
    
    download_model(args.model_name)
