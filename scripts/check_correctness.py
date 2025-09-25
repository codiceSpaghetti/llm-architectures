import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_architectures.models.gemma3.args import gemma3_configs
from llm_architectures.models.gemma3.model import Gemma3
from llm_architectures.models.gemma3.state_dict_adapter import Gemma3Converter

DEFAULT_CHECKPOINTS = {"gemma3": "google/gemma-3-4b-it"}


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_tokens: int = 20):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    for _ in range(max_tokens):
        logits = model(input_ids).logits if hasattr(model(input_ids), "logits") else model(input_ids)
        next_token = torch.argmax(logits[0, -1, :]).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        choices=["gemma3"],
        help=f"Model to check. Currently available models are: {list(DEFAULT_CHECKPOINTS.keys())}",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate from (default: Knowledge gained is potential, knowledge applied is)",
        default="Knowledge gained is potential, knowledge applied is",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=10, help="Maximum number of new tokens to generate (default: 10)"
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint or DEFAULT_CHECKPOINTS[args.model_name]
    prompt = args.prompt

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16, trust_remote_code=True)

    config = gemma3_configs[checkpoint]
    custom_model = Gemma3(config)
    converter = Gemma3Converter(config)
    custom_model.load_state_dict(converter.from_hf(hf_model.state_dict()), strict=False)
    custom_model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        hf_logits = hf_model(**inputs).logits[0, -1, :].cpu()
        custom_logits = custom_model(inputs["input_ids"])[0, -1, :].cpu()

    cosine_sim = F.cosine_similarity(hf_logits.unsqueeze(0), custom_logits.unsqueeze(0))
    print(f"Cosine similarity: {cosine_sim.item():.6f}")

    hf_output = generate(hf_model, tokenizer, prompt, args.max_new_tokens)
    custom_output = generate(custom_model, tokenizer, prompt, args.max_new_tokens)

    print(f"HF:     {hf_output}")
    print(f"Custom: {custom_output}")
    print(f"Match: {hf_output == custom_output}")


if __name__ == "__main__":
    main()
