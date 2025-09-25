# Modern LLM Architectures
*Clean, Educational PyTorch Implementations of State-of-the-Art Language Models*

## Project Vision

This project provides clean, readable PyTorch implementations of modern large language model architectures with a focus on **learning and understanding**. Inspired by Sebastian Raschka's excellent blog post ["The Big LLM Architecture Comparison"](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison), I aim to create a hands-on resource that makes cutting-edge architectures accessible to researchers, students, and practitioners.

## Why This Project?

- **Readability First**: Every implementation prioritizes clarity over performance optimization
- **Verified Correctness**: All implementations tested for numerical equivalence with HuggingFace models
- **Modern Architectures**: Coverage of the latest innovations in transformer design
- **Easy Navigation**: Consistent structure across all model implementations

## Verification

Test implementations against HuggingFace models:

```bash
uv run python -m scripts.check_correctness gemma3
```

## Supported Architectures

### Planned Models (Roadmap)

| Model | Status | Key Innovations | Implementation |
|-------|--------|----------------|---------------|
| **Gemma 3** | Done | local/global attn, Normalizer location, GeGLU | [code](llm_architectures/models/gemma3/model.py) |
| **Qwen 3** | In Progress | Dual chunk attention | Not yet implemented |
| **Llama 4** | Planned | RMSNorm, SwiGLU, RoPE | Not yet implemented |
| **DeepSeek V3/R1** | Planned | Multi-head latent attention, MoE | Not yet implemented |
| **OLMo 2** | Planned | Academic transparency focus | Not yet implemented |
| **Mistral Small 3.1** | Planned | Sliding window attention | Not yet implemented |
| **Qwen 3-Next** | Planned | Next-generation improvements | Not yet implemented |
| **SmolLM 3** | Planned | Efficiency optimizations | Not yet implemented |
| **Kimi 2** | Planned | Long context handling | Not yet implemented |
| **GPT-OSS** | Planned | Open-source GPT variant | Not yet implemented |
| **Grok 2.5** | Planned | xAI innovations | Not yet implemented |
| **GLM-4.5** | Planned | ChatGLM improvements | Not yet implemented |
