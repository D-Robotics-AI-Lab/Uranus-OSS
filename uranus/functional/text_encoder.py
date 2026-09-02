"""
Functional API for encoding text prompts with T5 text encoder.
"""

import torch
from uranus.modules.t5_encoder import WanTextEncoder, HuggingfaceTokenizer


def encode_prompt(
    tokenizer: HuggingfaceTokenizer,
    text_encoder: WanTextEncoder,
    prompt: str,
    device: torch.device,
    seq_len: int = 512,
) -> torch.Tensor:
    """
    Encode a text prompt into T5 embeddings.

    Args:
        tokenizer: HuggingfaceTokenizer instance
        text_encoder: WanTextEncoder (T5-XXL) instance
        prompt: Input text prompt
        device: Target device
        seq_len: Max sequence length for tokenization

    Returns:
        text_embeddings of shape [1, seq_len, 4096]
    """
    ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(device)
    mask = mask.to(device)
    prompt_emb = text_encoder(ids, mask)
    return prompt_emb * mask.unsqueeze(-1)
