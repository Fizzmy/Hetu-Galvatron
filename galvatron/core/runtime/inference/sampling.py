import torch
from torch import Tensor


def sample(logits: Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9) -> Tensor:
    """Sample next token from logits with temperature, top-k, and top-p filtering.

    Args:
        logits: [batch_size, vocab_size] unnormalized logits.
        temperature: Sampling temperature. 0 means greedy.
        top_k: Keep only top-k tokens. 0 disables.
        top_p: Nucleus sampling threshold. 1.0 disables.

    Returns:
        [batch_size] sampled token ids.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        threshold = torch.topk(logits, top_k, dim=-1)[0][..., -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_mask.scatter(dim=-1, index=sorted_indices, src=sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
