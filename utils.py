import torch
import torch.nn.functional as F

def compute_entropy(tensor, dim=1):
    """Shannon entropy along a given dimension."""
    probs = tensor / (tensor.sum(dim=dim, keepdim=True) + 1e-8)
    entropy = -probs * torch.log(probs + 1e-8)
    return entropy.sum(dim=dim)

def topk_mask(tensor, k):
    """Return a mask that keeps top-k elements per row and zeros out the rest."""
    topk_values, topk_indices = torch.topk(tensor, k, dim=1)
    mask = torch.zeros_like(tensor)
    mask.scatter_(1, topk_indices, 1.0)
    return mask

def apply_entropy_gating(rbf_activations, k=None):
    """Gate activations using either top-k or entropy-based score."""
    if k is not None:
        mask = topk_mask(rbf_activations, k)
        gated = rbf_activations * mask
        return gated
    else:
        # fallback (identity)
        return rbf_activations

def compute_accuracy(preds, labels):
    """Calculate classification accuracy."""
    return (preds.argmax(dim=1) == labels).float().mean().item()

def adaptive_k_mask(tensor, entropy, min_k=2, max_k=10):
    """
    Generate a per-sample top-k mask based on entropy score.
    tensor: (batch, rbf_dim)
    entropy: (batch,)
    Returns: gated_tensor (same shape), mask (optional)
    """
    batch_size, rbf_dim = tensor.shape
    max_entropy = entropy.max().item() + 1e-8

    # Normalize entropy to [0, 1], then scale to [min_k, max_k]
    scaled_k = ((entropy / max_entropy) * (max_k - min_k)) + min_k
    scaled_k = scaled_k.int().clamp(min_k, max_k)  # shape: (batch,)

    gated = torch.zeros_like(tensor)
    for i in range(batch_size):
        topk = scaled_k[i].item()
        top_vals, top_idx = torch.topk(tensor[i], topk)
        gated[i, top_idx] = tensor[i, top_idx]

    return gated

def get_adaptive_k_values(entropy, min_k=2, max_k=10):
    """
    Converts entropy vector to adaptive k values (per sample).
    entropy: (batch,)
    Returns: scaled_k (batch,) tensor of ints
    """
    max_entropy = entropy.max().item() + 1e-8
    scaled_k = ((entropy / max_entropy) * (max_k - min_k)) + min_k
    return scaled_k.int().clamp(min_k, max_k)