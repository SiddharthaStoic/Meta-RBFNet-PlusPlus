import torch
import torch.nn as nn
import torch.nn.functional as F

class MetaRBFNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, rbf_dim=16, num_classes=10):
        super().__init__()
        self.rbf_dim = rbf_dim
        self.input_dim = input_dim

        # Meta-network for generating RBF centers (μ): outputs (rbf_dim × input_dim)
        self.meta_mu = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rbf_dim * input_dim)  # reshaped later
        )

        # Meta-network for generating RBF spreads (σ)
        self.meta_sigma = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, rbf_dim)
        )

        self.value_weights = nn.Parameter(torch.randn(rbf_dim, num_classes))
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(num_classes),
            nn.ReLU(),
            nn.Linear(num_classes, num_classes)
        )

    def forward(self, x):
        batch_size, input_dim = x.size()

        # μ: learn (rbf_dim, input_dim) per sample
        mu = self.meta_mu(x).view(batch_size, self.rbf_dim, input_dim)  # shape: (batch, rbf_dim, input_dim)

        # σ: learn (rbf_dim) per sample
        sigma = torch.exp(self.meta_sigma(x)).unsqueeze(2)  # shape: (batch, rbf_dim, 1)

        # Expand input
        x_exp = x.unsqueeze(1).expand(-1, self.rbf_dim, -1)  # shape: (batch, rbf_dim, input_dim)

        # RBF distance and activation
        dists = ((x_exp - mu) ** 2).sum(dim=2) / (2 * sigma.squeeze(-1) ** 2 + 1e-8)
        rbf_activations = torch.exp(-dists)

        # 🔥 Entropy-aware adaptive-k gating
        from utils import compute_entropy, adaptive_k_mask
        entropy = compute_entropy(rbf_activations, dim=1)
        gated_activations = adaptive_k_mask(rbf_activations, entropy, min_k=2, max_k=self.rbf_dim)

        attention = F.softmax(gated_activations, dim=1)  # shape: (batch, rbf_dim)
        v = attention @ self.value_weights  # (batch, num_classes)
        out = self.out_mlp(v)

        return out, attention, mu, sigma, entropy