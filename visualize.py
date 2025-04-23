import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

def plot_centers(mu_tensor, title="Learned RBF Centers (2D)"):
    """Project learned RBF centers to 2D and plot (after flattening across batch and rbf)."""
    mu = mu_tensor.detach().cpu().numpy()

    if mu.ndim == 3:  # Shape: (batch, rbf_dim, input_dim)
        mu = mu.reshape(-1, mu.shape[-1])  # Flatten to (batch * rbf_dim, input_dim)

    pca = PCA(n_components=2)
    mu_2d = pca.fit_transform(mu)

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=mu_2d[:, 0], y=mu_2d[:, 1])
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()

def plot_spreads(sigma_tensor, title="Spread Distribution"):
    """Histogram of learned RBF spreads."""
    sigma = sigma_tensor.detach().cpu().numpy().flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(sigma, bins=20, alpha=0.7)
    plt.title(title)
    plt.xlabel("Spread (σ)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_activations(activations, title="RBF Activations"):
    """Heatmap of RBF activations for a mini-batch."""
    act = activations.detach().cpu().numpy()
    plt.figure(figsize=(8, 5))
    sns.heatmap(act, cmap="viridis")
    plt.title(title)
    plt.xlabel("RBF Neuron Index")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.show()