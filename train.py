from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import MetaRBFNet
from visualize import plot_centers, plot_spreads, plot_activations
from utils import get_adaptive_k_values
import matplotlib.pyplot as plt

# Load data
X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train))
test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test))

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64)

model = MetaRBFNet(input_dim=X.shape[1], num_classes=len(set(y)))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(20):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds, _, _, _, _ = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

from visualize import plot_centers, plot_spreads, plot_activations

xb, yb = next(iter(test_loader))
_, activations, mu, sigma, entropy = model(xb)

# Get adaptive-k values from entropy
adaptive_k = get_adaptive_k_values(entropy, min_k=2, max_k=model.rbf_dim)

# Convert to numpy
entropy_vals = entropy.detach().cpu().numpy()
k_vals = adaptive_k.detach().cpu().numpy()

# Plot Entropy vs Adaptive-k
plt.figure(figsize=(6, 4))
plt.scatter(entropy_vals, k_vals, c='purple', alpha=0.7)
plt.xlabel("Entropy of RBF Activations")
plt.ylabel("Adaptive k (Active Neurons)")
plt.title("Entropy vs Adaptive-k Mapping")
plt.grid(True)
plt.tight_layout()
plt.show()

plot_centers(mu)
plot_spreads(sigma)
plot_activations(activations)