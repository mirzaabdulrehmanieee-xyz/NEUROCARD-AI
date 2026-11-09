
import torch
import torch.nn.functional as F

from model import ECGNet
from prepare import X_train, y_train
from loss import cross_entropy


# --------------------------------------------------
# 1. Set random seed BEFORE creating the network
# --------------------------------------------------

torch.manual_seed(0)

# --------------------------------------------------
# 2. Create neural network
# --------------------------------------------------

model = ECGNet()

# --------------------------------------------------
# 3. Select first 64 training rows
# --------------------------------------------------

X = torch.tensor(
    X_train.iloc[:64].to_numpy(),
    dtype=torch.float32
)

y = torch.tensor(
    y_train.iloc[:64].to_numpy(),
    dtype=torch.long
)

B = X.shape[0]

# --------------------------------------------------
# 4. Forward pass
# --------------------------------------------------

logits = model(X)

# --------------------------------------------------
# 5. Calculate cross-entropy loss
# --------------------------------------------------

loss = cross_entropy(logits, y)

# --------------------------------------------------
# 6. Backpropagation using PyTorch
# --------------------------------------------------

model.zero_grad()

loss.backward()

# --------------------------------------------------
# 7. Calculate softmax probabilities manually
# --------------------------------------------------

Y_hat = torch.softmax(logits.detach(), dim=1)

# --------------------------------------------------
# 8. Convert actual labels into one-hot encoding
# --------------------------------------------------

Y = F.one_hot(y, num_classes=5).float()

# --------------------------------------------------
# 9. Manually calculate output-layer gradients
# --------------------------------------------------

manual_dW3 = ((Y_hat - Y).T @ model.a2.detach()) / B

manual_db3 = (Y_hat - Y).sum(dim=0) / B

# --------------------------------------------------
# 10. Compare manual gradients with PyTorch gradients
# --------------------------------------------------

difference_W3 = torch.max(
    torch.abs(manual_dW3 - model.W3.grad)
).item()

difference_b3 = torch.max(
    torch.abs(manual_db3 - model.b3.grad)
).item()

# --------------------------------------------------
# 11. Display results
# --------------------------------------------------

print("\n========== GRADIENT CHECK ==========")

print("Batch size:", B)
print("Input shape:", X.shape)
print("Logits shape:", logits.shape)

print("\nCross Entropy Loss:", loss.item())

print("\nManual Gradient Shapes:")
print("dW3:", manual_dW3.shape)
print("db3:", manual_db3.shape)

print("\nPyTorch Gradient Shapes:")
print("W3.grad:", model.W3.grad.shape)
print("b3.grad:", model.b3.grad.shape)

print("\nMaximum Absolute Differences:")
print("W3 gradient difference:", difference_W3)
print("b3 gradient difference:", difference_b3)

print("\nGradient Check Completed!")
