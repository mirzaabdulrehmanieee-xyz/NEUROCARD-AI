
import torch
import torch.nn as nn
import math


class ECGNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Layer 1: 187 inputs -> 128 neurons
        self.W1 = nn.Parameter(
            torch.randn(128, 187) * math.sqrt(2 / 187)
        )
        self.b1 = nn.Parameter(torch.zeros(128))

        # Layer 2: 128 inputs -> 64 neurons
        self.W2 = nn.Parameter(
            torch.randn(64, 128) * math.sqrt(2 / 128)
        )
        self.b2 = nn.Parameter(torch.zeros(64))

        # Layer 3: 64 inputs -> 5 output logits
        self.W3 = nn.Parameter(
            torch.randn(5, 64) * math.sqrt(2 / 64)
        )
        self.b3 = nn.Parameter(torch.zeros(5))

    def forward(self, x):

        # First hidden layer
        z1 = x @ self.W1.T + self.b1
        a1 = torch.clamp(z1, min=0)

        # Second hidden layer
        z2 = a1 @ self.W2.T + self.b2
        a2 = torch.clamp(z2, min=0)

        # Store second hidden layer activation
        # Required for manual gradient calculation
        self.a2 = a2

        # Output layer (raw logits, no activation)
        z3 = a2 @ self.W3.T + self.b3

        return z3


if __name__ == "__main__":

    # Set random seed before creating network
    torch.manual_seed(0)

    # Create model
    model = ECGNet()

    # Test mini-batch
    x = torch.randn(32, 187)

    # Forward pass
    logits = model(x)

    print("Output shape:", logits.shape)

    # Count trainable parameters
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("Total trainable parameters:", total_params)

    # Verify parameter shapes
    print("\nParameter Shapes:")

    for name, param in model.named_parameters():
        print(name, param.shape)

    # Verify bias initialization
    print("\nBias Initialization:")
    print("b1 all zeros:", torch.all(model.b1 == 0).item())
    print("b2 all zeros:", torch.all(model.b2 == 0).item())
    print("b3 all zeros:", torch.all(model.b3 == 0).item())

    # Verify trainability
    print(
        "\nAll parameters trainable:",
        all(p.requires_grad for p in model.parameters())
    )

    # Verify second hidden layer activation
    print("A2 shape:", model.a2.shape)
