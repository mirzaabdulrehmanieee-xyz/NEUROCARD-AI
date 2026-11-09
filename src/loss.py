
import torch


def cross_entropy(logits, y):
    """
    Calculate mean softmax cross-entropy loss.

    logits: (B, 5) - Raw network outputs
    y:      (B,)   - True class labels
    """

    # Calculate log-sum-exp for each sample
    log_sum_exp = torch.logsumexp(logits, dim=1)

    # Get the correct class logits
    correct_logits = logits[
        torch.arange(logits.shape[0]),
        y
    ]

    # Calculate individual losses
    losses = log_sum_exp - correct_logits

    # Return mean loss over the batch
    return losses.mean()


if __name__ == "__main__":

    # Sample logits for 3 heartbeat examples
    logits = torch.tensor([
        [2.0, 1.0, 0.5, 0.2, 0.1],
        [0.1, 0.2, 2.5, 0.3, 0.4],
        [0.5, 0.4, 0.3, 2.0, 0.1]
    ])

    # Actual heartbeat classes
    y = torch.tensor([0, 2, 3])

    # Calculate cross entropy
    loss = cross_entropy(logits, y)

    print("Logits shape:", logits.shape)
    print("Labels shape:", y.shape)
    print("Cross Entropy Loss:", loss.item())
