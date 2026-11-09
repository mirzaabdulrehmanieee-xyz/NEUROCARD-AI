
import torch
import copy

from model import ECGNet
from loss import cross_entropy

from prepare import (
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor
)


# --------------------------------------------------
# Training settings from assignment
# --------------------------------------------------

BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
EPOCHS = 30


# --------------------------------------------------
# Evaluate model on a dataset
# --------------------------------------------------

def evaluate(model, X, y):

    model.eval()

    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():

        for start in range(0, len(X), 1024):

            end = start + 1024

            xb = X[start:end]
            yb = y[start:end]

            logits = model(xb)

            loss = cross_entropy(logits, yb)

            total_loss += loss.item() * len(xb)

            predictions = logits.argmax(dim=1)

            total_correct += (predictions == yb).sum().item()

    average_loss = total_loss / len(X)

    accuracy = total_correct / len(X)

    return average_loss, accuracy


# --------------------------------------------------
# Task 4.1: Custom training function
# --------------------------------------------------

def train(alpha):

    # Set seed before creating a fresh network
    torch.manual_seed(0)

    model = ECGNet()

    # Initialize velocity for every parameter
    velocity = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
    }

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": []
    }

    # Track best validation model
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    # --------------------------------------------------
    # Epoch loop
    # --------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        # Shuffle training data every epoch
        indices = torch.randperm(len(X_train_tensor))

        # Mini-batch training
        for start in range(0, len(indices), BATCH_SIZE):

            batch_indices = indices[start:start + BATCH_SIZE]

            xb = X_train_tensor[batch_indices]
            yb = y_train_tensor[batch_indices]

            # Clear previous gradients
            model.zero_grad(set_to_none=True)

            # Forward pass
            logits = model(xb)

            # Calculate loss
            loss = cross_entropy(logits, yb)

            # Backward pass
            loss.backward()

            # Custom momentum update
            with torch.no_grad():

                for name, param in model.named_parameters():

                    # Obtain gradient
                    g = param.grad

                    # Weight decay for weights only
                    if name.startswith("W"):

                        g = g + WEIGHT_DECAY * param

                    # Update velocity
                    velocity[name].mul_(MOMENTUM).add_(
                        g,
                        alpha=(1 - MOMENTUM)
                    )

                    # Update parameter
                    param.add_(velocity[name], alpha=-alpha)

        # --------------------------------------------------
        # Calculate metrics after each epoch
        # --------------------------------------------------

        train_loss, train_acc = evaluate(
            model,
            X_train_tensor,
            y_train_tensor
        )

        val_loss, val_acc = evaluate(
            model,
            X_val_tensor,
            y_val_tensor
        )

        # Record epoch results
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_accuracy"].append(train_acc)
        history["val_accuracy"].append(val_acc)

        # --------------------------------------------------
        # Save best validation model
        # --------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

        # --------------------------------------------------
        # Print epoch results
        # --------------------------------------------------

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    # --------------------------------------------------
    # Restore best model
    # --------------------------------------------------

    model.load_state_dict(best_state)

    history["best_epoch"] = best_epoch

    history["best_val_loss"] = best_val_loss

    return model, history


# --------------------------------------------------
# Test Task 4.1
# --------------------------------------------------

if __name__ == "__main__":

    learning_rate = 0.01

    model, history = train(learning_rate)

    print("\n========== TRAINING COMPLETED ==========")

    print("Learning rate:", learning_rate)

    print("Best epoch:", history["best_epoch"])

    print(
        "Best validation loss:",
        history["best_val_loss"]
    )

    print(
        "Final training loss:",
        history["train_loss"][-1]
    )

    print(
        "Final validation loss:",
        history["val_loss"][-1]
    )

    print(
        "Final training accuracy:",
        history["train_accuracy"][-1]
    )

    print(
        "Final validation accuracy:",
        history["val_accuracy"][-1]
    )
