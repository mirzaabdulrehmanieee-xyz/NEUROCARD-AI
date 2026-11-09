import os
import torch
import matplotlib.pyplot as plt

from train import train

# Best learning rate selected from comparison
BEST_ALPHA = 0.5

# Create results folder if it does not exist
os.makedirs("results", exist_ok=True)

# Train final model
model, history = train(BEST_ALPHA)

# Save best model
torch.save(model.state_dict(), "results/best_model.pth")

print("\n========== FINAL MODEL ==========")
print("Learning Rate:", BEST_ALPHA)
print("Best Epoch:", history["best_epoch"])
print("Best Validation Loss:", history["best_val_loss"])

# Plot training and validation loss
plt.figure(figsize=(9, 5))
plt.plot(history["train_loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/final_loss.png")
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(9, 5))
plt.plot(history["train_accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("results/final_accuracy.png")
plt.show()

print("\nModel and graphs saved successfully!")
