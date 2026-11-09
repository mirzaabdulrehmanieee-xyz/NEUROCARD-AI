
import matplotlib.pyplot as plt

from train import train


# Learning rates specified in assignment
learning_rates = [0.01, 0.1, 0.5]

results = {}

print("\n========== LEARNING RATE COMPARISON ==========")

# Train a fresh network for each learning rate
for alpha in learning_rates:

    print(f"\n\nTraining with learning rate: {alpha}")

    model, history = train(alpha)

    results[alpha] = history


# --------------------------------------------------
# Find learning rate with lowest validation loss
# --------------------------------------------------

best_alpha = min(
    learning_rates,
    key=lambda a: min(results[a]["val_loss"])
)

best_loss = min(results[best_alpha]["val_loss"])

best_epoch = results[best_alpha]["val_loss"].index(best_loss) + 1


print("\n========== COMPARISON RESULTS ==========")

for alpha in learning_rates:

    minimum_loss = min(results[alpha]["val_loss"])

    epoch = results[alpha]["val_loss"].index(minimum_loss) + 1

    print(
        f"Learning rate: {alpha} | "
        f"Lowest validation loss: {minimum_loss:.6f} | "
        f"Epoch: {epoch}"
    )

print("\nBest learning rate:", best_alpha)
print("Lowest validation loss:", best_loss)
print("Best epoch:", best_epoch)


# --------------------------------------------------
# Plot validation loss for all learning rates
# --------------------------------------------------

epochs = range(1, 31)

plt.figure(figsize=(10, 6))

for alpha in learning_rates:

    plt.plot(
        epochs,
        results[alpha]["val_loss"],
        label=f"Learning rate = {alpha}"
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")

plt.title("Validation Loss Comparison for Different Learning Rates")

plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/learning_rate_comparison.png",
    dpi=300
)

plt.show()

print("\nGraph saved in results/learning_rate_comparison.png")
