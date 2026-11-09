
import torch
from model import ECGNet
from prepare import X_test_tensor, y_test_tensor

# 1. Load the saved model
model = ECGNet()

model.load_state_dict(
    torch.load("results/best_model.pth", map_location="cpu")
)

model.eval()

# 2. Predict classes for all test rows
with torch.no_grad():
    logits = model(X_test_tensor)
    predictions = torch.argmax(logits, dim=1)

# 3. Calculate accuracy manually
correct = (predictions == y_test_tensor).sum().item()
total = len(y_test_tensor)

accuracy = correct / total

# 4. Calculate 5 x 5 confusion matrix manually
num_classes = 5

confusion_matrix = torch.zeros(
    (num_classes, num_classes), dtype=torch.int64
)

for actual, predicted in zip(y_test_tensor, predictions):
    confusion_matrix[actual.item(), predicted.item()] += 1

# 5. Calculate per-class accuracy
per_class_accuracy = []

for i in range(num_classes):
    correct_class = confusion_matrix[i, i].item()
    total_class = confusion_matrix[i].sum().item()

    class_accuracy = correct_class / total_class
    per_class_accuracy.append(class_accuracy)

# 6. Calculate balanced accuracy
balanced_accuracy = sum(per_class_accuracy) / num_classes

# 7. Display results
print("\n========== TASK 5 RESULTS ==========")

print("\nTotal test samples:", total)
print("Correct predictions:", correct)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix.numpy())

class_names = ["N", "S", "V", "F", "Q"]

print("\nPer-Class Accuracy:")

for i in range(num_classes):
    print(
        f"Class {class_names[i]}: "
        f"{per_class_accuracy[i] * 100:.2f}%"
    )

print(
    f"\nBalanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)

print("\n========== TESTING COMPLETED ==========")
