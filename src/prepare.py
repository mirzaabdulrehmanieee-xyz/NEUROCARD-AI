import numpy as np
import torch
from load_data import load_raw
# Load original datasets
raw = load_raw()
# Separate training data
X_train = raw["train"].iloc[:, :187]
y_train = raw["train"].iloc[:, 187]
# Separate testing data
X_test = raw["test"].iloc[:, :187]
y_test = raw["test"].iloc[:, 187]
# Check original label data types
print("Original training label dtype:", y_train.dtype)
print("Original testing label dtype:", y_test.dtype)
# Convert labels into integers (0-4)
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
# Verify conversion
print("\nAfter conversion:")
print("Training label dtype:", y_train.dtype)
print("Testing label dtype:", y_test.dtype)

# Check unique classes
print("\nTraining classes:", np.unique(y_train))
print("Testing classes:", np.unique(y_test))

# Check dimensions
print("\nDataset dimensions:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
# Task 1.3: Minimum and maximum signal values
print("\nSignal value analysis:")
print("Minimum training signal value:", X_train.min().min())
print("Maximum training signal value:", X_train.max().max())

# Task 1.4: Class distribution

class_names = ["N", "S", "V", "F", "Q"]

train_counts = np.bincount(y_train, minlength=5)
test_counts = np.bincount(y_test, minlength=5)

print("\nTraining class distribution:")
for i in range(5):
    print(f"{class_names[i]}: {train_counts[i]}")

print("\nTesting class distribution:")
for i in range(5):
    print(f"{class_names[i]}: {test_counts[i]}")

# Task 1.6: Train and validation split

rng = np.random.default_rng(42)

idx = rng.permutation(len(raw["train"]))

n_val = int(0.1 * len(raw["train"]))

val_idx, train_idx = idx[:n_val], idx[n_val:]

# Split training signals
X_train_split = X_train.iloc[train_idx]
X_val = X_train.iloc[val_idx]

# Split training labels
y_train_split = y_train.iloc[train_idx]
y_val = y_train.iloc[val_idx]

print("\nTrain/Validation/Test split:")
print("Training samples:", len(X_train_split))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))

print("\nFeature dimensions:")
print("X_train:", X_train_split.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)

print("\nLabel dimensions:")
print("y_train:", y_train_split.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

# Task 1.7: Convert data into PyTorch tensors

X_train_tensor = torch.tensor(
    X_train_split.to_numpy(),
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_split.to_numpy(),
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val.to_numpy(),
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val.to_numpy(),
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test.to_numpy(),
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.long
)

print("\nPyTorch Tensor Verification:")

print("X_train:", X_train_tensor.shape, X_train_tensor.dtype)
print("y_train:", y_train_tensor.shape, y_train_tensor.dtype)

print("X_val:", X_val_tensor.shape, X_val_tensor.dtype)
print("y_val:", y_val_tensor.shape, y_val_tensor.dtype)

print("X_test:", X_test_tensor.shape, X_test_tensor.dtype)
print("y_test:", y_test_tensor.shape, y_test_tensor.dtype)
