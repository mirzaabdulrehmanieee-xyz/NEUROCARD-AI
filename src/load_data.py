
from pathlib import Path
import pandas as pd
FILES = {
    "train": "mitbih_train.csv",
    "test": "mitbih_test.csv"}

def load_raw(data_dir="data"):
    """
    Load training and testing ECG datasets
    without modifying the original data.
    """
    data_dir = Path(data_dir)
    frames = {}
    for split, name in FILES.items():
        path = data_dir / name

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Download the dataset first.")

        frames[split] = pd.read_csv(path, header=None)

    return frames

if __name__ == "__main__":
    raw = load_raw()

    for split, df in raw.items():
        print(
            f"{split}: {df.shape[0]} rows, "
            f"{df.shape[1]} columns"
        )

    print(
        raw["train"].iloc[
            :3, [0, 1, 2, 185, 186, 187]
        ].to_string())
