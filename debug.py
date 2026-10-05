import os
import pandas as pd

IMG_DIR = "dataset/sign_data"
df = pd.read_csv("dataset/train_data.csv", header=None)

# check first 5 paths
for i in range(5):
    path = os.path.join(IMG_DIR, str(df.iloc[i, 0]).replace("/", os.sep))
    print(f"Path: {path}")
    print(f"Exists: {os.path.exists(path)}")
    print()