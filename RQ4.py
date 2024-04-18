import os
import argparse
import pandas as pd
from tqdm import tqdm

from Model import Model

SEED = 42

parser = argparse.ArgumentParser(description="RQ 2 (Cross-architecture detection).")
parser.add_argument("--input", "-i", nargs="?", help="Target architecture.")
args = parser.parse_args()

collection = pd.read_csv("data/IoT_Malware_Collection_2021.csv")
collection = pd.concat(
    [collection, pd.read_csv("data/IoT_Benign_Collection_2021.csv")], ignore_index=True
)
collection.fillna("benign", inplace=True)

train_paths, test_paths = [], []
for i, row in tqdm(collection.iterrows()):
    hash = row["SHA-256"]
    if not os.path.exists(path := f"image/malware/{hash}.png"):
        path = f"image/benign/{hash}.png"
    if row["Architecture"] == args.input:
        test_paths.append(path)
    else:
        train_paths.append(path)
print(len(train_paths), len(test_paths))

model = Model(
    train_paths=train_paths,
    test_paths=test_paths,
    report_path=f"result/RQ4/{args.input}.txt",
)
model.run()
