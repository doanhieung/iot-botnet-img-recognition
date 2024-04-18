import pandas as pd
from tqdm import tqdm
from glob import glob
from datetime import datetime
from sklearn.utils import shuffle

from Model import Model

SEED = 42

collection_path = "data/IoT_Malware_Collection_2021.csv"
collection = pd.read_csv(collection_path)

paths = []
for i, row in tqdm(collection.iterrows()):
    hash = row["SHA-256"]
    path = f"image/malware/{hash}.png"

    date = row["Date"]
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    paths.append((path, date))

paths = sorted(paths, key=lambda e: e[1])
paths, _ = zip(*paths)
malware_paths = list(paths)

benign_paths = list(glob("image/benign/*"))
benign_paths = shuffle(benign_paths, random_state=SEED)

mal_split = int(len(malware_paths) * 7 / 10)
beg_split = int(len(benign_paths) * 7 / 10)
train_paths = malware_paths[:mal_split] + benign_paths[:beg_split]
test_paths = malware_paths[mal_split:] + benign_paths[beg_split:]

model = Model(
    train_paths=train_paths, test_paths=test_paths, report_path=f"result/RQ2.txt"
)
model.run()
