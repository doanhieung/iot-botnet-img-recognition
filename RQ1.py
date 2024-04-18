from glob import glob
from sklearn.model_selection import train_test_split

from Model import Model

SEED = 42

malware_paths = glob(f"image/malware/*")
benign_paths = glob(f"image/benign/*")
print(len(malware_paths), len(benign_paths))

paths = malware_paths + benign_paths
labels = [1] * len(malware_paths) + [0] * len(benign_paths)
train_paths, test_paths, _, _ = train_test_split(
    paths, labels, test_size=0.3, random_state=SEED, stratify=labels
)

model = Model(
    train_paths=train_paths, test_paths=test_paths, report_path=f"result/RQ1.txt"
)
model.run()
