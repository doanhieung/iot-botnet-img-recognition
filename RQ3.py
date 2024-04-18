import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

malware_df = pd.read_csv("data/IoT_Malware_Collection_2021.csv")
benign_df = pd.read_csv("data/IoT_Benign_Collection_2021.csv")
all_df = pd.concat([malware_df, benign_df], ignore_index=True)
all_df["Label"] = all_df["Label"].fillna("Benign")

gafgyt_l = all_df.query("Label == 'Gafgyt'")["SHA-256"].values
mirai_l = all_df.query("Label == 'Mirai'")["SHA-256"].values
benign_l = all_df.query("Label == 'Benign'")["SHA-256"].values
print(len(gafgyt_l), len(mirai_l), len(benign_l))


def get_label(path):
    hash_str = os.path.splitext(os.path.basename(path))[0]
    if hash_str in benign_l:
        return 0
    elif hash_str in gafgyt_l:
        return 1
    elif hash_str in mirai_l:
        return 2
    else:
        return 3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 72, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(72 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
            nn.Softmax(1),
        )

    def forward(self, x):
        return self.net(x)


class MalDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath, 0)
        image = np.reshape(image, (64, 64, 1))

        img_tensor = torch.from_numpy(image).float()
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor /= 255.0

        label = get_label(image_filepath)

        return img_tensor, label


class Model:
    def __init__(self, train_paths, test_paths, report_path, batch_size=32):
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.batch_size = batch_size
        self.report_path = report_path

    def run(self):
        # Load dataset
        train_set = MalDataset(self.train_paths)
        test_set = MalDataset(self.test_paths)
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=True)

        # Define model
        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(), lr=0.00001)

        # Train the model
        num_iterations = 5000
        step_per_epoch = len(train_set) // self.batch_size + 1
        if (epochs := num_iterations // step_per_epoch + 1) < 10:
            epochs = 10
        for epoch in range(epochs):
            running_loss = 0.0
            for data in train_loader:
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            print("Epoch %d/%d - loss: %.4f" % (epoch + 1, epochs, running_loss))
            running_loss = 0.0

        # Test
        y_true, y_pred = [], []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        clf_rp = metrics.classification_report(y_true, y_pred, digits=4)
        cfn_matrix = metrics.confusion_matrix(y_true, y_pred)

        with open(self.report_path, "w") as f:
            f.write(clf_rp + "\n")
            f.write(str(cfn_matrix) + "\n")


SEED = 42

malware_paths = glob(f"image/malware/*")
benign_paths = glob(f"image/benign/*")
print(len(malware_paths), len(benign_paths))

paths = malware_paths + benign_paths
labels = []
for path in tqdm(paths):
    labels.append(get_label(path))
train_paths, test_paths, _, _ = train_test_split(
    paths, labels, test_size=0.3, random_state=SEED, stratify=labels
)

print("Start...")
model = Model(
    train_paths=train_paths, test_paths=test_paths, report_path=f"result/RQ3.txt"
)
model.run()
