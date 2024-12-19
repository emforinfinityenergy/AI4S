import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.label_df = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = f"{self.label_df.iloc[idx, 0]}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path)
        label = self.label_df.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义卷积神经网络 Design the CNN
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(3),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((80, 60)),
    transforms.ToTensor(),
])
train_dataset = MyDataset(img_dir="train_t2/image_train", label_file="train_t2/label_train.csv", transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=25, shuffle=True)

net = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
device = "cpu"

NUM_EPOCHS = 20
for epoch in range(NUM_EPOCHS):
    net.train()
    print("epoch:", epoch)
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_accuracy = correct / total
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Accuracy: {epoch_accuracy}")
