"""
Trains a weather classification model using transfer learning with ResNet-18.

Flow
----
- loads and preprocesses image data from the training and test directories
- trains a ResNet-18 model with custom classification head
- implements checkpointing to resume training from last saved epoch
- evaluates performance after each epoch
- saves the model and latest checkpoint

Dependencies
------------
- PyTorch
- torchvision
- OpenCV
- glob, os, shutil, random
"""

import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

torch.backends.cudnn.benchmark= True

data_dir = 'E:/facultate/licenta/pda/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_data'
train_dir = os.path.join(data_dir, 'E:/facultate/licenta/pda/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/train')
test_dir = os.path.join(data_dir, 'E:/facultate/licenta/pda/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/test')

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)


# --- Image Transformation ---
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- Dataset  ---
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Model Definition ---
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

checkpoint_path='checkpoint.pth'
start_epoch=0
num_epochs = 10

def find_latest_checkpoint():
    """
    Finds the most recent checkpoint file in the checkpoint directory.

    Returns
    -------
    str or None
        Path to the latest checkpoint file, or None if no checkpoint exists.
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    if not checkpoints:
        return None

    latest_epoch = -1
    latest_checkpoint = None

    for checkpoint in checkpoints:
        try:
            epoch_num = int(checkpoint.split('_epoch_')[1].split('.')[0])
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_checkpoint = checkpoint
        except (ValueError, IndexError):
            continue

    return latest_checkpoint


latest_checkpoint = find_latest_checkpoint()
if latest_checkpoint:
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from {latest_checkpoint}")
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting training from scratch.")

# --- Training and Validation ---
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = val_loss / len(test_loader)
    val_acc = 100. * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
          f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")

    latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pth')
    torch.save(checkpoint, latest_path)

torch.save(model.state_dict(), 'weather_classification_model.pth')
