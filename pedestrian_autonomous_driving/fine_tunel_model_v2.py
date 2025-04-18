"""
Final version of the training model. New approach includes:
-progressive layer freezing
-k-fold cross-validation
-improved data augmentation

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
from PIL import Image
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import numpy as np
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import copy

torch.backends.cudnn.benchmark = True

base_data_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset'
train_dir = os.path.join(base_data_dir, 'train')
test_dir = os.path.join(base_data_dir, 'test')
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def create_model():
    model = models.resnet18(pretrained=True)
    num_classes = len(full_dataset.classes)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model

def train_model_with_kfold(model_fn, dataset, num_epochs=10, num_folds=5):
    results = {}

    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)

        train_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=train_subsampler
        )
        val_loader = DataLoader(
            dataset,
            batch_size=32,
            sampler=val_subsampler
        )

        model = model_fn()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=0.001, weight_decay=1e-4)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                      verbose=True, min_lr=1e-6)

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            if epoch == 3:
                print("Unfreezing layer4...")
                for param in model.layer4.parameters():
                    param.requires_grad = True

            if epoch == 6:
                print("Unfreezing layer3...")
                for param in model.layer3.parameters():
                    param.requires_grad = True

            print(f'Epoch {epoch + 1}/{num_epochs}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    dataloader = train_loader
                else:
                    model.eval()
                    dataloader = val_loader

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(dataloader.sampler)
                epoch_acc = running_corrects.double() / len(dataloader.sampler)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val':
                    scheduler.step(epoch_loss)

                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

            print()

        model.load_state_dict(best_model_wts)

        fold_checkpoint_path = os.path.join(checkpoint_dir, f'model_fold_{fold}.pth')
        torch.save(model.state_dict(), fold_checkpoint_path)

        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * correct / total

        print(f'Fold {fold} Test Accuracy: {test_acc:.2f}%')

        results[fold] = test_acc

    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
    print('--------------------------------')
    sum_acc = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value:.2f}%')
        sum_acc += value
    print(f'Average Test Accuracy: {sum_acc / len(results):.2f}%')

    return results


def predict_with_ensemble(image_path, model_paths, device, class_names):
    """
    Makes a prediction using an ensemble of models

    Parameters
    ----------
    image_path : str
        Path to the image file
    model_paths : list
        List of paths to saved model weights
    device : torch.device
        The device to run inference on
    class_names : list
        The list of class names

    Returns
    -------
    predicted_class : str
        The predicted class name
    confidence : float
        Confidence score
    probabilities : list
        The list of probability scores for all classes
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    ensemble_probs = []
    for model_path in model_paths:
        model = create_model()
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            ensemble_probs.append(probs)

    avg_probs = torch.stack(ensemble_probs).mean(dim=0)

    _, predicted = torch.max(avg_probs.unsqueeze(0), 1)
    predicted_class = class_names[predicted.item()]
    confidence = avg_probs[predicted].item() * 100

    return predicted_class, confidence, avg_probs.tolist()


if __name__ == "__main__":
    print("Starting K-fold cross-validation training...")

    fold_results = train_model_with_kfold(create_model, full_dataset, num_epochs=10, num_folds=5)

    print("\nTraining final model on entire dataset...")
    final_model = create_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_model = final_model.to(device)

    criterion = nn.CrossEntropyLoss()
    for param in final_model.parameters():
        param.requires_grad = False
    for param in final_model.fc.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, final_model.parameters()),
                           lr=0.001, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                  verbose=True, min_lr=1e-6)

    train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)

    num_epochs = 15
    best_model_wts = copy.deepcopy(final_model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        if epoch == 3:
            print("Unfreezing layer4...")
            for param in final_model.layer4.parameters():
                param.requires_grad = True

        if epoch == 6:
            print("Unfreezing layer3...")
            for param in final_model.layer3.parameters():
                param.requires_grad = True

        if epoch == 9:
            print("Unfreezing layer2...")
            for param in final_model.layer2.parameters():
                param.requires_grad = True

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        final_model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = final_model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        final_model.eval()
        val_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = final_model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100. * correct / total

        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')

        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(final_model.state_dict())

        print()

    final_model.load_state_dict(best_model_wts)

    torch.save(final_model.state_dict(), 'weather_classification_model_improved.pth')
    print(f"Final model saved with validation accuracy: {best_acc:.2f}%")