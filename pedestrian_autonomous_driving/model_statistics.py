import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np


def create_model():
    """Create the same model architecture as during training"""
    model = models.resnet18(pretrained=False)
    num_classes = len(os.listdir(os.path.join(base_data_dir, 'train')))

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on specified dataloader"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            for i in range(len(labels)):
                label = labels[i].item()
                prediction = preds[i].item()

                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0

                class_total[label] += 1
                if label == prediction:
                    class_correct[label] += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    class_acc = {cls: class_correct.get(cls, 0) / class_total.get(cls, 1)
                 for cls in class_total.keys()}

    return epoch_loss, epoch_acc.item(), class_acc


base_data_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset'
checkpoint_dir = 'checkpoints'
test_dir = os.path.join(base_data_dir, 'test')
train_dir = os.path.join(base_data_dir, 'train')

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)
train_dataset = datasets.ImageFolder(train_dir, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes
print(f"Classes: {class_names}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

criterion = nn.CrossEntropyLoss()
all_checkpoints = []

for file in os.listdir(checkpoint_dir):
    if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, file)
        epoch_num = int(file.split('_')[-1].split('.')[0])
        all_checkpoints.append((epoch_num, checkpoint_path, 'regular'))

for file in os.listdir(checkpoint_dir):
    if file.startswith('model_fold_') and file.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, file)
        fold_num = int(file.split('_')[-1].split('.')[0])
        all_checkpoints.append((fold_num, checkpoint_path, 'kfold'))

final_model_path = 'weather_classification_model_improved.pth'
if os.path.exists(final_model_path):
    all_checkpoints.append((-1, final_model_path, 'final'))
else:
    original_model_path = 'weather_classification_model.pth'
    if os.path.exists(original_model_path):
        all_checkpoints.append((-1, original_model_path, 'original'))

print(f"Found {len(all_checkpoints)} checkpoint files to evaluate")

results = {
    'regular': {'train': [], 'test': [], 'class_acc': []},
    'kfold': {'train': [], 'test': [], 'class_acc': []},
    'final': {'train': [], 'test': [], 'class_acc': []},
    'original': {'train': [], 'test': [], 'class_acc': []}
}

for epoch_or_fold, checkpoint_path, model_type in sorted(all_checkpoints):
    print(f"\nEvaluating {model_type} model: {os.path.basename(checkpoint_path)}")
    model = create_model()

    try:
        if model_type == 'regular':
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {epoch_or_fold}")
        else:
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model weights from {os.path.basename(checkpoint_path)}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        continue

    model = model.to(device)
    train_loss, train_acc, train_class_acc = evaluate_model(model, train_loader, criterion, device)
    test_loss, test_acc, test_class_acc = evaluate_model(model, test_loader, criterion, device)

    print(f"Train Accuracy: {train_acc * 100:.2f}%, Test Accuracy: {test_acc * 100:.2f}%")
    print("Per-class Test Accuracy:")
    for class_idx, acc in test_class_acc.items():
        class_name = class_names[class_idx]
        print(f"{class_name}: {acc * 100:.2f}%")

    results[model_type]['train'].append((epoch_or_fold, train_acc))
    results[model_type]['test'].append((epoch_or_fold, test_acc))
    results[model_type]['class_acc'].append((epoch_or_fold, test_class_acc))

if results['regular']['train']:
    plt.figure(figsize=(10, 6))

    train_data = sorted(results['regular']['train'])
    test_data = sorted(results['regular']['test'])

    epochs = [e for e, _ in train_data]
    train_acc = [a * 100 for _, a in train_data]
    test_acc = [a * 100 for _, a in test_data]

    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_acc, 'r-', label='Test Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_vs_epoch.png')
    print("\nPlot saved as 'accuracy_vs_epoch.png'")

if len(results['kfold']['test']) > 1:
    plt.figure(figsize=(10, 6))

    fold_data = sorted(results['kfold']['test'])
    folds = [f for f, _ in fold_data]
    fold_acc = [a * 100 for _, a in fold_data]

    plt.bar(folds, fold_acc)
    plt.axhline(y=sum(fold_acc) / len(fold_acc), color='r', linestyle='-', label='Average')
    plt.title('K-Fold Cross Validation Results')
    plt.xlabel('Fold')
    plt.ylabel('Test Accuracy (%)')
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('kfold_results.png')
    print("Plot saved as 'kfold_results.png'")

final_models = []
if results['final']['test']:
    final_models.append(('Improved', results['final']['test'][0][1]))
if results['original']['test']:
    final_models.append(('Original', results['original']['test'][0][1]))

if len(final_models) > 0:
    plt.figure(figsize=(8, 6))

    models = [m for m, _ in final_models]
    accs = [a * 100 for _, a in final_models]

    plt.bar(models, accs)
    plt.title('Final Model Comparison')
    plt.ylabel('Test Accuracy (%)')
    plt.ylim([0, 100])

    for i, v in enumerate(accs):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')

    plt.savefig('model_comparison.png')
    print("Plot saved as 'model_comparison.png'")

if results['final']['test'] or results['original']['test']:
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    best_model_path = final_model_path if os.path.exists(final_model_path) else original_model_path

    model = create_model()
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Plot saved as 'confusion_matrix.png'")

print("\n======= SUMMARY STATISTICS =======")

class_avg_acc = {cls: [] for cls in class_names}

for model_type in ['final', 'kfold', 'regular']:
    for _, class_acc_dict in results[model_type]['class_acc']:
        for class_idx, acc in class_acc_dict.items():
            class_name = class_names[class_idx]
            class_avg_acc[class_name].append(acc)

print("\nAverage Accuracy per Class:")
for class_name, accs in class_avg_acc.items():
    if accs:
        avg_acc = sum(accs) / len(accs)
        print(f"{class_name}: {avg_acc * 100:.2f}%")

best_acc = 0
best_model_name = "None"

for model_type in ['final', 'kfold', 'original']:
    for _, acc in results[model_type]['test']:
        if acc > best_acc:
            best_acc = acc
            best_model_name = model_type

print(f"\nBest Model: {best_model_name} with {best_acc * 100:.2f}% test accuracy")

if results['kfold']['test']:
    kfold_accs = [acc for _, acc in results['kfold']['test']]
    kfold_avg = sum(kfold_accs) / len(kfold_accs)
    print(f"K-fold Cross-validation Average: {kfold_avg * 100:.2f}%")