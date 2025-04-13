import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


def load_model(model_path):
    """
    Loads the trained weather classification model

    Parameters
    ----------
    model_path : str
        Path to the saved model weights

    Returns
    -------
    model : torch.nn.Module
        The loaded model
    device : torch. Device
        The device model is loaded on
    """
    model = models.resnet18(pretrained=False)

    train_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/train'
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    num_classes = len(class_names)

    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return model, device, class_names


def predict_image(image_path, model, device, class_names):
    """
    Makes a prediction for a single image

    Parameters
    ----------
    image_path : str
        Path to the image file
    model : torch.nn.Module
        The loaded model
    device : torch. Device
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

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_class = class_names[predicted.item()]
    confidence = probabilities[predicted].item() * 100

    return predicted_class, confidence, probabilities.tolist()


def display_prediction(image_path, predicted_class, confidence, class_names, probabilities):
    """
    Displays the image with its prediction results

    Parameters
    ----------
    image_path : str
        Path to the image file
    predicted_class : str
        The predicted class name
    confidence : float
        Confidence score (0-100%)
    class_names : list
        The list of class names
    probabilities : list
        The list of probability scores for all classes
    """
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    y_pos = range(len(class_names))

    plt.barh(y_pos, [p * 100 for p in probabilities], align='center')
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability (%)')
    plt.title('Class Probabilities')

    plt.tight_layout()
    plt.show()

def predict_batch(image_dir, model, device, class_names):
    """
    Makes predictions for all images in a folder

    Parameters
    ----------
    image_dir : str
        Path to the folder containing images
    model : torch.nn.Module
        The loaded model
    device : torch. Device
        The device to run inference on
    class_names : list
        The list of class names

    Returns
    -------
    results : list
        List of tuples containing (image_path, predicted_class, confidence)
    """
    results = []

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            predicted_class, confidence, _ = predict_image(image_path, model, device, class_names)
            results.append((image_file, predicted_class, confidence))
            print(f"Image: {image_file} - Predicted: {predicted_class} ({confidence:.2f}%)")

    return results

if __name__ == "__main__":
    model_path = 'weather_classification_model.pth'
    model, device, class_names = load_model(model_path)
    print(f"Model loaded successfully on {device}")
    print(f"Class names: {class_names}")

    image_path = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/test/cloudy/frame_00027.jpg'
    if image_path and os.path.isfile(image_path):
        predicted_class, confidence, probabilities = predict_image(image_path, model, device, class_names)
        print(f"\nPrediction results for {os.path.basename(image_path)}:")
        print(f"Predicted weather: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")

        print("\nClass probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {probabilities[i] * 100:.2f}%")

        try:
            display_prediction(image_path, predicted_class, confidence, class_names, probabilities)
        except ImportError:
            print("Matplotlib not available for visualization.")

    test_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/test/cloudy'
    if os.path.exists(test_dir):
        batch_results = predict_batch(test_dir, model, device, class_names)
        print("\nBatch Prediction Results:")
        for image_name, pred_class, conf in batch_results:
            print(f"{image_name}: {pred_class} ({conf:.2f}%)")