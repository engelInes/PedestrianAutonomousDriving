from PIL import Image
import matplotlib
from matplotlib import transforms

from fine_tunel_model_v2 import create_model, predict_with_ensemble, checkpoint_dir

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pyttsx3
import os
import torch
import numpy as np


def speak_prediction(predicted_class, confidence):
    """
    Uses text-to-speech to announce the prediction.

    Parameters
    ----------
    predicted_class : str
        The predicted weather class
    confidence : float
        Confidence score (0-100%)
    """
    engine = pyttsx3.init()
    message = f"The predicted weather is {predicted_class} with {confidence:.1f} percent confidence."
    engine.say(message)
    engine.runAndWait()


def load_model(model_path, use_ensemble=False):
    """
    Loads the trained weather classification model

    Parameters
    ----------
    model_path : str or list
        Path to the saved model weights or list of paths for ensemble
    use_ensemble : bool
        Whether to use ensemble of models

    Returns
    -------
    model : torch.nn.Module or list
        The loaded model(s)
    device : torch.Device
        The device model is loaded on
    class_names : list
        List of class names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get class names
    train_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/train'
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]

    if use_ensemble:
        # Return paths for ensemble prediction
        return model_path, device, class_names
    else:
        # Load single model
        model = create_model()  # Assuming create_model function is accessible
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to(device)
        return model, device, class_names


def predict_image(image_path, model, device, class_names, use_ensemble=False):
    """
    Makes a prediction for a single image

    Parameters
    ----------
    image_path : str
        Path to the image file
    model : torch.nn.Module or list
        The loaded model or list of model paths for ensemble
    device : torch.Device
        The device to run inference on
    class_names : list
        The list of class names
    use_ensemble : bool
        Whether to use ensemble prediction

    Returns
    -------
    predicted_class : str
        The predicted class name
    confidence : float
        Confidence score
    probabilities : list
        The list of probability scores for all classes
    """
    if use_ensemble:
        return predict_with_ensemble(image_path, model, device, class_names)

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


def predict_batch(image_dir, model, device, class_names, use_ensemble=False, confidence_threshold=50):
    """
    Makes predictions for all images in a folder

    Parameters
    ----------
    image_dir : str
        Path to the folder containing images
    model : torch.nn.Module or list
        The loaded model or list of model paths for ensemble
    device : torch.Device
        The device to run inference on
    class_names : list
        The list of class names
    use_ensemble : bool
        Whether to use ensemble prediction
    confidence_threshold : float
        Minimum confidence required for a valid prediction

    Returns
    -------
    results : list
        List of tuples containing (image_path, predicted_class, confidence)
    """
    results = []
    low_confidence_count = 0

    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_file)
            predicted_class, confidence, probabilities = predict_image(
                image_path, model, device, class_names, use_ensemble)

            if confidence >= confidence_threshold:
                speak_prediction(predicted_class, confidence)
                results.append((image_file, predicted_class, confidence))
                print(f"Image: {image_file} - Predicted: {predicted_class} ({confidence:.2f}%)")
            else:
                low_confidence_count += 1
                print(f"Image: {image_file} - Low confidence prediction: {predicted_class} ({confidence:.2f}%)")

    if low_confidence_count > 0:
        print(
            f"\n{low_confidence_count} images had predictions below the confidence threshold ({confidence_threshold}%)")

    return results


if __name__ == "__main__":
    # Check if ensemble models exist
    ensemble_paths = [os.path.join(checkpoint_dir, f'model_fold_{i}.pth') for i in range(5)]
    use_ensemble = all(os.path.exists(path) for path in ensemble_paths)

    if use_ensemble:
        print("Using ensemble prediction with 5-fold models")
        model_path = ensemble_paths
    else:
        print("Using single model prediction")
        model_path = 'weather_classification_model_improved.pth'
        if not os.path.exists(model_path):
            model_path = 'weather_classification_model.pth'

    model, device, class_names = load_model(model_path, use_ensemble)
    print(f"Model loaded successfully on {device}")
    print(f"Class names: {class_names}")

    image_path = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/test/cloudy/frame_00027.jpg'
    if image_path and os.path.isfile(image_path):
        predicted_class, confidence, probabilities = predict_image(
            image_path, model, device, class_names, use_ensemble)

        print(f"\nPrediction results for {os.path.basename(image_path)}:")
        print(f"Predicted weather: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        speak_prediction(predicted_class, confidence)

        print("\nClass probabilities:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {probabilities[i] * 100:.2f}%")

        try:
            display_prediction(image_path, predicted_class, confidence, class_names, probabilities)
        except ImportError:
            print("Matplotlib not available for visualization.")

    # test_dir = 'E:/facultate/licenta/pda_backup/PedestrianAutonomousDriving/pedestrian_autonomous_driving/weather_dataset/test/cloudy'
    # if os.path.exists(test_dir):
    #     batch_results = predict_batch(test_dir, model, device, class_names,
    #                                   use_ensemble, confidence_threshold=70)
    #
    #     # Print summary statistics
    #     if batch_results:
    #         classes = {}
    #         for _, pred_class, conf in batch_results:
    #             if pred_class in classes:
    #                 classes[pred_class].append(conf)
    #             else:
    #                 classes[pred_class] = [conf]
    #
    #         print("\nBatch Prediction Summary:")
    #         for class_name, confs in classes.items():
    #             avg_conf = sum(confs) / len(confs)
    #             print(f"{class_name}: {len(confs)} images, Avg conf: {avg_conf:.2f}%")