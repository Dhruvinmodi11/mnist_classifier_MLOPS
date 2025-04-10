import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

# Use relative imports within the package
from .model import MNISTClassifier
from .data_loader import load_mnist_data

def load_model(model_path, model_class=MNISTClassifier):
    """
    Load a trained model from disk
    
    Args:
        model_path: Path to the model file
        model_class: Model class to instantiate
        
    Returns:
        model: Loaded model
    """
    # Check if model exists
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create model instance
    model = model_class()
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model, device

def predict_single_image(model, image, device):
    """
    Make a prediction for a single image
    
    Args:
        model: Trained PyTorch model
        image: Image tensor (C, H, W)
        device: Device to use for inference
        
    Returns:
        prediction: Predicted class
        probabilities: Class probabilities
    """
    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        
    # Get prediction and probabilities
    _, prediction = torch.max(output, 1)
    return prediction.item(), probabilities[0]

def visualize_predictions(images, labels, predictions, probabilities, num_samples=5):
    """
    Visualize model predictions
    
    Args:
        images: Batch of images
        labels: Ground truth labels
        predictions: Model predictions
        probabilities: Prediction probabilities
        num_samples: Number of samples to visualize
    """
    # Create figure
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(min(num_samples, len(images))):
        # Display image
        plt.subplot(num_samples, 2, 2*i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis('off')
        
        # Display probabilities
        plt.subplot(num_samples, 2, 2*i + 2)
        probs = probabilities[i].cpu().numpy()
        plt.bar(range(10), probs)
        plt.xticks(range(10))
        plt.xlabel('Digit')
        plt.ylabel('Probability')
        plt.title('Class Probabilities')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()
    
    print(f"Prediction visualization saved as 'predictions.png'")

def evaluate_model(model, test_loader, device):
    """
    Evaluate model on test data
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        device: Device to use for inference
        
    Returns:
        accuracy: Model accuracy
        confusion_matrix: Confusion matrix
    """
    correct = 0
    total = 0
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int)
    
    # Store some examples for visualization
    example_images = []
    example_labels = []
    example_preds = []
    example_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probabilities = torch.exp(outputs)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # Store some examples
            if len(example_images) < 5:
                for i in range(min(5, len(images))):
                    if len(example_images) < 5:
                        example_images.append(images[i])
                        example_labels.append(labels[i].item())
                        example_preds.append(predicted[i].item())
                        example_probs.append(probabilities[i])
    
    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Visualize some predictions
    visualize_predictions(
        example_images, 
        example_labels, 
        example_preds, 
        example_probs
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix.cpu().numpy(), interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels and ticks
    classes = list(range(10))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j].item(), 'd'),
                     ha="center", va="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    print(f"Confusion matrix saved as 'confusion_matrix.png'")
    
    return accuracy, confusion_matrix

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MNIST Classifier Inference')
    parser.add_argument('--model_path', type=str, default='../models/mnist_best.pth',
                        help='Path to the trained model')
    args = parser.parse_args()

    # These imports need to be adjusted if running inference.py directly
    from model import MNISTClassifier
    from data_loader import load_mnist_data

    # Load model
    model, device = load_model(args.model_path)
    
    # Load test data
    _, test_loader = load_mnist_data(batch_size=64)
    
    # Evaluate model
    accuracy, confusion_matrix = evaluate_model(model, test_loader, device) 