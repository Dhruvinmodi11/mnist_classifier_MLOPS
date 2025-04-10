import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path
import mlflow
import mlflow.pytorch

# Use relative imports within the package
from .data_loader import load_mnist_data
from .model import MNISTClassifier

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001, 
                momentum=0.9, log_interval=100, save_dir="../models",
                mlflow_tracking_uri=None, mlflow_experiment_name="MNIST_Training"):
    """
    Train the model, evaluate, and log with MLflow
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        epochs: Number of training epochs
        lr: Learning rate
        momentum: Momentum for SGD optimizer
        log_interval: How often to log training progress
        save_dir: Directory to save model (outside MLflow)
        mlflow_tracking_uri: URI for MLflow tracking server (optional)
        mlflow_experiment_name: Name for the MLflow experiment
        
    Returns:
        model: Trained model
        best_test_acc: Best test accuracy achieved
    """
    # Setup MLflow
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)

    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Log hyperparameters
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("batch_size", train_loader.batch_size)
        mlflow.log_param("optimizer", "SGD") # Example, could be dynamic

        # Create save directory if it doesn't exist (for non-MLflow artifacts)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        mlflow.log_param("device", str(device))
        
        model = model.to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        
        # Re-introduce lists to store metrics across epochs
        train_losses_history = []
        test_losses_history = []
        train_accs_history = []
        test_accs_history = []

        # Training loop
        start_time = time.time()
        best_test_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % log_interval == 0:
                    print(f'Epoch: {epoch}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                          f'Loss: {loss.item():.6f}')
            
            train_loss /= len(train_loader)
            train_acc = 100. * correct / total
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            # Append to history lists
            train_losses_history.append(train_loss)
            train_accs_history.append(train_acc)
            
            # Evaluation phase
            model.eval()
            test_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    test_loss += loss.item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
            
            test_loss /= len(test_loader)
            test_acc = 100. * correct / total
            mlflow.log_metric("test_loss", test_loss, step=epoch)
            mlflow.log_metric("test_acc", test_acc, step=epoch)
            # Append to history lists
            test_losses_history.append(test_loss)
            test_accs_history.append(test_acc)
            
            print(f'Epoch: {epoch}/{epochs}\t'
                  f'Train Loss: {train_loss:.4f}\t'
                  f'Train Acc: {train_acc:.2f}%\t'
                  f'Test Loss: {test_loss:.4f}\t'
                  f'Test Acc: {test_acc:.2f}%')
            
            # Save best model based on test accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                # Save locally (optional)
                torch.save(model.state_dict(), save_path / "mnist_best_local.pth") 
                # Log model with MLflow
                mlflow.pytorch.log_model(model, "model", registered_model_name="mnist-cnn-best")
                mlflow.log_metric("best_test_acc", best_test_acc, step=epoch)
                print(f"Logged best model with accuracy: {best_test_acc:.2f}% to MLflow")
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.2f} seconds')
        print(f'Best test accuracy: {best_test_acc:.2f}%')
        mlflow.log_metric("training_time_seconds", total_time)

        # Log final model separately (optional, could rely on best)
        mlflow.pytorch.log_model(model, "final_model")
        torch.save(model.state_dict(), save_path / "mnist_final_local.pth")
        print(f"Saved final model locally to {save_path / 'mnist_final_local.pth'}")

        # Plot and log training history figure using the accumulated lists
        fig = plot_training_history(
            train_losses_history, 
            test_losses_history, 
            train_accs_history, 
            test_accs_history, 
            save_plot=False # Don't save locally, just return fig
        )
        if fig:
            mlflow.log_figure(fig, "training_history.png")
            print("Logged training history plot to MLflow")

    return model, best_test_acc

def plot_training_history(train_losses, test_losses, train_accs, test_accs, save_dir="../models", save_plot=True):
    """
    Plot training and test losses and accuracies
    
    Args:
        train_losses: List of training losses
        test_losses: List of test losses
        train_accs: List of training accuracies
        test_accs: List of test accuracies
        save_dir: Directory to save plots (if save_plot is True)
        save_plot: Whether to save the plot locally
        
    Returns:
        fig: Matplotlib figure object (if save_plot is False)
    """
    if not train_losses or not test_losses or not train_accs or not test_accs:
        print("Warning: Missing metrics data, cannot plot training history.")
        return None
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)

    # Plot losses
    axs[0].plot(epochs_range, train_losses, label='Train Loss')
    axs[0].plot(epochs_range, test_losses, label='Test Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].set_title('Training and Test Losses')
    
    # Plot accuracies
    axs[1].plot(epochs_range, train_accs, label='Train Accuracy')
    axs[1].plot(epochs_range, test_accs, label='Test Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    axs[1].set_title('Training and Test Accuracies')
    
    fig.tight_layout()
    
    if save_plot:
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        plot_path = save_path / "training_history.png"
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Training history plots saved to {plot_path}")
        return None
    else:
        return fig # Return the figure object for MLflow logging

# Remove the direct execution block if train.py is not meant to be run directly
# Or adjust imports if it needs to run stand-alone
# if __name__ == "__main__":
#     # Load data
#     from data_loader import load_mnist_data
#     from model import MNISTClassifier
#
#     train_loader, test_loader = load_mnist_data(batch_size=128)
#     
#     # Create model
#     model = MNISTClassifier()
#     
#     # Train model
#     model, history = train_model(
#         model, train_loader, test_loader, epochs=5
#     )
#     
#     # Plot training history
#     plot_training_history(history['train_loss'], history['test_loss'], 
#                           history['train_acc'], history['test_acc']) 