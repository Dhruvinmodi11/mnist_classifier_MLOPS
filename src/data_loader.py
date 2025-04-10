import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

def load_mnist_data(batch_size=64, shuffle=True, random_seed=42):
    """
    Load MNIST dataset for training and testing
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        random_seed: Random seed for reproducibility
        
    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

def visualize_samples(data_loader, num_samples=5):
    """
    Visualize random samples from the dataset
    
    Args:
        data_loader: DataLoader containing the data
        num_samples: Number of samples to visualize
    """
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Create a figure
    plt.figure(figsize=(12, 4))
    
    # Plot images
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png')
    plt.close()
    
    return images, labels

if __name__ == "__main__":
    # Load data
    train_loader, test_loader = load_mnist_data()
    
    # Display dataset information
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # Visualize some samples
    visualize_samples(train_loader)
    print("Sample images saved as 'data_samples.png'") 