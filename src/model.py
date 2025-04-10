import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification
    """
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Max pooling, dropout
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

def get_model_summary(model, input_size=(1, 1, 28, 28)):
    """
    Print a summary of the model architecture
    
    Args:
        model: PyTorch model
        input_size: Size of input tensor (batch_size, channels, height, width)
    """
    device = next(model.parameters()).device
    dummy_input = torch.zeros(input_size).to(device)
    
    # Forward pass with dummy input to track tensor dimensions
    model.eval()
    output = model(dummy_input)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_size}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params

if __name__ == "__main__":
    # Create model
    model = MNISTClassifier()
    
    # Print model summary
    get_model_summary(model) 