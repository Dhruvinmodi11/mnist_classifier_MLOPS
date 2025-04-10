# MNIST Digit Classifier

This project implements a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. The project demonstrates fundamental machine learning concepts including data processing, model architecture design, training, evaluation, and inference.

## Project Structure

```
mnist_classifier/
├── data/               # Directory for storing MNIST dataset (auto-downloaded)
├── models/             # Directory for saving trained models
├── src/                # Source code
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── model.py        # CNN model architecture
│   ├── train.py        # Model training and evaluation
│   └── inference.py    # Model inference and visualization
├── main.py             # Main entry point to run the complete pipeline
└── requirements.txt    # Project dependencies
```

## Features

- PyTorch-based CNN architecture optimized for digit recognition
- Data loading and preprocessing pipeline
- Model training with validation
- Performance visualization (accuracy, loss curves)
- Inference capabilities with confusion matrix and prediction visualization
- GPU support when available

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist-classifier.git
   cd mnist-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the complete pipeline

```bash
python main.py
```

This will:
- Download the MNIST dataset (if not already present)
- Create and train the CNN model (default: 5 epochs)
- Evaluate the model on the test set
- Generate visualizations

You can customize the execution with various command-line arguments:
```bash
python main.py --batch_size 64 --epochs 10 --learning_rate 0.01 --visualize
```

### Training the model separately

```bash
cd src
python train.py
```

### Evaluating and making predictions

```bash
cd src
python inference.py --model_path ../models/mnist_best.pth
```

## Model Architecture

The model architecture is a CNN with the following structure:
- 2 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- 2 fully connected layers
- Log softmax output for 10 digit classes

## Results

The model typically achieves 98-99% accuracy on the MNIST test set after 5 epochs of training.

## License

MIT

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and Corinna Cortes: http://yann.lecun.com/exdb/mnist/ 