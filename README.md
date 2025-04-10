# MNIST Digit Classifier with MLflow and Optuna

This project implements a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset, enhanced with modern MLOps practices including experiment tracking and hyperparameter optimization.

## Features

- **PyTorch-based CNN** architecture optimized for digit recognition (98%+ accuracy)
- **Data loading and preprocessing** pipeline with torchvision
- **Model training** with validation and comprehensive metrics
- **Performance visualization** (accuracy curves, loss curves, confusion matrix)
- **MLflow integration** for experiment tracking and model registry
- **Optuna implementation** for automated hyperparameter tuning
- **GPU support** when available for faster training

## Project Structure

```
mnist_classifier/
├── data/                       # Directory for storing MNIST dataset (auto-downloaded)
├── models/                     # Directory for saving trained models
├── src/                        # Source code
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py                # CNN model architecture
│   ├── train.py                # Model training with MLflow integration
│   └── inference.py            # Model inference and visualization
├── main.py                     # Main entry point for standard training
├── tune.py                     # Hyperparameter tuning with Optuna
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Advanced ML Engineering Features

### MLflow Integration
- **Experiment Tracking:** All training runs are logged, including hyperparameters, metrics over time, and artifacts.
- **Model Registry:** Best models are automatically registered for easy retrieval and deployment.
- **Artifact Management:** Training curves, confusion matrices, and model checkpoints are saved for each run.
- **Run Comparison:** Easily compare different training configurations through the MLflow UI.

### Optuna Hyperparameter Tuning
- **Automated Search:** Efficiently searches the hyperparameter space to find optimal model configurations.
- **Bayesian Optimization:** Uses intelligent sampling to focus on promising areas of the search space.
- **Integration with MLflow:** Each trial is logged as a nested run in MLflow for comprehensive tracking.
- **Persistence:** Tuning studies are saved to SQLite, allowing for resumable optimization sessions.

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- matplotlib
- numpy
- mlflow
- optuna

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/mnist-classifier-mlops.git
   cd mnist-classifier-mlops
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Standard Training with MLflow Tracking

```bash
python main.py
```

This will:
- Download the MNIST dataset (if not already present)
- Train the CNN model (default: 5 epochs) with MLflow logging
- Save the trained model both locally and to MLflow model registry
- Generate visualizations of model performance

You can customize the execution with various command-line arguments:
```bash
python main.py --batch_size 64 --epochs 10 --learning_rate 0.01 --mlflow_experiment_name "MNIST_Custom_Run"
```

### Hyperparameter Tuning with Optuna

```bash
python tune.py --n_trials 20
```

This will:
- Run 20 trials with different hyperparameter configurations
- Log all trials to MLflow for comparison
- Report the best hyperparameters found
- Store the optimization history in a SQLite database

### Viewing Results

Start the MLflow UI to view your experiments:
```bash
mlflow ui
```

Then open your browser at http://localhost:5000 to explore your runs.

## Model Architecture

The model is a CNN with:
- 2 convolutional layers with batch normalization
- Max pooling and dropout for regularization
- 2 fully connected layers
- Log softmax output for 10 digit classes

## Results

The model typically achieves 98-99% accuracy on the MNIST test set after 5 epochs of training.

With hyperparameter optimization using Optuna, we can achieve even better results by finding optimal learning rates and momentum values.

## License

MIT

## Acknowledgements

- The MNIST dataset is provided by Yann LeCun and Corinna Cortes: http://yann.lecun.com/exdb/mnist/ 