#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for MNIST classifier project.
This script orchestrates the complete workflow: data loading, 
model creation, training, evaluation, and visualization.
"""

import os
import argparse
from pathlib import Path

# Import project modules
from src.data_loader import load_mnist_data, visualize_samples
from src.model import MNISTClassifier, get_model_summary
from src.train import train_model
from src.inference import evaluate_model

def main(args):
    """
    Main function to run the complete MNIST classification pipeline.
    
    Args:
        args: Command line arguments
    """
    print("\n" + "="*50)
    print("MNIST Digit Classification Pipeline")
    print("="*50 + "\n")
    
    # Ensure model directory exists
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    
    # Step 1: Load and visualize data
    print("\n[Step 1/4] Loading and preprocessing data...")
    train_loader, test_loader = load_mnist_data(
        batch_size=args.batch_size, 
        shuffle=True,
        random_seed=args.seed
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Visualize some training samples
    if args.visualize:
        print("Visualizing sample images...")
        visualize_samples(train_loader, num_samples=5)
    
    # Step 2: Create and summarize model
    print("\n[Step 2/4] Creating model architecture...")
    model = MNISTClassifier()
    get_model_summary(model)
    
    # Step 3: Train the model
    if not args.skip_training:
        print("\n[Step 3/4] Training model...")
        model, best_test_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            momentum=args.momentum,
            log_interval=args.log_interval,
            save_dir=args.model_dir,
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=args.mlflow_experiment_name
        )
    else:
        print("\n[Step 3/4] Skipping training (--skip_training flag set)")
        best_test_acc = -1
        
    # Step 4: Evaluate model on test data
    print("\n[Step 4/4] Evaluating model...")
    
    model_to_evaluate = None
    if args.skip_training:
        local_model_path = os.path.join(args.model_dir, "mnist_best_local.pth")
        if os.path.exists(local_model_path):
             print(f"Loading locally saved best model from {local_model_path}")
             import torch
             device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
             model = MNISTClassifier()
             model.load_state_dict(torch.load(local_model_path, map_location=device))
             model = model.to(device)
             model_to_evaluate = model
        else:
            print("Local model not found, attempting to load 'best' model from MLflow...")
            try:
                import mlflow
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name(args.mlflow_experiment_name)
                if not experiment:
                    raise ValueError(f"MLflow experiment '{args.mlflow_experiment_name}' not found.")
                runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
                if not runs:
                    raise ValueError(f"No runs found in experiment '{args.mlflow_experiment_name}'. Please train a model first.")
                best_run_id = runs[0].info.run_id
                model_uri = f"runs:/{best_run_id}/model"
                print(f"Loading model from MLflow URI: {model_uri}")
                model_to_evaluate = mlflow.pytorch.load_model(model_uri)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_to_evaluate = model_to_evaluate.to(device)

            except Exception as e:
                raise FileNotFoundError(f"Could not load model locally or from MLflow. Error: {e}. Please train the model first.") from e

    else:
        model_to_evaluate = model
        device = next(model.parameters()).device

    if model_to_evaluate is None:
         raise RuntimeError("Failed to obtain a model for evaluation.")

    accuracy, confusion_matrix = evaluate_model(model_to_evaluate, test_loader, device)
    
    print("\n" + "="*50)
    final_acc_display = best_test_acc if not args.skip_training else accuracy
    print(f"Pipeline completed! Final accuracy: {final_acc_display:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Digit Classification Pipeline")
    
    # Data parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval during training")
    parser.add_argument("--skip_training", action="store_true", help="Skip training and load best model for evaluation")
    
    # Model parameters
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save/load local models")
    
    # MLflow parameters
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking server URI (optional)")
    parser.add_argument("--mlflow_experiment_name", type=str, default="MNIST_Training", help="MLflow experiment name")

    # Visualization
    parser.add_argument("--visualize", action="store_true", help="Visualize data samples")
    
    args = parser.parse_args()
    main(args) 