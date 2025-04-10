#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter tuning script using Optuna and MLflow.
"""

import optuna
import mlflow
import torch
import argparse
from pathlib import Path

from src.data_loader import load_mnist_data
from src.model import MNISTClassifier
from src.train import train_model

def objective(trial, args):
    """
    Optuna objective function.
    Trains a model with hyperparameters suggested by Optuna,
    logs the run with MLflow, and returns the metric to optimize (test accuracy).
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.5, 0.99)
    # Example: Tune number of epochs as well
    # epochs = trial.suggest_int("epochs", 3, 10) 
    epochs = args.epochs # Keep epochs fixed for this example or tune as above

    # Load data
    train_loader, test_loader = load_mnist_data(batch_size=args.batch_size)
    
    # Create model
    model = MNISTClassifier()

    # Setup MLflow experiment for this trial
    # Naming convention for nested runs
    parent_run = mlflow.active_run()
    parent_run_id = parent_run.info.run_id if parent_run else None
    experiment_name = f"{args.mlflow_experiment_name}_Tuning"
    
    # Train model with suggested hyperparameters, MLflow logs within train_model
    try:
        model, best_test_acc = train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            log_interval=args.log_interval,
            save_dir=args.model_dir, # Local save dir (optional)
            mlflow_tracking_uri=args.mlflow_tracking_uri,
            mlflow_experiment_name=experiment_name # Log trial to tuning experiment
        )
        # Add Optuna trial info as tags to the MLflow run
        current_run = mlflow.active_run()
        if current_run:
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("optuna_trial_state", trial.state.name)
            if parent_run_id:
                mlflow.set_tag("mlflow.parentRunId", parent_run_id)

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Log failure and return a poor value to Optuna
        current_run = mlflow.active_run()
        if current_run:
            mlflow.set_tag("optuna_trial_number", trial.number)
            mlflow.set_tag("optuna_trial_state", "FAILED")
            if parent_run_id:
                 mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            mlflow.log_param("error", str(e))
        return 0.0 # Indicate failure to Optuna

    # Return the metric Optuna should maximize
    return best_test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for MNIST Classifier")

    # Optuna parameters
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--optuna_study_name", type=str, default="mnist_tuning_study", help="Optuna study name")
    parser.add_argument("--optuna_storage", type=str, default="sqlite:///optuna_mnist.db", help="Optuna storage URI (e.g., sqlite:///study.db)")

    # Inherit relevant parameters from main.py (or redefine defaults)
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per trial")
    parser.add_argument("--log_interval", type=int, default=500, help="Logging interval during training (less frequent for tuning)")
    parser.add_argument("--model_dir", type=str, default="models/tuning", help="Directory for tuning artifacts")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None, help="MLflow tracking server URI")
    parser.add_argument("--mlflow_experiment_name", type=str, default="MNIST_Hyperparameter_Tuning", help="Base MLflow experiment name for tuning")

    args = parser.parse_args()

    # Create Optuna study
    # Direction is maximize because we return test accuracy from objective
    study = optuna.create_study(
        study_name=args.optuna_study_name,
        storage=args.optuna_storage,
        load_if_exists=True, # Resume study if it already exists
        direction="maximize"
    )

    # Setup base MLflow experiment for the overall tuning process
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)

    # Start a parent MLflow run for the entire Optuna study
    with mlflow.start_run(run_name="Optuna_Tuning_Study") as parent_run:
        mlflow.log_param("n_trials", args.n_trials)
        mlflow.log_param("optuna_study_name", args.optuna_study_name)
        mlflow.log_param("optuna_storage", args.optuna_storage)
        
        print(f"Starting Optuna study: {args.optuna_study_name}")
        print(f"MLflow Parent Run ID: {parent_run.info.run_id}")
        print(f"Optuna Storage: {args.optuna_storage}")

        # Run Optuna optimization
        # We pass args to the objective using a lambda function
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

        # Log best trial results to the parent MLflow run
        best_trial = study.best_trial
        print("\nStudy statistics: ")
        print(f"  Number of finished trials: {len(study.trials)}")
        print(f"  Best trial ({best_trial.number}):")
        print(f"    Value (Best Test Accuracy): {best_trial.value:.4f}")
        print("    Params: ")
        for key, value in best_trial.params.items():
            print(f"      {key}: {value}")
            mlflow.log_param(f"best_trial_{key}", value)
        mlflow.log_metric("best_trial_accuracy", best_trial.value)
        mlflow.set_tag("best_trial_number", best_trial.number)

    print(f"\nOptuna study complete. Best trial logged under MLflow Parent Run ID: {parent_run.info.run_id}")
    print(f"Access MLflow UI to see results (default: run 'mlflow ui' in the terminal in the project directory)")
    print(f"Optuna study data saved to: {args.optuna_storage}") 