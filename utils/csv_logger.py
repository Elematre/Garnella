"""
CSV logging utility for experiment results.

Handles appending results to CSV file with proper formatting.
"""

import os
import pandas as pd
from datetime import datetime


def log_results_to_csv(
    model_name: str,
    train_score: float,
    val_score: float,
    csv_path: str = "results/baseline_results.csv",
    include_metrics: dict = None
):
    """
    Append experiment results to CSV file.
    
    If file doesn't exist, create it with headers.
    If file exists, append new row.
    
    Args:
        model_name: Name of the baseline model
        train_score: Training score (1.0 - MAE/4)
        val_score: Validation score (1.0 - MAE/4)
        csv_path: Path to CSV file (default: results/baseline_results.csv)
        include_metrics: Optional dict of additional metrics to log
                        (e.g., {"train_mae": 0.43, "val_accuracy": 0.62})
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.dirname(csv_path)
    if results_dir and not os.path.exists(results_dir):
        os.makedirs(results_dir, exist_ok=True)
    
    # Prepare row data
    now = datetime.now()
    row = {
        "model_name": model_name,
        "train_score": train_score,
        "val_score": val_score,
        "run_date": now.strftime("%Y-%m-%d"),
        "run_time": now.strftime("%H:%M:%S"),
        "timestamp": now.isoformat()
    }
    
    # Add any extra metrics
    if include_metrics:
        row.update(include_metrics)
    
    # Create or append to CSV
    if os.path.exists(csv_path):
        # File exists: read, append, write back
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        # File doesn't exist: create new dataframe
        df = pd.DataFrame([row])
    
    # Write to CSV
    df.to_csv(csv_path, index=False)
    print(f"✓ Results logged to {csv_path}")
