#!/usr/bin/env python3
"""
ExperimentBaseline.py

Unified entry point for running baseline sentiment classification experiments.

Usage:
    python ExperimentBaseline.py --model BoW_LogReg
    python ExperimentBaseline.py --model BoW_LogReg --output results/my_results.csv
    sbatch ExperimentBaseline.py --model BoW_LogReg

Available models:
    - BoW_LogReg: Bag of Words + Logistic Regression
"""

import argparse
import sys
from pathlib import Path

# Add repo root and Baselines to path for imports
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))

from baselineModels.bow_logreg import BoWLogReg
from baselineModels.gemma_logreg import GemmaLogReg
from baselineModels.gemma_mlp_noEv import GemmaMLP_NoEV
from baselineModels.gemma_xgboost import GemmaXGBoost
from baselineModels.gemma_mlp_ev import GemmaMLP_EV
from utils.csv_logger import log_results_to_csv


# Registry mapping model name -> class
BASELINE_REGISTRY = {
    "BoW_LogReg": BoWLogReg,
    "Gemma_LogReg": GemmaLogReg,
    "Gemma_MLP_NoEV": GemmaMLP_NoEV,
    "Gemma_XGBoost": GemmaXGBoost,
    "Gemma_MLP_EV": GemmaMLP_EV,
}


def get_baseline(name: str):
    """
    Retrieve baseline class by name.

    Args:
        name: Baseline model name (e.g., "BoW_LogReg")

    Returns:
        Baseline class (not instantiated)

    Raises:
        KeyError: If baseline name not found in registry
    """
    if name not in BASELINE_REGISTRY:
        available = ", ".join(BASELINE_REGISTRY.keys())
        raise KeyError(
            f"Baseline '{name}' not found.\n"
            f"Available baselines: {available}"
        )
    return BASELINE_REGISTRY[name]


def list_available_baselines() -> list:
    """
    Get list of available baseline names.

    Returns:
        List of baseline names
    """
    return list(BASELINE_REGISTRY.keys())


def main():
    # define parser with detailed help and examples
    parser = argparse.ArgumentParser(
        description="Run sentiment classification baseline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
            Examples:
            # Run BoW_LogReg baseline
            python ExperimentBaseline.py --model Gemma_LogReg
            
            # Run best baseline (Gemma_MLP_EV)
            python ExperimentBaseline.py --model Gemma_MLP_EV
            
            # Run with custom output path
            python ExperimentBaseline.py --model Gemma_LogReg --output results/my_results.csv
            
            # Submit via sbatch
            sbatch ExperimentBaseline.py --model Gemma_MLP_NoEV

            Available models:
            {', '.join(list_available_baselines())}
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Baseline model name (e.g., BoW_LogReg)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_results.csv",
        help="Path to output CSV file (default: results/baseline_results.csv)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        default="/cluster/courses/cil/text-classification/data/train.csv",
        help="Path to training data CSV (default: data/train.csv)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SENTIMENT CLASSIFICATION BASELINE EXPERIMENT")
    print("=" * 70)
    print(f"Model:       {args.model}")
    print(f"Data:        {args.data}")
    print(f"Output:      {args.output}")
    print("=" * 70)
    
    # Get baseline class and instantiate
    try:
        BaselineClass = get_baseline(args.model)
    except KeyError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nAvailable models: {', '.join(list_available_baselines())}")
        sys.exit(1)
    
    baseline = BaselineClass()

    # preprocess
    print(f"\n▶ Loading and preprocessing data...")
    try:
        baseline.load_and_preprocess_data(args.data)
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        sys.exit(1)
    
    # train
    print(f"\n▶ Training model...")
    try:
        baseline.train()
    except Exception as e:
        print(f"❌ Error during training: {e}")
        sys.exit(1)
    
    # evaluate
    print(f"\n▶ Evaluating model...")
    try:
        results = baseline.evaluate()
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)
    
    # log results
    print(f"\n▶ Logging results...")
    try:
        # Extract key metrics for CSV
        include_metrics = {
            "train_mae": results["train_mae"],
            "train_accuracy": results["train_accuracy"],
            "val_mae": results["val_mae"],
            "val_accuracy": results["val_accuracy"],
        }
        
        log_results_to_csv(
            model_name=args.model,
            train_score=results["train_score"],
            val_score=results["val_score"],
            csv_path=args.output,
            include_metrics=include_metrics
        )
    except Exception as e:
        print(f"❌ Error logging results: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Model:        {args.model}")
    print(f"Train Score:  {results['train_score']:.4f}")
    print(f"Val Score:    {results['val_score']:.4f}")
    print(f"Results:      {args.output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()