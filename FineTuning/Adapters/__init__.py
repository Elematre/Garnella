"""
Adapters module for parameter-efficient fine-tuning.

This module provides a factory interface for loading and applying various
parameter-efficient adapters (LoRA-XS, RxR sharing, etc.) to pre-trained models.

Usage:
------
from Adapters import load_adapter

# Load a model
model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large", ...)

# Apply an adapter
model_with_adapter = load_adapter("lora-xs", model, config_dir="./Adapters/config")

# Or with no adapter (baseline)
model_baseline = load_adapter("none", model)
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import PreTrainedModel

from .base import AdapterBase
from .classifier_only import XLMRobertaClassifierOnly
from .lora import XLMRobertaLoRA
from .lora_xs import XLMRobertaLoRAXS
from .rxr_shared import XLMRobertaRxRShared

logger = logging.getLogger(__name__)


def load_adapter(
    adapter_name: str,
    model: PreTrainedModel,
    config_dir: Optional[Union[str, Path]] = None,
) -> PreTrainedModel:
    """
    Load and apply a parameter-efficient adapter to a model.
    
    Factory function that handles instantiation and application of various adapters.
    
    Args:
        adapter_name: Name of the adapter to use. Options:
            - "none": No adapter (baseline, full fine-tuning)
            - "classifier-only": Freeze encoder, train only classifier head
            - "lora": LoRA (low-rank adaptation with random initialization)
            - "lora-xs": LoRA-XS (extended LoRA with SVD support, for later development)
            - "rxr-shared": Custom RxR matrix sharing (placeholder)
        
        model: Pre-trained model to adapt (e.g., xlm-roberta-large).
        
        config_dir: Ignored for LoRA (config is inlined). Used for RxR-shared only.
    
    Returns:
        The adapted model, ready for training. If adapter_name is "none",
        returns the original model unchanged.
    
    Raises:
        ValueError: If adapter_name is not recognized.
        FileNotFoundError: If config_dir is specified but config file not found.
    
    Examples:
        >>> model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-large")
        >>> model = load_adapter("lora", model)
        >>> # Now model has LoRA applied, ready for training
        
        >>> model = load_adapter("none", model)  # Baseline
    """
    
    adapter_name = adapter_name.lower().strip()
    
    logger.info(f"Loading adapter: {adapter_name}")
    logger.info(f"Model: {model.__class__.__name__}")
    
    # Handle "none" adapter (no modification)
    if adapter_name == "none":
        logger.info("No adapter applied (baseline, full fine-tuning)")
        return model
    
    # Instantiate and apply adapter
    if adapter_name == "classifier-only":
        adapter = XLMRobertaClassifierOnly()
        model = adapter.apply(model)
        logger.info("Successfully applied Classifier-Only adapter")
    
    elif adapter_name == "lora":
        adapter = XLMRobertaLoRA()
        model = adapter.apply(model)
        logger.info("Successfully applied LoRA adapter")
    
    elif adapter_name == "lora-xs":
        adapter = XLMRobertaLoRAXS()
        model = adapter.apply(model)
        logger.info("Successfully applied LoRA-XS adapter")
    
    elif adapter_name == "rxr-shared":
        config_path = None
        if config_dir is not None:
            config_dir = Path(config_dir)
            config_path = config_dir / "rxr_shared_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(
                    f"Config file not found for adapter '{adapter_name}': {config_path}\n"
                    f"Searched in: {config_dir}"
                )
        adapter = XLMRobertaRxRShared(config_path=str(config_path) if config_path else None)
        model = adapter.apply(model)
        logger.info("Successfully applied RxR-shared adapter")
    
    else:
        raise ValueError(
            f"Unknown adapter: '{adapter_name}'\n"
            f"Supported adapters: 'none', 'classifier-only', 'lora', 'lora-xs', 'rxr-shared'"
        )
    
    return model


__all__ = [
    "load_adapter",
    "AdapterBase",
    "XLMRobertaClassifierOnly",
    "XLMRobertaLoRA",
    "XLMRobertaLoRAXS",
    "XLMRobertaRxRShared",
]
