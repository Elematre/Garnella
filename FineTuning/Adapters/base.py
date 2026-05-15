"""
base.py
-------
Abstract base class for parameter-efficient fine-tuning adapters.

Each adapter wraps a pre-trained model and applies parameter-efficient modifications
(e.g., LoRA-XS, custom RxR matrix sharing) to reduce trainable parameters while
maintaining or improving performance.

Adapters are model-level wrappers that apply modifications at initialization time
and return a ready-to-train model compatible with HuggingFace Trainer.
"""

from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import yaml

logger = logging.getLogger(__name__)


class AdapterBase(ABC):
    """
    Abstract base class for all parameter-efficient adapters.
    
    Subclasses must implement the apply() method to return a modified model
    with parameter-efficient modifications applied.
    """
    
    def __init__(self, adapter_name: str, config_path: Optional[str] = None):
        """
        Initialize adapter.
        
        Args:
            adapter_name: Name of the adapter (e.g., "lora_xs", "rxr_shared")
            config_path: Path to adapter config file (YAML). If None, use defaults.
        """
        self.adapter_name = adapter_name
        self.config_path = config_path
        self.config: Dict[str, Any] = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load adapter config from YAML file, or return empty dict if not specified."""
        if self.config_path is None:
            logger.info(f"No config file specified for {self.adapter_name}, using defaults")
            return {}
        
        config_path = Path(self.config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded config for {self.adapter_name} from {self.config_path}")
        return config or {}
    
    @abstractmethod
    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply adapter modifications to the model.
        
        Args:
            model: The base pre-trained model to adapt.
            
        Returns:
            The modified model with adapter applied. Should be ready for training.
        """
        pass
    
    def get_trainable_params(self, model: torch.nn.Module) -> int:
        """Count the number of trainable parameters in the adapted model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_total_params(self, model: torch.nn.Module) -> int:
        """Count the total number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    def log_param_stats(self, model: torch.nn.Module) -> None:
        """Log training statistics: trainable vs total parameters."""
        trainable = self.get_trainable_params(model)
        total = self.get_total_params(model)
        ratio = (trainable / total) * 100 if total > 0 else 0
        
        logger.info(f"[{self.adapter_name}] Trainable params: {trainable:,} / {total:,} ({ratio:.2f}%)")
        logger.info(f"[{self.adapter_name}] Parameter reduction: {(total - trainable) / total * 100:.1f}%")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.adapter_name})"
