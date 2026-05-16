"""
lora.py
-------
LoRA adapter implementation using PEFT library.

LoRA applies low-rank decomposition to reduce parameters during fine-tuning:
1. Low-rank decomposition via PEFT's LoraConfig
2. Randomly initialized A/B matrices
3. Minimal parameter overhead (~1-5M for large models)

Reference: https://arxiv.org/abs/2405.17604
"""

import logging
from typing import Optional, Any, Dict
from xml.parsers.expat import model

import torch
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

from .base import AdapterBase

logger = logging.getLogger(__name__)


class XLMRobertaLoRA(AdapterBase):
    """
    LoRA adapter for XLM-RoBERTa models.
    
    Applies LoRA with randomly initialized low-rank matrices.
    
    Features:
    - Random low-rank initialization (A, B)
    - Minimal parameter overhead (~1-5M for large models)
    - ~100x parameter reduction compared to full fine-tuning
    """
    
    # Default configuration inlined
    DEFAULT_CONFIG = {
        "lora_rank": 16,
        "target_modules": ["query", "key", "value", "dense"],
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "SEQ_CLS",
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LoRA adapter.
        
        Args:
            config_path: Ignored. Config is inlined in DEFAULT_CONFIG.
        """
        super().__init__(adapter_name="lora", config_path=None)
        logger.info("LoRA adapter using inlined default configuration")
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply LoRA adapter to the model.
        
        Steps:
        1. Create PEFT LoRA config
        2. Wrap model with get_peft_model (uses random initialization)
        
        Args:
            model: Pre-trained XLM-RoBERTa model.
        
        Returns:
            Model with LoRA applied, ready for training.
        """
        
        logger.info(f"Applying LoRA to model: {model.__class__.__name__}")
        
        config = self.DEFAULT_CONFIG
        lora_rank = config["lora_rank"]
        target_modules = config["target_modules"]
        
        logger.info(f"LoRA rank: {lora_rank}")
        logger.info(f"Target modules: {target_modules}")
        
        # Create PEFT LoRA config
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=config["lora_alpha"],
            target_modules=target_modules,
            lora_dropout=config["lora_dropout"],
            bias=config["bias"],
            task_type=config["task_type"],
        )
        
        # Wrap model with PEFT (uses random initialization by default)
        model = get_peft_model(model, lora_config)
        logger.info("Model wrapped with PEFT LoRA config")
        
        # Always unfreeze the classifier when using PEFT for classification
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True

        # Log final parameter statistics
        self.log_param_stats(model)
        
        return model
    
    def __repr__(self) -> str:
        return f"XLMRobertaLoRA(rank={self.DEFAULT_CONFIG['lora_rank']})"
