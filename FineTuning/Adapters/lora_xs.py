"""
lora_xs.py
----------
LoRA-XS adapter implementation using PEFT library.

LoRA-XS applies:
1. Low-rank decomposition via PEFT's LoraConfig
2. Randomly initialized A/B matrices (frozen after initialization)
3. Trainable R x R matrix between frozen components

Result: Parameter efficiency ~100x for large models like xlm-roberta-large.

Reference: https://arxiv.org/abs/2405.17604
"""

import logging
from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from transformers import PreTrainedModel

from .base import AdapterBase

logger = logging.getLogger(__name__)


class LoraXSLinear(nn.Module):
    """
    LoRA-XS linear layer wrapper.
    
    Wraps a PEFT LoRA layer and injects a trainable R matrix between frozen A and B.
    Forward pass: y = original_output + (input @ A @ R @ B) * alpha / rank
    
    Where:
    - A: frozen low-rank matrix (input_dim, rank)
    - R: trainable intermediate matrix (rank, rank)  
    - B: frozen low-rank matrix (rank, output_dim)
    """
    
    def __init__(
        self,
        lora_layer: LoraLinear,
        lora_rank: int,
        r_matrix_init: str = "kaiming",
        r_matrix_std: float = 0.02,
    ):
        """
        Initialize LoRA-XS wrapper.
        
        Args:
            lora_layer: The original PEFT LoRA layer to wrap
            lora_rank: Rank of the LoRA decomposition
            r_matrix_init: Initialization method for R matrix ("kaiming" or "normal")
            r_matrix_std: Standard deviation for normal initialization
        """
        super().__init__()
        self.lora_layer = lora_layer
        self.lora_rank = lora_rank
        
        # Extract A and B from the LoRA layer
        self.lora_A = lora_layer.lora_A["default"]
        self.lora_B = lora_layer.lora_B["default"]
        
        # CRITICAL FIX: PEFT initializes B to zero. If we freeze B at zero, 
        # R will receive 0 gradients! We must initialize B randomly first.
        nn.init.kaiming_uniform_(self.lora_B.weight, a=math.sqrt(5))
        
        # Now freeze A and B
        self.lora_A.weight.requires_grad_(False)
        self.lora_B.weight.requires_grad_(False)
        
        # Create trainable R matrix (rank x rank)
        r_matrix = nn.Parameter(torch.empty(lora_rank, lora_rank))
        if r_matrix_init == "kaiming":
            nn.init.kaiming_uniform_(r_matrix, a=math.sqrt(5))
        else:
            nn.init.normal_(r_matrix, std=r_matrix_std)
        self.register_parameter("lora_r", r_matrix)
        
        # Get alpha and scale factor from original LoRA layer
        if isinstance(lora_layer.lora_alpha, dict):
            self.lora_alpha = lora_layer.lora_alpha.get("default", lora_rank)
        else:
            self.lora_alpha = lora_layer.lora_alpha

        self.scaling = self.lora_alpha / lora_rank
        
        # Copy over other attributes for compatibility
        self.weight = lora_layer.weight  # Original weight reference
        self.in_features = lora_layer.in_features
        self.out_features = lora_layer.out_features
        self.bias = lora_layer.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute the clean, frozen base model output directly
        result = self.lora_layer.base_layer(x)
        
        # 2. Guard rail: bypass adapter tracking if disabled or merged
        if self.lora_layer.disable_adapters or self.lora_layer.merged:
            return result
            
        # 3. Apply the custom LoRA-XS pathway
        x_A = torch.nn.functional.linear(x, self.lora_A.weight)
        x_AR = torch.nn.functional.linear(x_A, self.lora_r)
        x_ARB = torch.nn.functional.linear(x_AR, self.lora_B.weight)
    
        return result + x_ARB * self.scaling
    
    @classmethod
    def from_lora_linear(
        cls,
        lora_layer: LoraLinear,
        lora_rank: int,
        r_matrix_init: str = "kaiming",
        r_matrix_std: float = 0.02,
    ) -> "LoraXSLinear":
        """
        Create LoRA-XS wrapper from a PEFT LoRA layer.
        
        Args:
            lora_layer: The PEFT LoRA layer to wrap
            lora_rank: Rank of LoRA decomposition
            r_matrix_init: Initialization for R matrix
            r_matrix_std: Std dev for normal init
            
        Returns:
            LoraXSLinear wrapper instance
        """
        return cls(lora_layer, lora_rank, r_matrix_init, r_matrix_std)


def _replace_with_lora_xs(
    model: PreTrainedModel,
    lora_rank: int,
    r_matrix_init: str = "kaiming",
    r_matrix_std: float = 0.02,
) -> int:
    """
    Replace PEFT LoRA layers with LoRA-XS wrappers.
    
    Traverses the model tree, finds LoRA linear layers, and wraps them
    with LoraXSLinear to inject trainable R matrices.
    
    Args:
        model: Model with PEFT LoRA applied
        lora_rank: Rank of LoRA decomposition
        r_matrix_init: Initialization method for R matrices
        r_matrix_std: Std dev for normal initialization
        
    Returns:
        Number of LoRA layers replaced
    """
    replaced_count = 0
    
    # Traverse model to find and replace LoRA layers
    # Use list() to avoid modifying the module structure while iterating
    modules_to_replace = []
    
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if isinstance(child_module, LoraLinear):
                modules_to_replace.append((parent_module, child_name, child_module))
                
    for parent_module, child_name, child_module in modules_to_replace:
        # Create LoRA-XS wrapper
        xs_layer = LoraXSLinear.from_lora_linear(
            child_module,
            lora_rank,
            r_matrix_init,
            r_matrix_std,
        )
        
        # Replace the module in the parent
        setattr(parent_module, child_name, xs_layer)
        replaced_count += 1
        logger.debug(f"Replaced LoRA layer: {child_name}")
    
    return replaced_count


class XLMRobertaLoRAXS(AdapterBase):
    """
    LoRA-XS adapter for XLM-RoBERTa models.
    
    Applies LoRA with randomly initialized A/B matrices and trainable R x R matrix.
        
    Features:
    - Random low-rank initialization (A, B)
    - Trainable intermediate matrix (R) with kaiming/normal initialization
    - Minimal parameter overhead (~1-5M for large models)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LoRA-XS adapter.
        
        Args:
            config_path: Path to lora_xs_config.yaml. If None, uses built-in defaults.
        """
        super().__init__(adapter_name="lora_xs", config_path=config_path)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that required config keys are present."""
        required_keys = ["lora_rank", "target_modules"]
        for key in required_keys:
            if key not in self.config:
                logger.warning(f"Missing config key '{key}', will use defaults")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default LoRA-XS configuration if none provided."""
        return {
            "lora_rank": 16,
            "target_modules": ["query", "key", "value", "dense"],
            "peft": {
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "SEQ_CLS",
            },
            "init": {
                "r_matrix_init": "kaiming",
                "r_matrix_std": 0.02,
            },
            "logging": {
                "log_param_stats": True,
            },
        }
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get config value with fallback to default."""
        if not self.config:
            default_config = self._get_default_config()
            return default_config.get(key, default)
        return self.config.get(key, default)
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply LoRA-XS adapter to the model.
        
        Steps:
        1. Wrap model with PEFT LoRA config
        2. Replace PEFT LoRA layers with LoRA-XS wrappers (adds R matrices)
        3. Log parameter statistics
        
        Args:
            model: Pre-trained model to adapt
            
        Returns:
            Model with LoRA-XS applied
        """
        logger.info(f"Applying LoRA-XS to model: {model.__class__.__name__}")

        lora_rank = self._get_config_value("lora_rank", 16)
        target_modules = self._get_config_value("target_modules", ["query", "key", "value", "dense"])
        peft_config = self._get_config_value("peft", {})
        init_config = self._get_config_value("init", {})
        logging_config = self._get_config_value("logging", {})

        # Step 1: Create and apply PEFT LoRA config
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=peft_config.get("lora_alpha", lora_rank),
            target_modules=target_modules,
            lora_dropout=peft_config.get("lora_dropout", 0.1),
            bias=peft_config.get("bias", "none"),
            task_type=peft_config.get("task_type", "SEQ_CLS"),
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Model wrapped with PEFT LoRA (rank={lora_rank})")

        # Step 2: Replace PEFT LoRA layers with LoRA-XS wrappers
        r_matrix_init = init_config.get("r_matrix_init", "kaiming")
        r_matrix_std = init_config.get("r_matrix_std", 0.02)
        
        replaced_count = _replace_with_lora_xs(
            model,
            lora_rank,
            r_matrix_init,
            r_matrix_std,
        )
        
        logger.info(
            f"Injected {replaced_count} trainable R matrices "
            f"({lora_rank}×{lora_rank} each, "
            f"{replaced_count * lora_rank * lora_rank} total params)"
        )

        # Step 3: Log final parameter statistics
        if logging_config.get("log_param_stats", True):
            self.log_param_stats(model)

        return model
            

    
    def __repr__(self) -> str:
        return f"XLMRobertaLoRAXS(rank={self._get_config_value('lora_rank', 16)})"
