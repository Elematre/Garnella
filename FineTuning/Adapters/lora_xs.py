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
from typing import Optional, Dict, Any, List, Tuple
import math
import numpy as np

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from transformers import PreTrainedModel

from .base import AdapterBase
from .utils.svd_init import get_frozen_ab_matrices, get_adaptive_frozen_ab_matrices

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
        adaptive: bool = False,
        threshold: float = 0.90,
        min_rank: int = 2,
    ):
        """
        Initialize LoRA-XS wrapper.
        
        Args:
            lora_layer: The original PEFT LoRA layer to wrap
            lora_rank: Rank of the LoRA decomposition (used as max_rank if adaptive)
            adaptive: Whether to use adaptive rank selection
            threshold: Cumulative singular value threshold for adaptive rank
            min_rank: Minimum rank for adaptive rank
        """
        super().__init__()
        self.lora_layer = lora_layer
        self.adaptive = adaptive
        
        # Extract A and B from the LoRA layer
        self.lora_A = lora_layer.lora_A["default"]
        self.lora_B = lora_layer.lora_B["default"]
        
        # Inject SVD-based initialization
        # weight is (out_features, in_features)
        base_weight = self.lora_layer.base_layer.weight.data.clone().detach().cpu().numpy()
        

        # Get alpha and scale factor from original LoRA layer
        if isinstance(lora_layer.lora_alpha, dict):
            self.lora_alpha = lora_layer.lora_alpha.get("default", lora_rank) # peft original behavior uses max rank for default scaling
        else:
            self.lora_alpha = lora_layer.lora_alpha

        # Resolve rank and get matrices — unified path
        if self.adaptive:
            A_tensor, B_tensor, self.lora_rank = get_adaptive_frozen_ab_matrices(
                weight=base_weight, max_rank=lora_rank,
                threshold=threshold, min_rank=min_rank
            )
            self.scaling = 1
        else:
            A_tensor, B_tensor = get_frozen_ab_matrices(weight=base_weight, rank=lora_rank)
            self.lora_rank = lora_rank
            self.scaling = self.lora_alpha / self.lora_rank

        # Always resize and populate A/B weights with the actual SVD results
        dev, dtype = self.lora_A.weight.device, self.lora_A.weight.dtype

        self.lora_A.weight = nn.Parameter(B_tensor.to(device=dev, dtype=dtype))  # (k, in_features)
        self.lora_B.weight = nn.Parameter(A_tensor.to(device=dev, dtype=dtype))  # (out_features, k)

        self.lora_A.out_features = self.lora_rank
        self.lora_B.in_features = self.lora_rank
        # Now freeze A and B
        self.lora_A.weight.requires_grad_(False)
        self.lora_B.weight.requires_grad_(False)
        
        # Create trainable R matrix (rank x rank) and initialize to zero!!!
        r_matrix = nn.Parameter(torch.zeros(self.lora_rank, self.lora_rank, device=self.lora_B.weight.device))
        self.register_parameter("lora_r", r_matrix)
        
        # Copy over other attributes for compatibility
        self.weight = lora_layer.weight  # Original weight reference
        self.in_features = lora_layer.in_features
        self.out_features = lora_layer.out_features
        self.bias = lora_layer.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute the clean, frozen base model output directly
        result = self.lora_layer.base_layer(x)
        
        x_A = torch.nn.functional.linear(x, self.lora_A.weight)
        x_AR = torch.nn.functional.linear(x_A, self.lora_r)
        x_ARB = torch.nn.functional.linear(x_AR, self.lora_B.weight)
    
        return result + x_ARB * self.scaling
    
    @classmethod
    def from_lora_linear(
        cls,
        lora_layer: LoraLinear,
        lora_rank: int,
        adaptive: bool = False,
        threshold: float = 0.90,
        min_rank: int = 2,
    ) -> "LoraXSLinear":
        """
        Create LoRA-XS wrapper from a PEFT LoRA layer.
        
        Args:
            lora_layer: The PEFT LoRA layer to wrap
            lora_rank: Rank of LoRA decomposition
            adaptive: Whether to use adaptive rank selection
            threshold: Cumulative singular value threshold
            min_rank: Minimum rank for adaptive rank
            
        Returns:
            LoraXSLinear wrapper instance
        """
        return cls(lora_layer, lora_rank, adaptive, threshold, min_rank)


def _replace_with_lora_xs(
    model: PreTrainedModel,
    lora_rank: int,
    adaptive: bool = False,
    threshold: float = 0.90,
    min_rank: int = 2,
) -> Tuple[int, List[int]]:
    """
    Replace PEFT LoRA layers with LoRA-XS wrappers.
    
    Traverses the model tree, finds LoRA linear layers, and wraps them
    with LoraXSLinear to inject trainable R matrices.
    
    Returns:
        replaced_count: Number of LoRA layers replaced
        ranks: List of ranks chosen for each layer
    """
    replaced_count = 0
    ranks = []
    
    # Traverse model to find and replace LoRA layers
    # Use list() to avoid modifying the module structure while iterating
    modules_to_replace = []
    
    # visits every node
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            # visits every child node of the current parent node
            if isinstance(child_module, LoraLinear):
                modules_to_replace.append((parent_module, child_name, child_module))
                
    for parent_module, child_name, child_module in modules_to_replace:
        # Create LoRA-XS wrapper
        xs_layer = LoraXSLinear.from_lora_linear(
            child_module,
            lora_rank,
            adaptive=adaptive,
            threshold=threshold,
            min_rank=min_rank,
        )
        
        # Replace the module in the parent
        setattr(parent_module, child_name, xs_layer)
        replaced_count += 1
        ranks.append(xs_layer.lora_rank)
        logger.debug(f"Replaced LoRA layer: {child_name} with rank {xs_layer.lora_rank}")
    
    return replaced_count, ranks


class XLMRobertaLoRAXS(AdapterBase):
    """
    LoRA-XS adapter for XLM-RoBERTa models.
    
    Applies LoRA with randomly initialized A/B matrices and trainable R x R matrix.
        
    Features:
    - Random low-rank initialization (A, B)
    - Trainable intermediate matrix (R) with kaiming/normal initialization
    - Minimal parameter overhead (~1-5M for large models)
    """
    
    # Default configuration inlined
    DEFAULT_CONFIG = {
        "lora_rank": 32,
        "target_modules": ["query", "key", "value", "dense"],
        "adaptiveRank": False,
        "criterionThreshold": 0.90,
        "minRank": 2,
        "maxRank": 128,
        "peft": {
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "SEQ_CLS",
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize LoRA-XS adapter.
        
        Args:
            config_path: Path to lora_xs_config.yaml. If None, tries to load from 
                         Adapters/config/lora_xs_config.yaml, otherwise uses hardcoded defaults.
        """
        # Try to load from default config location if not specified
        if config_path is None:
            from pathlib import Path
            adapter_dir = Path(__file__).parent
            default_config_path = adapter_dir / "config" / "lora_xs_config.yaml"
            
            if default_config_path.exists():
                config_path = str(default_config_path)
                logger.info(f"Found default config at {default_config_path}, using it")
            else:
                logger.debug(f"No config file at {default_config_path}, will use hardcoded defaults")
        
        super().__init__(adapter_name="lora_xs", config_path=config_path)
        
        # Merge loaded config with defaults for any missing keys
        if not self.config:
            logger.info("No config loaded, using hardcoded defaults")
            self.config = self.DEFAULT_CONFIG.copy()
        else:
            # Ensure all required keys exist by merging with defaults
            merged_config = self.DEFAULT_CONFIG.copy()
            merged_config.update(self.config)
            self.config = merged_config
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get config value with fallback to default."""
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

        adaptive_rank = self._get_config_value("adaptiveRank", False)
        threshold = self._get_config_value("criterionThreshold", 0.90)
        min_rank = self._get_config_value("minRank", 2)
        base_lora_rank = self._get_config_value("lora_rank", 16)
        
        if adaptive_rank:
            lora_rank = self._get_config_value("maxRank", 128)
            logger.info(f"Adaptive rank enabled: Threshold={threshold}, min={min_rank}, max={lora_rank}")
        else:
            lora_rank = base_lora_rank

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

        
        replaced_count, ranks = _replace_with_lora_xs(
            model,
            lora_rank,
            adaptive=adaptive_rank,
            threshold=threshold,
            min_rank=min_rank,
        )
        
        if adaptive_rank and ranks:
            logger.info(
                f"Adaptive Ranks Summary | Average: {np.mean(ranks):.1f}, "
                f"Max: {np.max(ranks)}, Min: {np.min(ranks)}, Std: {np.std(ranks):.2f}"
            )
            
            total_r_params = sum(r * r for r in ranks)
            logger.info(f"Injected {replaced_count} trainable R matrices ({total_r_params} total params)")
        else:
            logger.info(
                f"Injected {replaced_count} trainable R matrices "
                f"({lora_rank}×{lora_rank} each, "
                f"{replaced_count * lora_rank * lora_rank} total params)"
            )

        # Step 3: Log final parameter statistics
        self.log_param_stats(model)

        return model
            

    
    def __repr__(self) -> str:
        return f"XLMRobertaLoRAXS(rank={self._get_config_value('lora_rank', 16)})"
