"""
classifier_only.py
------------------
Classifier-only fine-tuning adapter.

Freezes the entire pretrained model and only trains the classification head.
This is the baseline approach for parameter-efficient fine-tuning.

Features:
- Freezes all encoder parameters
- Only trains the classifier layer(s)
- Minimal parameter overhead (~2-10K trainable params for large models)
- ~99.9% parameter reduction compared to full fine-tuning
"""

import logging
from typing import Optional

from transformers import PreTrainedModel

from .base import AdapterBase

logger = logging.getLogger(__name__)


class XLMRobertaClassifierOnly(AdapterBase):
    """
    Classifier-only fine-tuning for XLM-RoBERTa models.
    
    Freezes the entire pretrained encoder and only trains the classification head.
    This is the baseline approach for parameter-efficient fine-tuning.
    
    Features:
    - Freezes all encoder parameters
    - Only trains classifier layer(s)
    - Minimal parameter overhead (~2-10K for large models)
    - ~99.9% parameter reduction compared to full fine-tuning
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Classifier-Only adapter.
        
        Args:
            config_path: Ignored. No config needed for this adapter.
        """
        super().__init__(adapter_name="classifier_only", config_path=None)
        logger.info("Classifier-Only adapter: freezing encoder, training only classifier")
    
    def apply(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply classifier-only fine-tuning to the model.
        
        Steps:
        1. Freeze all model parameters
        2. Unfreeze classifier layer(s)
        3. Log parameter statistics
        
        Args:
            model: Pre-trained XLM-RoBERTa model.
        
        Returns:
            Model with frozen encoder and trainable classifier.
        """
        
        logger.info(f"Applying Classifier-Only to model: {model.__class__.__name__}")
        
        # Step 1: Freeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = False
        
        # Step 2: Unfreeze classifier layer(s)
        for name, param in model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
        
        logger.info("Model frozen except for classifier layer")
        
        # Step 3: Log final parameter statistics
        self.log_param_stats(model)
        
        return model
    
    def __repr__(self) -> str:
        return "XLMRobertaClassifierOnly()"
