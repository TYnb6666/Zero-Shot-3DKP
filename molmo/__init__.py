"""Vendored Molmo model code (allenai/Molmo-7B-D-0924, commit cab33fb).

Importing this package registers all four Molmo classes with the transformers
Auto machinery so they load correctly with trust_remote_code=False.
"""
from .config_molmo import MolmoConfig
from .image_preprocessing_molmo import MolmoImageProcessor
from .preprocessing_molmo import MolmoProcessor
from .modeling_molmo import MolmoForCausalLM
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
)

AutoConfig.register("molmo", MolmoConfig, exist_ok=True)
AutoImageProcessor.register(MolmoConfig, MolmoImageProcessor, exist_ok=True)
AutoProcessor.register(MolmoConfig, MolmoProcessor, exist_ok=True)
AutoModelForCausalLM.register(MolmoConfig, MolmoForCausalLM, exist_ok=True)

__all__ = ["MolmoConfig", "MolmoForCausalLM", "MolmoImageProcessor", "MolmoProcessor"]
