"""
configs/ - YAML configuration files and loader for VocalCoach.

Quick start:
    from configs.loader import load_model_config
    cfg = load_model_config("pitch")   # system.yaml + pitch.yaml merged
"""

from configs.loader import load_config, load_model_config, load_preprocessing_config, merge_configs

__all__ = ["load_config", "load_model_config", "load_preprocessing_config", "merge_configs"]
