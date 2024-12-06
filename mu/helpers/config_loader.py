from omegaconf import OmegaConf
from typing import Any, List

def load_config(config_path: str, overrides: List[str] = None) -> Any:
    """
    Load configuration from a YAML file and apply overrides.

    Args:
        config_path (str): Path to the configuration file.
        overrides (List[str], optional): List of overrides in key=value format. Defaults to None.

    Returns:
        Any: Loaded and overridden configuration.
    """
    config = OmegaConf.load(config_path)
    if overrides:
        for override in overrides:
            key, value = override.split('=')
            OmegaConf.update(config, key, value)
    return config
