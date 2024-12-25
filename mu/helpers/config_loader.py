import yaml
import os 


def load_config(yaml_path):
    """Loads the configuration from a YAML file."""
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    return {}
