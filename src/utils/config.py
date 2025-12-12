"""
Configuration Management
Load and save YAML configuration files.
"""
import yaml
from typing import Dict, Any
import os


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output path
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two configurations, with override taking precedence.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


if __name__ == "__main__":
    # Test config loading
    config = {
        'model': {
            'd_model': 512,
            'n_heads': 8
        },
        'training': {
            'batch_size': 32
        }
    }
    
    save_config(config, '/tmp/test_config.yaml')
    loaded = load_config('/tmp/test_config.yaml')
    print(f"Loaded config: {loaded}")
