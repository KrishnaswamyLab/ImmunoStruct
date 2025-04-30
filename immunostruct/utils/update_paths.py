import platform
import os

__all__ = ["update_paths"]

def update_paths(config):
    """Update paths with correct ROOT directory."""
    current_platform = platform.system()
    
    if current_platform == 'Windows':
        ROOT_DIR = "\\".join(os.path.realpath(__file__).split("\\")[:-2])
        separator = "\\"
    elif current_platform == 'Linux':
        ROOT_DIR = "/".join(os.path.realpath(__file__).split("/")[:-2])
        separator = "/"
    else:
        raise NotImplementedError(f"Unsupported platform: {current_platform}")
    
    for key, value in vars(config).items():
        if isinstance(value, str) and "$ROOT" in value:
            new_value = value.replace("$ROOT", ROOT_DIR)
            if current_platform == 'Windows':
                new_value = new_value.replace("/", "\\")
            setattr(config, key, new_value)
