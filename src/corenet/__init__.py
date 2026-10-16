"""CoReNet: the Combinatorial Residual Network, configured by a decay-chain description in YAML."""

from .checkpoint import load_checkpoint, save_checkpoint
from .models.config import ConfigError, NetworkSpec, load_config, parse_config
from .models.network import CoReNet

__version__ = "0.1.0"

__all__ = [
    "CoReNet",
    "ConfigError",
    "NetworkSpec",
    "__version__",
    "load_checkpoint",
    "load_config",
    "parse_config",
    "save_checkpoint",
]
