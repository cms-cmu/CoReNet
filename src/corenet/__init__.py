"""CoReNet: the Combinatorial Residual Network, configured by a decay-chain description in YAML."""

from .models.config import ConfigError, NetworkSpec, load_config, parse_config
from .models.network import CoReNet

__version__ = "0.1.0"

__all__ = ["CoReNet", "NetworkSpec", "ConfigError", "load_config", "parse_config", "__version__"]
