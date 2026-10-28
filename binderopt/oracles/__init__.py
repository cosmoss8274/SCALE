"""Structure-oracle backends for the SCALE loop."""

from .registry import ORACLES, get_oracle, register_oracle
from .replay import ReplayOracle

__all__ = ["get_oracle", "register_oracle", "ORACLES", "ReplayOracle"]
