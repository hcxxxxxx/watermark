"""MelShield reproduction package."""

from .mel import MelConfig, MelFrontend, NormalizationStats
from .watermark import ExtractionResult, MelShield, MelShieldConfig, ReferenceRecord

__all__ = [
    "ExtractionResult",
    "MelConfig",
    "MelFrontend",
    "MelShield",
    "MelShieldConfig",
    "NormalizationStats",
    "ReferenceRecord",
]
