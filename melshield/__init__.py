"""MelShield reproduction package."""

from .mel import MelConfig, MelFrontend, NormalizationStats
from .relmel import RelMelConfig, RelMelMark, RelMelReference, RelMelResult
from .watermark import ExtractionResult, MelShield, MelShieldConfig, ReferenceRecord

__all__ = [
    "ExtractionResult",
    "MelConfig",
    "MelFrontend",
    "MelShield",
    "MelShieldConfig",
    "NormalizationStats",
    "ReferenceRecord",
    "RelMelConfig",
    "RelMelMark",
    "RelMelReference",
    "RelMelResult",
]
