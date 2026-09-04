"""Importing this package registers every provider.

To add your own: create a module here, subclass the right stage base class,
decorate with @register, and import it below.
"""

from . import adaptation, mixing, separation, transcription, voice  # noqa: F401
from .base import (  # noqa: F401
    REGISTRY,
    AdaptationProvider,
    Credential,
    MissingCredential,
    MixProvider,
    Option,
    Provider,
    ProviderNotImplemented,
    SeparationProvider,
    Stage,
    TranscriptionProvider,
    VoiceProvider,
    catalogue,
    get,
    register,
)

__all__ = [
    "REGISTRY", "Stage", "Provider", "Credential", "Option", "register", "get", "catalogue",
    "SeparationProvider", "TranscriptionProvider", "AdaptationProvider",
    "VoiceProvider", "MixProvider", "MissingCredential", "ProviderNotImplemented",
]
