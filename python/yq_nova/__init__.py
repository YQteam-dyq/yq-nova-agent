"""yq-nova: minimal Python client for the yq-nova Agent memory service."""

from .client import Client, NovaApiError

__all__ = ["Client", "NovaApiError"]
__version__ = "0.1.0"