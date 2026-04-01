"""
compress_dataset.py — re-exports from geninfo for backwards compatibility.

These functions are now part of the geninfo module:

    from geninfo import compress, generate_synthetic, validate, test_privacy
"""
from geninfo import compress, generate_synthetic, validate, test_privacy

__all__ = ["compress", "generate_synthetic", "validate", "test_privacy"]
