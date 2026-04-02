"""
compress_dataset.py — re-exports from geneticinfo for backwards compatibility.

These functions are now part of the geneticinfo module:

    from geneticinfo import compress, generate_synthetic, validate, test_privacy
"""
from geneticinfo import compress, generate_synthetic, validate, test_privacy

__all__ = ["compress", "generate_synthetic", "validate", "test_privacy"]
