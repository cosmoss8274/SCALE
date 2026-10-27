"""binderopt: a clean, single-pipeline protein-binder optimization library.

This package replaces five near-duplicate optimization pipelines with one
parameterized implementation. Target-specific behavior
(sequence, paths, weights, iteration counts) lives entirely in
``scale/configs/`` YAML files; the code under ``binderopt/`` is target-agnostic.
"""

__version__ = "0.1.0"
