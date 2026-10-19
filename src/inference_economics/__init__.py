"""Measure what a model's accuracy costs to serve.

An accuracy number ranks models on one axis and hides the other. This package produces
both in one run, because measuring them separately is how a team ends up deciding from
whichever table was on screen.
"""

from inference_economics.config import BenchConfig, ConfigError

__all__ = ["BenchConfig", "ConfigError"]
__version__ = "0.1.0"
