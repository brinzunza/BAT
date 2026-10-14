"""
Bayesian Regime-Switching Model

A Bayesian approach to detecting market regime changes using Hidden Markov Models.
"""

from .model import BayesianRegimeSwitchingModel
from .inference import BayesianInference
from .data_loader import MarketDataLoader

__version__ = "0.1.0"
__all__ = [
    "BayesianRegimeSwitchingModel",
    "BayesianInference",
    "MarketDataLoader",
]
