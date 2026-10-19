import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class Portfolio:
    tickers: List[str]
    spot_prices: np.ndarray
    weights: np.ndarray
    volatilities: np.ndarray
    drifts: np.ndarray
    correlation_matrix: np.ndarray
    def __post_init__(self):
        self.spot_prices = np.asarray(self.spot_prices,dtype=np.float64)
        self.weights = np.asarray(self.weights,dtype=np.float64)
        self.volatilities = np.asarray(self.volatilities,dtype=np.float64)
        self.drifts = np.asarray(self.drifts,dtype=np.float64)
        self.correlation_matrix = np.asarray(self.correlation_matrix, dtype=np.float64)
        n=len(self.tickers)
        if not (len(self.spot_prices) == len(self.weights) == len(self.volatilities) == len(self.drifts) == n):
            raise ValueError ("all input arrays must have the same length as tickers")
        if self.correlation_matrix.shape != (n,n):
            raise ValueError("correlation matrix must be a square matrix of shape (n,n)")
        total_weight = np.sum(self.weights)
        if not np.isclose(total_weight, 1.0, atol=1e-4):
            raise ValueError("weights must sum to 1")
        if np.any(self.spot_prices <=0):
            raise ValueError("spot prices must be positive")
        if np.any(self.volatilities < 0):
            raise ValueError("volatilities must be non-negative")
        if not np.allclose(np.diag(self.correlation_matrix), 1.0):
            raise ValueError("diagonal elements of correlation matrix must be 1")
        if not np.allclose(self.correlation_matrix,self.correlation_matrix.T):
            raise ValueError("Correlation matrix must be symmetric")
    @property
    def total_value(self):
        return float(np.sum(self.weights * self.spot_prices))
    @property
    def num_assets(self) :
        return len(self.tickers)
    