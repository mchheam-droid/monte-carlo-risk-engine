import numpy as np
from dataclasses import dataclass

@dataclass
class RiskReport:
    """Market risk metrics report container."""
    var_95: float
    var_99: float
    var_999: float
    cvar_95: float
    cvar_99: float
    cvar_999: float
    expected_pnl: float
    worst_case_loss: float
    pnl_samples: np.ndarray

class RiskMetrics:
    """Quantitative risk analytics suite (VaR & Expected Shortfall)."""

    @staticmethod
    def compute_pnl(initial_value: float, terminal_values: np.ndarray) -> np.ndarray:
        """Calculates portfolio Profit and Loss (PnL)."""
        return np.asarray(terminal_values, dtype=np.float64) - initial_value

    @staticmethod
    def compute_var(losses: np.ndarray, alpha: float = 0.99) -> float:
        """Computes Value at Risk (VaR) at confidence level alpha."""
        losses = np.asarray(losses, dtype=np.float64)
        return float(np.percentile(losses, alpha * 100))

    @staticmethod
    def compute_cvar(losses: np.ndarray, alpha: float = 0.99) -> float:
        """Computes Conditional VaR (Expected Shortfall / tail risk) beyond VaR."""
        losses = np.asarray(losses, dtype=np.float64)
        var = RiskMetrics.compute_var(losses, alpha)
        tail_losses = losses[losses >= var]
        return float(np.mean(tail_losses))

    @classmethod
    def evaluate(cls, initial_value: float, terminal_values: np.ndarray) -> RiskReport:
        """Generates full portfolio risk evaluation report."""
        pnl = cls.compute_pnl(initial_value, terminal_values)
        losses = -pnl  # Loss is defined as negative PnL

        return RiskReport(
            var_95=cls.compute_var(losses, 0.95),
            var_99=cls.compute_var(losses, 0.99),
            var_999=cls.compute_var(losses, 0.999),
            cvar_95=cls.compute_cvar(losses, 0.95),
            cvar_99=cls.compute_cvar(losses, 0.99),
            cvar_999=cls.compute_cvar(losses, 0.999),
            expected_pnl=float(np.mean(pnl)),
            worst_case_loss=float(np.max(losses)),
            pnl_samples=pnl
        )