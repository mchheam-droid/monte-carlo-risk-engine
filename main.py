import time
import numpy as np
from risk_engine import Portfolio, RiskMetrics
from risk_engine._risk_engine import SimulationEngine, AssetData

def run_risk_pipeline():
    print("=" * 70)
    print(" MONTE CARLO PORTFOLIO RISK ENGINE")
    print(" Quantitative Finance & Computing (C++ / Python)")
    print("=" * 70)

    # 1. Define multi-asset portfolio (5 US Tech Equities)
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    spot_prices = np.array([225.0, 430.0, 125.0, 180.0, 190.0])
    weights = np.array([0.25, 0.25, 0.20, 0.15, 0.15])
    volatilities = np.array([0.22, 0.20, 0.38, 0.24, 0.26]) # Annualized vol
    drifts = np.array([0.08, 0.07, 0.15, 0.08, 0.09])       # Expected annual drift

    # Realistic empirical correlation matrix
    corr_matrix = np.array([
        [1.00, 0.72, 0.65, 0.68, 0.60],
        [0.72, 1.00, 0.62, 0.75, 0.66],
        [0.65, 0.62, 1.00, 0.58, 0.55],
        [0.68, 0.75, 0.58, 1.00, 0.70],
        [0.60, 0.66, 0.55, 0.70, 1.00]
    ])

    portfolio = Portfolio(
        tickers=tickers,
        spot_prices=spot_prices,
        weights=weights,
        volatilities=volatilities,
        drifts=drifts,
        correlation_matrix=corr_matrix
    )

    initial_value = portfolio.total_value
    print(f"\n[Portfolio Initialized] Assets: {portfolio.num_assets} | Base Value: ${initial_value:.2f}")
    for i, sym in enumerate(tickers):
        print(f"  - {sym:<6}: Spot=${spot_prices[i]:>6.2f} | Shares={weights[i]*100:>4.1f}% | Vol={volatilities[i]*100:>4.1f}%")

    # 2. Convert to C++ AssetData format
    cpp_assets = [
        AssetData(spot=spot_prices[i], weight=weights[i], vol=volatilities[i], drift=drifts[i])
        for i in range(len(tickers))
    ]

    # Flatten correlation matrix for C++ 1D linear algebra layout
    flat_corr = corr_matrix.flatten().tolist()

    # 3. Instantiate C++ Simulation Engine (auto-detects hardware threads)
    engine = SimulationEngine(num_threads=0, seed=1337)
    num_paths = 1_000_000
    horizon_days = 1.0  # 1-day risk horizon

    print(f"\n[Monte Carlo Simulation] Running {num_paths:,} correlated paths across CPU cores...")

    # Timed C++ execution
    t_start = time.perf_counter()
    terminal_values = engine.simulate_portfolio_values(
        assets=cpp_assets,
        correlation_matrix=flat_corr,
        num_paths=num_paths,
        horizon_days=horizon_days,
        use_antithetic=True
    )
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    throughput = num_paths / (elapsed_ms / 1000.0)

    print(f"[Engine Benchmark] Completed in {elapsed_ms:.2f} ms ({throughput:,.0f} paths/sec)")

    # 4. Quantitative Market Risk Evaluation
    report = RiskMetrics.evaluate(initial_value, terminal_values)

    print("\n" + "=" * 70)
    print(" MARKET RISK REPORT (1-Day Horizon)")
    print("=" * 70)
    print(f" Portfolio Value      : ${initial_value:,.2f}")
    print(f" Expected PnL (Mean)  : ${report.expected_pnl:,.2f}")
    print(f" Worst-Case Loss      : ${report.worst_case_loss:,.2f}")
    print("-" * 70)
    print(f" Value at Risk (VaR 95.0%)        : ${report.var_95:,.2f}  ({(report.var_95/initial_value)*100:.2f}%)")
    print(f" Value at Risk (VaR 99.0%)        : ${report.var_99:,.2f}  ({(report.var_99/initial_value)*100:.2f}%)")
    print(f" Value at Risk (VaR 99.9%)        : ${report.var_999:,.2f}  ({(report.var_999/initial_value)*100:.2f}%)")
    print("-" * 70)
    print(f" Conditional VaR (CVaR / ES 95.0%): ${report.cvar_95:,.2f}  ({(report.cvar_95/initial_value)*100:.2f}%)")
    print(f" Conditional VaR (CVaR / ES 99.0%): ${report.cvar_99:,.2f}  ({(report.cvar_99/initial_value)*100:.2f}%)")
    print(f" Conditional VaR (CVaR / ES 99.9%): ${report.cvar_999:,.2f}  ({(report.cvar_999/initial_value)*100:.2f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_risk_pipeline()