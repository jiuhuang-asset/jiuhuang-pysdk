from jiuhuang.dash import display_backtesting
import pandas as pd

# 模拟主数据
main_data = pd.DataFrame({
    "symbol": ["AAPL", "AAPL", "AAPL"],
    "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    "open": [100.0, 101.0, 102.0],
    "high": [105.0, 106.0, 107.0],
    "low": [99.0, 100.0, 101.0],
    "close": [104.0, 105.0, 106.0],
    "volume": [1000000, 1100000, 1200000],
    "buy_signal": [True, False, False],
    "sell_signal": [False, False, True],
    "strategy": ["ma_cross", "ma_cross", "ma_cross"],
})

# 模拟性能数据
perf_data = pd.DataFrame({
    "total_return": [0.15, 0.12],
    "sharpe_ratio": [1.5, 1.3],
    "max_drawdown": [-0.1, -0.08],
}, index=["strategy1", "strategy2"])

display_backtesting(main_data, perf_data)
