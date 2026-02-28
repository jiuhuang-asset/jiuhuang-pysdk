import pandas as pd
import numpy as np
from jiuhuang.dash import display_backtesting

# 创建模拟的 main_data
dates = pd.date_range("2024-01-01", periods=30, freq="D")
main_data = pd.DataFrame(
    {
        "symbol": ["AAPL"] * 30,
        "date": dates,
        "open": np.random.uniform(100, 110, 30),
        "high": np.random.uniform(110, 120, 30),
        "low": np.random.uniform(90, 100, 30),
        "close": np.random.uniform(100, 115, 30),
        "volume": np.random.randint(1000000, 5000000, 30),
        "buy_signal": [True if i % 5 == 0 else False for i in range(30)],
        "sell_signal": [True if i % 7 == 0 else False for i in range(30)],
        "strategy": ["ma_cross"] * 30,
    }
)

# 创建模拟的 perf_data
perf_data = pd.DataFrame(
    {
        "metric": ["total_return", "sharpe_ratio", "max_drawdown", "win_rate"],
        "value": [0.15, 1.8, -0.08, 0.62],
    }
)

display_backtesting(main_data, perf_data)
