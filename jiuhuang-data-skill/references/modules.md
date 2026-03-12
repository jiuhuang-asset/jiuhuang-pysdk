# JiuHuang Modules

## jiuhuang.data

### JiuhuangData

Main class for data retrieval.

```python
from jiuhuang.data import JiuhuangData, DataTypes

jh = JiuhuangData()
```

#### Methods

##### search_data(keyword, top_n=5)
Search data types by keyword.

##### describe_data(data_type)
Get data type detailed documentation.

##### get_data(data_type, **kwargs)
Fetch data from API.

### DataTypes

Enum with 138+ data types. See `data_types.py` for full list.

```python
from jiuhuang.data import DataTypes

# Usage
DataTypes.STOCK_ZH_A_HIST_QFQ
DataTypes.MACRO_CHINA_CPI
```

## jiuhuang.data_types

Helper functions for data types.

### get_table_comment(data_type)

Get short description:

```python
from jiuhuang.data_types import get_table_comment, DataTypes

comment = get_table_comment(DataTypes.STOCK_ZH_A_HIST_QFQ)
```

### get_table_fields(data_type)

Get output fields list:

```python
from jiuhuang.data_types import get_table_fields, DataTypes

fields = get_table_fields(DataTypes.STOCK_ZH_A_HIST_QFQ)
```

### get_table_unique_keys(data_type)

Get unique keys for upsert:

```python
from jiuhuang.data_types import get_table_unique_keys, DataTypes

keys = get_table_unique_keys(DataTypes.STOCK_ZH_A_HIST_QFQ)
```

## jiuhuang.strategy

Built-in trading strategies.

```python
from jiuhuang.strategy import *
```

### Available Strategies

| Strategy | Description |
|----------|-------------|
| `StrategyTurtle` | Turtle trading - breakout strategy |
| `StrategyMovingAverageCrossover` | MA crossover |
| `StrategyBuyAndHold` | Buy and hold benchmark |
| `StrategyVolumeTrend` | Volume trend |
| `StrategyVolumeDivergence` | Volume divergence |
| `StrategyMeanReversion` | Mean reversion |
| `StrategyRSI` | RSI based |
| `StrategyBollingerBands` | Bollinger bands |
| `StrategyMomentum` | Momentum |
| `StrategyBreakout` | Breakout |
| `StrategyDualThrust` | Dual Thrust |

### Custom Strategy

```python
from jiuhuang.strategy import Strategy
import pandas as pd

class MyStrategy(Strategy):
    def __init__(self, entry_window=20, exit_window=10):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        # Your logic here
        data["buy_signal"] = ...
        data["sell_signal"] = ...
        return data
```

## jiuhuang.backtest

```python
from jiuhuang.backtest import backtest
```

### backtest(strategies, stock_price, stock_info)

Run backtesting:

```python
trading_history, backtest_perf = backtest(
    strategies,  # Dict of strategy name -> Strategy instance
    stock_price,  # Stock price DataFrame
    stock_info,   # Stock info DataFrame
)
```

## jiuhuang.dash

```python
from jiuhuang.dash import display_backtesting
```

### display_backtesting(trading_history, backtest_perf)

Show interactive dashboard:

```python
display_backtesting(trading_history, backtest_perf)
```

## jiuhuang.metrics

Performance metrics:

```python
from jiuhuang.metrics import calculate_metrics

metrics = calculate_metrics(returns)
```

## jiuhuang.risk_management

Risk management utilities.
