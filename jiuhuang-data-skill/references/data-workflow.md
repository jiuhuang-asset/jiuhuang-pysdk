# JiuHuang Data Workflow

Complete workflow for exploring and fetching data from JiuHuang.

## Initialize JiuHuang

```python
from jiuhuang.data import JiuhuangData

# Uses JIUHUANG_API_KEY and JIUHUANG_API_URL env vars
jh = JiuhuangData()

# Or explicitly set credentials
jh = JiuhuangData(api_url="https://data.jiuhuang.xyz", api_key="your_key")
```

## Step 1: Search for Data Types

Search available data types by keyword (supports Chinese and English):

```python
# Search by Chinese keyword
results = jh.search_data("A股 股价 前复权", top_n=5)
# Returns: [{"STOCK_ZH_A_HIST_QFQ": "A股股价日数据(前复权)"}, ...]

# Search by English keyword
results = jh.search_data("stock_hist", top_n=3)
```

## Step 2: Describe Data Type

Get detailed information including input parameters and output fields:

```python
from jiuhuang.data import DataTypes

# Get full description (markdown format)
description = jh.describe_data(DataTypes.STOCK_ZH_A_HIST_QFQ)

# Returns markdown with:
# - Interface description
# - Input parameters table
# - Output fields table
# - Code example
# - Data sample
```

## Step 3: Fetch Data

Get data with parameters from describe_data:

```python
# Daily stock data
stock = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_QFQ,
    symbol="000001",
    start="2025-01-01",
    end="2025-12-31"
)

# Multiple symbols (comma-separated)
symbols = "000001,600036,600519"
stock = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_QFQ,
    symbol=symbols,
    start="2025-01-01",
    end="2025-12-31"
)
```

## Helper Functions

### get_table_comment

Get short description of a data type:

```python
from jiuhuang.data_types import get_table_comment, DataTypes

comment = get_table_comment(DataTypes.MACRO_CHINA_CPI)
# Returns: "中国CPI居民消费价格指数"
```

### get_table_fields

Get list of output fields:

```python
from jiuhuang.data_types import get_table_fields, DataTypes

fields = get_table_fields(DataTypes.STOCK_ZH_A_HIST_QFQ)
# Returns: ['date', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount', ...]
```

## Time Formats

Different data types use different time formats:

| Data Type | Format | Example |
|-----------|--------|---------|
| Daily | YYYY-MM-DD | "2025-01-01" |
| Monthly | YYYY-MM | "2025-01" |
| Minute | YYYY-MM-DD HH:MM:SS | "2025-01-01 09:30:00" |

## Common Data Types

```python
# Stock data
DataTypes.STOCK_ZH_A_HIST          # Unadjusted
DataTypes.STOCK_ZH_A_HIST_QFQ      # Forward-adjusted
DataTypes.STOCK_ZH_A_HIST_HFQ      # Backward-adjusted
DataTypes.STOCK_ZH_A_SPOT          # Real-time minute data
DataTypes.STOCK_INDIVIDUAL_INFO_EM # Stock basic info

# ETF/Fund data
DataTypes.FUND_ETF_HIST_EM        # ETF daily
DataTypes.FUND_ETF_HIST_EM_QFQ     # ETF forward-adjusted
DataTypes.FUND_ETF_HIST_EM_HFQ     # ETF backward-adjusted

# Macro data
DataTypes.MACRO_CHINA_CPI          # CPI
DataTypes.MACRO_CHINA_GDP          # GDP
DataTypes.MACRO_CHINA_PPI          # PPI
DataTypes.MACRO_CHINA_LPR          # Loan Prime Rate

# Global indices
DataTypes.INDEX_GLOBAL_HIST_EM    # Global index history
DataTypes.INDEX_GLOBAL_SPOT_EM    # Global index spot
```
