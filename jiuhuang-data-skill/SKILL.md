---
name: jiuhuang
description: JiuHuang (韭皇) financial data framework for A-share stocks, macro data, ETFs, and backtesting. Provides MCP server for AI agents.
---

This skill guides working with JiuHuang - a financial data acquisition and backtesting framework for Chinese A-share data.

## Core Principle

**Always explore data types dynamically first** - never assume what data types exist. Use `search_data()` and `describe_data()` to find and understand available data before fetching.

## Essential Workflow

1. **Search** for data types: `jh.search_data("keyword", top_n=5)`
2. **Describe** to understand params: `jh.describe_data(DataTypes.XXX)`
3. **Fetch** the data: `jh.get_data(DataTypes.XXX, **kwargs)`

See `references/data-workflow.md` for detailed examples.

## MCP Server

For AI agent integration, run:
```bash
python -m jiuhuang.mcp
```

Tools: `search_data`, `describe_data`, `get_data`

See `references/mcp-server.md` for setup details.

## Key Modules

- `jiuhuang.data` - JiuhuangData, DataTypes
- `jiuhuang.strategy` - Built-in strategies
- `jiuhuang.backtest` - backtest()
- `jiuhuang.dash` - display_backtesting()

See `references/modules.md` for complete API.

## Common Patterns

- **Stock price**: `DataTypes.STOCK_ZH_A_HIST_QFQ`
- **ETF data**: `DataTypes.FUND_ETF_HIST_EM_QFQ`
- **Macro data**: `DataTypes.MACRO_CHINA_CPI`, `DataTypes.MACRO_CHINA_GDP`

## Environment

- `JIUHUANG_API_KEY` - Get from https://jiuhuang.xyz
- `JIUHUANG_API_URL` - Default: https://data.jiuhuang.xyz
