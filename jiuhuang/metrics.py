import pandas as pd
import quantstats as qs
import numpy as np


__all__ = ["cal_metrics", "cal_metrics_from_returns"]


def calculate_returns(df: pd.DataFrame, key: str = "symbol", date_column: str = "date") -> pd.DataFrame:
    """Calculate daily returns for each stock.

    Args:
        df: DataFrame with stock data
        key: 分组键，通常是股票代码字段名
        date_column: 时间字段名称，默认 "date"

    Returns:
        DataFrame with added 'return' column
    """
    result_df = df.copy()
    result_df = result_df.sort_values([key, date_column]).reset_index(drop=True)
    result_df["return"] = result_df.groupby(key)["close"].pct_change().fillna(0)
    return result_df


def calculate_strategy_returns(
    df: pd.DataFrame,
    commission_rate: float = 0,
    stamp_tax_rate: float = 0,
    date_column: str = "date",
) -> pd.DataFrame:
    """Calculate strategy returns based on position and market returns, including transaction fees.

    Args:
        df: DataFrame with stock data and positions
        commission_rate: 买卖双向手续费率
        stamp_tax_rate: 印花税率，仅对卖出收取 (默认 0.0005 = 0.05%)
        date_column: 时间字段名称，默认 "date"

    Returns:
        DataFrame with added 'strategy_return' column
    """
    result_df = df.copy()
    result_df = calculate_returns(result_df, date_column=date_column)

    result_df["strategy_return"] = result_df["return"] * result_df["position"]

    result_df["prev_position"] = (
        result_df.groupby("symbol")["position"].shift(1).fillna(0)
    )
    result_df["is_selling"] = (result_df["position"] == 0) & (
        result_df["prev_position"] == 1
    )
    result_df["commission_fee"] = (
        (result_df["position"] != result_df["prev_position"])
        & (result_df["position"] == 1)
    ) * commission_rate  # 买入时的手续费

    # 卖出时的总费用 = 手续费 + 印花税
    result_df["selling_fee"] = result_df["is_selling"] * (
        commission_rate + stamp_tax_rate
    )
    result_df["total_fees"] = result_df["commission_fee"] + result_df["selling_fee"]

    result_df["strategy_return"] = (
        result_df["strategy_return"] - result_df["total_fees"]
    )

    # Calculate cumulative return
    result_df["cumulative_return"] = result_df.groupby("symbol")[
        "strategy_return"
    ].transform(lambda x: (1 + x).cumprod() - 1)

    # Calculate max drawdown
    result_df["drawdown"] = result_df.groupby("symbol")["cumulative_return"].transform(
        lambda x: (x.cummax() - x) / (1 + x.cummax())
    )

    result_df = result_df.drop(
        ["prev_position", "is_selling", "commission_fee", "selling_fee", "total_fees"],
        axis=1,
    )

    return result_df


def cal_metrics(
    df: pd.DataFrame,
    commission_rate: float = 0.0002,
    stamp_tax_rate: float = 0.0005,
    date_column: str = "date",
) -> pd.Series:
    """Calculate multiple strategy metrics using quantstats for each symbol.

    Args:
        df: DataFrame with stock data, signals, and positions.
        commission_rate: Commission rate for buying/selling.
        stamp_tax_rate: Stamp tax rate for selling.
        date_column: 时间字段名称，默认 "date"

    Returns:
        pd.Series with multi-level index (metric_name, symbol).
    """
    df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate, date_column)
    metrics = [
        "累积收益率",
        "最大回撤",
        "胜率",
        "夏普比率",
        "卡玛比率",
        "索提诺比率",
        "收益率标准差",
        "风险价值(VaR)",
    ]

    results = []
    for symbol, group in df.groupby("symbol"):
        # Extract strategy returns for this symbol
        returns = group.set_index(date_column)["strategy_return"]

        # Compute metrics
        cumulative_return = group.groupby("symbol")["cumulative_return"].last().iloc[0]
        max_dd = qs.stats.max_drawdown(returns)
        win_rate = (returns > 0).mean()
        sharpe = qs.stats.sharpe(returns)
        calmar = qs.stats.calmar(returns) if max_dd != 0 else np.nan
        sortino = qs.stats.sortino(returns)
        volatility = returns.std()
        var = qs.stats.value_at_risk(returns)

        symbol_results = pd.Series(
            [cumulative_return, max_dd, win_rate, sharpe, calmar, sortino, volatility, var],
            index=pd.MultiIndex.from_product([[symbol], metrics]),
        )
        results.append(symbol_results)

    combined_series = pd.concat(results)
    return combined_series


def cal_metrics_from_returns(df: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Calculate metrics from already computed strategy returns.

    Args:
        df: DataFrame with stock data and 'strategy_return' column already calculated.
        date_column: 时间字段名称，默认 "date"

    Returns:
        pd.Series with multi-level index (metric_name, symbol).
    """
    metrics = [
        "累积收益率",
        "最大回撤",
        "胜率",
        "夏普比率",
        "卡玛比率",
        "索提诺比率",
        "收益率标准差",
        "风险价值(VaR)",
    ]

    results = []
    for symbol, group in df.groupby("symbol"):
        returns = group.set_index(date_column)["strategy_return"]

        cumulative_return = group.groupby("symbol")["cumulative_return"].last().iloc[0]
        max_dd = qs.stats.max_drawdown(returns)
        win_rate = (returns > 0).mean()
        sharpe = qs.stats.sharpe(returns)
        calmar = qs.stats.calmar(returns) if max_dd != 0 else np.nan
        sortino = qs.stats.sortino(returns)
        volatility = returns.std()
        var = qs.stats.value_at_risk(returns)

        symbol_results = pd.Series(
            [cumulative_return, max_dd, win_rate, sharpe, calmar, sortino, volatility, var],
            index=pd.MultiIndex.from_product([[symbol], metrics]),
        )
        results.append(symbol_results)

    combined_series = pd.concat(results)
    return combined_series
