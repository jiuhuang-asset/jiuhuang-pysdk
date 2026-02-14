import pandas as pd

def calculate_returns(df: pd.DataFrame, key="symbol") -> pd.DataFrame:
    """Calculate daily returns for each stock.

    Args:
        df: DataFrame with stock data

    Returns:
        DataFrame with added 'return' column
    """
    result_df = df.copy()
    result_df = result_df.sort_values([key, "date"]).reset_index(drop=True)
    result_df["return"] = result_df.groupby(key)["close"].pct_change().fillna(0)
    return result_df



def calculate_strategy_returns(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> pd.DataFrame:
    """Calculate strategy returns based on position and market returns, including transaction fees.

    Args:
        df: DataFrame with stock data and positions
        commission_rate: 买卖双向手续费率 
        stamp_tax_rate: 印花税率，仅对卖出收取 (默认 0.001 = 0.1%)

    Returns:
        DataFrame with added 'strategy_return' column
    """
    result_df = df.copy()
    result_df = calculate_returns(result_df)

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
    result_df = result_df.drop(
        ["prev_position", "is_selling", "commission_fee", "selling_fee", "total_fees"],
        axis=1,
    )

    return result_df


def cal_cum_ret(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> float:
    """Calculate cumulative strategy return.

    Args:
        df: DataFrame with stock data, signals and positions

    Returns:
        Series with cumulative returns per stock
    """
    result_df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate)

    # Calculate cumulative returns per stock
    def _cum_ret_group(g):
        if g.empty:
            return 0.0
        return (1 + g["strategy_return"]).cumprod().iloc[-1] - 1

    cum_per_stock = result_df.groupby("symbol", group_keys=False).apply(_cum_ret_group)
    cum_per_stock.name = "cum_return"
    return cum_per_stock


def cal_yearly_ret(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> float:
    """Calculate yearly return for each stock.

    Args:
        df: DataFrame with stock data, signals and positions

    Returns:
        Series with yearly return per stock
    """
    result_df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate)

    # Calculate yearly return per stock
    def _yearly_return_group(g):
        if g.empty:
            return 0.0

        # 计算策略累计收益
        cumulative_returns = (1 + g["strategy_return"]).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1

        # 计算交易天数
        n_days = len(g)
        if n_days == 0:
            return 0.0

        # 年化收益率计算 (假设一年252个交易日)
        n_years = n_days / 252.0
        if n_years == 0:
            return 0.0

        yearly_return = (cumulative_returns.iloc[-1]) ** (1 / n_years) - 1
        return yearly_return

    yearly_return_per_stock = result_df.groupby("symbol", group_keys=False).apply(
        _yearly_return_group
    )
    yearly_return_per_stock.name = "yearly_return"
    return yearly_return_per_stock


def cal_max_drawdown(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> float:
    """Calculate maximum drawdown for each stock.

    Args:
        df: DataFrame with stock data, signals and positions

    Returns:
        Series with maximum drawdown per stock
    """
    result_df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate)

    # Calculate cumulative returns and max drawdown per stock
    def _max_drawdown_group(g):
        if g.empty:
            return 0.0

        cumulative = (1 + g["strategy_return"]).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    max_dd_per_stock = result_df.groupby("symbol", group_keys=False).apply(
        _max_drawdown_group
    )
    max_dd_per_stock.name = "max_drawdown"
    return max_dd_per_stock


def cal_win_rate(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> float:
    """Calculate win rate (percentage of positive returns when in position).

    Args:
        df: DataFrame with stock data, signals and positions

    Returns:
        Series with win rate per stock
    """
    result_df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate)

    # Calculate win rate per stock
    def _win_rate_group(g):
        if g.empty:
            return 0.0

        # Only count periods when we were in position
        in_position_returns = g[g["position"] == 1]["strategy_return"]
        if len(in_position_returns) == 0:
            return 0.0

        wins = (in_position_returns > 0).sum()
        total_trades = len(in_position_returns)
        return wins / total_trades if total_trades > 0 else 0.0

    win_rate_per_stock = result_df.groupby("symbol", group_keys=False).apply(
        _win_rate_group
    )
    win_rate_per_stock.name = "win_rate"
    return win_rate_per_stock


def cal_sharpe_ratio(
    df: pd.DataFrame, commission_rate: float = 0.0002, stamp_tax_rate: float = 0.001
) -> float:
    """Calculate Sharpe ratio for each stock.

    Args:
        df: DataFrame with stock data, signals and positions

    Returns:
        Series with Sharpe ratio per stock
    """
    result_df = calculate_strategy_returns(df, commission_rate, stamp_tax_rate)

    # Calculate Sharpe ratio per stock
    def _sharpe_ratio_group(g):
        if g.empty:
            return 0.0

        # Get strategy returns when in position
        strategy_returns = g["strategy_return"]

        if len(strategy_returns) == 0:
            return 0.0

        # Calculate mean return and standard deviation
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        # Avoid division by zero
        if std_return == 0:
            return 0.0 if mean_return <= 0 else float("inf")

        # Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        sharpe = mean_return / std_return

        # Annualize the Sharpe ratio (assuming 252 trading days per year)
        annualized_sharpe = sharpe * (252**0.5)

        return annualized_sharpe

    sharpe_per_stock = result_df.groupby("symbol", group_keys=False).apply(
        _sharpe_ratio_group
    )
    sharpe_per_stock.name = "sharpe_ratio"
    return sharpe_per_stock
