import pandas as pd
from typing import Dict, Callable
from .strategy import Strategy
from .risk_management import RiskManagementParams, risk_management_one_symbol
from .metrics import cal_metrics


def build_position(
    df: pd.DataFrame,
    buy_signal_name: str = "buy_signal",
    sell_signal_name: str = "sell_signal",
    use_next_day_return: bool = True,
    rmp: RiskManagementParams = RiskManagementParams(),
) -> pd.DataFrame:
    """Build position based on buy and sell signals for each stock with risk management.

    Args:
        df: DataFrame with stock data and signals
        buy_signal_name: Column name for buy signals
        sell_signal_name: Column name for sell signals
        use_next_day_return: Whether to apply signals to next day's positions
        rmp: RiskManagementParams object containing risk management parameters

    Returns:
        DataFrame with added 'position' column
    """
    result_df = df.copy()
    result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

    # Handle signal timing
    if use_next_day_return:
        buy_signal = result_df.groupby("symbol")[buy_signal_name].shift(1).fillna(0)
        sell_signal = result_df.groupby("symbol")[sell_signal_name].shift(1).fillna(0)
    else:
        buy_signal = result_df[buy_signal_name].fillna(0)
        sell_signal = result_df[sell_signal_name].fillna(0)

    result_df["position"] = 0

    # For each stock, calculate position based on signals and risk management
    for symbol in result_df["symbol"].unique():
        stock_mask = result_df["symbol"] == symbol
        stock_data = result_df.loc[stock_mask].copy()
        positions = risk_management_one_symbol(stock_data, buy_signal, sell_signal, rmp)
        result_df.loc[stock_mask, "position"] = positions

    return result_df


def evaluate_strategies(
    price: pd.DataFrame,
    strategies: dict[str, Strategy],
    symbols: list | None = None,
    use_next_day_return: bool = True,
    metric_func: Callable = cal_metrics,
    rmps: Dict[str, RiskManagementParams] = {},
    return_plot_data: bool = False,
) -> pd.DataFrame:
    results: dict[str, pd.Series] = {}
    _plot_data_cols = price.columns.to_list() + [
        "buy_signal",
        "sell_signal",
        "position",
        "strategy",
    ]
    if "created_at" in _plot_data_cols:
        _plot_data_cols.remove("created_at")
    _plot_datas = []
    for strat_name, strat in strategies.items():
        df_sig = strat(price)
        rmp = rmps.get(strat_name, RiskManagementParams())
        df_with_pos = build_position(
            df_sig,
            buy_signal_name="buy_signal",
            sell_signal_name="sell_signal",
            use_next_day_return=use_next_day_return,
            rmp=rmp,
        )
        metric_series = metric_func(df_with_pos)
        results[strat_name] = metric_series

        if return_plot_data:
            df_with_pos["strategy"] = strat_name
            _plot_datas.append(df_with_pos)
    # Combine into a DataFrame. Use union of stock_codes present in any result
    combined = pd.DataFrame(results)
    combined = combined.reset_index()
    combined = combined.rename(columns={"level_0": "symbol", "level_1": "metric"})
    return combined, pd.concat(_plot_datas).reset_index()[_plot_data_cols]


def backtest(
    strategies: Dict[str, Strategy],
    hist_price_data: pd.DataFrame,
    stock_info: pd.DataFrame  = pd.DataFrame(),
    rmps: Dict[str, RiskManagementParams] = {},
    commission_rate: float = 0.0002,
    stamp_tax_rate: float = 0.001,
    return_plot_data: bool = False,
    metric_decimal: int = 2,
    use_next_day_return: bool = True,
):
    if hist_price_data.empty:
        print("没有价格数据")
        return
    eval_results, plot_data = evaluate_strategies(
        hist_price_data,
        strategies,
        metric_func=lambda d: cal_metrics(d, commission_rate, stamp_tax_rate),
        rmps=rmps,
        return_plot_data=return_plot_data,
        use_next_day_return=use_next_day_return,
    )
    eval_results = eval_results.round(metric_decimal)
    melted_eval_results = eval_results.melt(
        id_vars=['symbol', 'metric'], 
        var_name='strategy', 
        value_name='value'
    )

    # Step 2: Pivot the melted DataFrame to spread metrics into columns
    reshaped_eval_results = melted_eval_results.pivot_table(
        index=['symbol', 'strategy'], 
        columns='metric', 
        values='value'
    ).reset_index()

    # Optional: Flatten column names if needed
    reshaped_eval_results.columns.name = None  # Remove the 'metric' label from columns

    if not stock_info.empty:
        reshaped_eval_results = reshaped_eval_results.merge(stock_info[["symbol", "symbol_name", "industry"]].drop_duplicates(), on="symbol", how="left")
    
    return reshaped_eval_results, plot_data