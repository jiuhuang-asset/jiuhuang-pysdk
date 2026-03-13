import pandas as pd
from typing import Dict, Callable
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn, TaskProgressColumn
from .strategy import Strategy
from .risk_management import RiskManagementParams, risk_management_one_symbol
from .metrics import cal_metrics, cal_metrics_from_returns, calculate_strategy_returns
from rich import print as rprint

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
    use_next_day_return: bool = True,
    metric_func: Callable = cal_metrics,
    rmps: Dict[str, RiskManagementParams] = {},
    commission_rate: float = 0.0002,
    stamp_tax_rate: float = 0.0005,
) -> pd.DataFrame:
    perf_results: dict[str, pd.Series] = {}
    _trading_history_datas = []
    _trading_history_cols = price.columns.to_list() + [
        "buy_signal",
        "sell_signal",
        "position",
        "strategy",
        "strategy_return",
        "cumulative_return",
        "drawdown",
    ]
    if "created_at" in _trading_history_cols:
        _trading_history_cols.remove("created_at")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"[cyan]Evaluating {len(strategies)} strategies...", total=len(strategies)
        )
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
            # Calculate strategy returns (includes cumulative_return)
            strat_trading_histroy = calculate_strategy_returns(
                df_with_pos, commission_rate, stamp_tax_rate
            )
            metric_series = metric_func(strat_trading_histroy)
            perf_results[strat_name] = metric_series
            strat_trading_histroy["strategy"] = strat_name
            _trading_history_datas.append(strat_trading_histroy)
            progress.update(task, advance=1)
    # Combine into a DataFrame. Use union of stock_codes present in any result
    combined = pd.DataFrame(perf_results)
    combined = combined.reset_index()
    combined_performance = combined.rename(
        columns={"level_0": "symbol", "level_1": "metric"}
    )
    return (
        combined_performance,
        pd.concat(_trading_history_datas).reset_index()[_trading_history_cols],
    )


def backtest(
    strategies: Dict[str, Strategy],
    hist_price_data: pd.DataFrame,
    stock_info: pd.DataFrame = pd.DataFrame(),
    rmps: Dict[str, RiskManagementParams] = {},
    commission_rate: float = 0.0002,
    stamp_tax_rate: float = 0.0005,
    metric_decimal: int = 2,
    use_next_day_return: bool = True,
):
    if hist_price_data.empty:
        rprint("[bold yellow]没有价格数据")
        return
    eval_results, trading_history = evaluate_strategies(
        hist_price_data,
        strategies,
        metric_func=lambda d: cal_metrics_from_returns(d),
        rmps=rmps,
        use_next_day_return=use_next_day_return,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
    )
    eval_results = eval_results.round(metric_decimal)
    melted_eval_results = eval_results.melt(
        id_vars=["symbol", "metric"], var_name="strategy", value_name="value"
    )

    # Step 2: Pivot the melted DataFrame to spread metrics into columns
    reshaped_eval_results = melted_eval_results.pivot_table(
        index=["symbol", "strategy"], columns="metric", values="value"
    ).reset_index()

    # Optional: Flatten column names if needed
    reshaped_eval_results.columns.name = None  # Remove the 'metric' label from columns

    if not stock_info.empty:
        reshaped_eval_results = reshaped_eval_results.merge(
            stock_info[["symbol", "name", "industry"]].drop_duplicates(),
            on="symbol",
            how="left",
        )

    return trading_history, reshaped_eval_results
