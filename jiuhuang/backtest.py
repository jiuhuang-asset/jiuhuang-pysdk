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
    dt_column: str = "date",
) -> pd.DataFrame:
    """根据买卖信号构建持仓，结合风险管理规则。

    Args:
        df: 包含股票数据和信号的 DataFrame
        buy_signal_name: 买入信号列名
        sell_signal_name: 卖出信号列名
        use_next_day_return: 是否将信号应用到次日持仓
        rmp: 风险管理参数对象
        dt_column: 时间字段名称，默认 "date"

    Returns:
        新增 'position' 列的 DataFrame
    """
    result_df = df.copy()
    result_df = result_df.sort_values(["symbol", dt_column]).reset_index(drop=True)

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
    dt_column: str = "date",
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
            # Set dt_column for the strategy
            strat.dt_column = dt_column
            df_sig = strat(price)
            rmp = rmps.get(strat_name, RiskManagementParams())
            df_with_pos = build_position(
                df_sig,
                buy_signal_name="buy_signal",
                sell_signal_name="sell_signal",
                use_next_day_return=use_next_day_return,
                rmp=rmp,
                dt_column=dt_column,
            )
            # Calculate strategy returns (includes cumulative_return)
            strat_trading_histroy = calculate_strategy_returns(
                df_with_pos, commission_rate, stamp_tax_rate, dt_column
            )
            metric_series = metric_func(strat_trading_histroy, dt_column=dt_column)
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
    dt_column: str = "date",
):
    """回测策略表现。

    Args:
        strategies: 策略字典，键为策略名称，值为策略函数
        hist_price_data: 历史价格数据 DataFrame
        stock_info: 股票信息 DataFrame（可选，包含 symbol, name, industry）
        rmps: 各策略的风险管理参数字典
        commission_rate: 佣金费率（默认 0.0002）
        stamp_tax_rate: 印花税率（默认 0.0005）
        metric_decimal: 指标小数位数（默认 2）
        use_next_day_return: 是否使用次日收益率（默认 True）
        dt_column: 时间字段名称，默认 "date"

    Returns:
        tuple: (trading_history, reshaped_eval_results)
            - trading_history: 交易历史记录
            - reshaped_eval_results: 重塑后的评估指标结果
    """
    if hist_price_data.empty:
        rprint("[bold yellow]没有价格数据")
        return
    if dt_column not in hist_price_data.columns:
        raise ValueError(f"hist_price_data缺少`{dt_column}`字段, 或者你可以指定`dt_column`参数")
    eval_results, trading_history = evaluate_strategies(
        hist_price_data,
        strategies,
        metric_func=lambda d: cal_metrics_from_returns(d, dt_column=dt_column),
        rmps=rmps,
        use_next_day_return=use_next_day_return,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        dt_column=dt_column,
    )
    eval_results = eval_results.round(metric_decimal)
    melted_eval_results = eval_results.melt(
        id_vars=["symbol", "metric"], var_name="strategy", value_name="value"
    )

    reshaped_eval_results = melted_eval_results.pivot_table(
        index=["symbol", "strategy"], columns="metric", values="value"
    ).reset_index()

    reshaped_eval_results.columns.name = None 

    if not stock_info.empty:
        reshaped_eval_results = reshaped_eval_results.merge(
            stock_info[["symbol", "name", "industry"]].drop_duplicates(),
            on="symbol",
            how="left",
        )

    return trading_history, reshaped_eval_results
