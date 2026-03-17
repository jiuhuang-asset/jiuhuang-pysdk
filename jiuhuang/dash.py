import webview
import pandas as pd
import os


class BacktestingView:
    def __init__(self, trading_hist: pd.DataFrame, perf_data: pd.DataFrame, date_column: str = "date"):
        self.date_column = date_column
        # 保留原始日期格式，支持分钟级数据
        cols = [
            "symbol",
            date_column,
            "open",
            "high",
            "low",
            "close",
            "volume",
            "buy_signal",
            "sell_signal",
            "strategy",
            "strategy_return",
            "cumulative_return",
            "drawdown",
        ]
        # 确保日期列存在
        available_cols = [c for c in cols if c in trading_hist.columns]
        self.trading_hist = (
            trading_hist[available_cols]
            .rename(columns={date_column: "date"})
            .assign(date=lambda x: x["date"].astype(str))
            .to_dict(orient="records")
        )
        # 区分数值字段和非数值字段，分别填充
        non_numeric_cols = ["symbol", "strategy", "name", "industry"]
        numeric_cols = [c for c in perf_data.columns if c not in non_numeric_cols]
        perf_filled = perf_data.copy()
        perf_filled[numeric_cols] = perf_filled[numeric_cols].fillna(0)
        perf_filled[non_numeric_cols] = perf_filled[non_numeric_cols].fillna("-")
        self.perf_data = perf_filled.to_dict(orient="records")

    def init_data(self):
        return {"trading_hist": self.trading_hist, "perf_data": self.perf_data, "date_column": self.date_column}


def display_backtesting(trading_hist: pd.DataFrame, perf_data: pd.DataFrame, date_column: str = "date"):
    """显示回测结果可视化看板。

    使用 PyWebView 打开本地 HTML 页面展示回测的交易历史和策略表现指标。
    支持日频和分钟级数据的时间字段。

    Args:
        trading_hist: 回测交易历史数据 DataFrame，需包含以下列：
            - symbol: 股票代码
            - date/trade_date/其他时间字段: 时间戳
            - open/high/low/close: OHLC 价格
            - volume: 成交量
            - buy_signal: 买入信号 (0 或 1)
            - sell_signal: 卖出信号 (0 或 1)
            - strategy: 策略名称
            - strategy_return: 策略收益率
            - cumulative_return: 累积收益率
            - drawdown: 回撤
        perf_data: 策略表现指标 DataFrame，包含各策略的绩效指标
        date_column: 时间字段名称，默认 "date"。支持 "date", "trade_date", "datetime" 等

    Returns:
        None: 该函数直接打开可视化窗口，不返回值

    Example:
        >>> from jiuhuang.backtest import backtest
        >>> from jiuhuang.dash import display_backtesting
        >>> from jiuhuang.strategy import StrategyTurtle
        >>>
        >>> # 运行回测
        >>> trading_history, results = backtest(
        ...     strategies={"turtle": StrategyTurtle()},
        ...     hist_price_data=price_df,
        ...     date_column="trade_date"
        ... )
        >>>
        >>> # 显示可视化看板
        >>> display_backtesting(trading_history, results, date_column="trade_date")
    """
    api = BacktestingView(trading_hist, perf_data, date_column)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "front_src", "bt-dash", "index.html")

    window = webview.create_window("回测结果看板", html_path, js_api=api)
    webview.start()
