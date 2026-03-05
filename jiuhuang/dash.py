import webview
import pandas as pd
import os


class BacktestingView:
    def __init__(self, trading_hist: pd.DataFrame, perf_data: pd.DataFrame):
        self.trading_hist = (
            trading_hist[
                [
                    "symbol",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "buy_signal",
                    "sell_signal",
                    "strategy",
                ]
            ]
            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
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
        return {"trading_hist": self.trading_hist, "perf_data": self.perf_data}


def display_backtesting(trading_hist, pref_data):
    api = BacktestingView(trading_hist, pref_data)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, "front_src", "bt-dash", "index.html")

    window = webview.create_window("回测结果看板", html_path, js_api=api)
    webview.start()
