from jiuhuang.data import JiuhuangData, DataTypes
from jiuhuang.strategy import Strategy, StrategyBuyAndHold
from jiuhuang.backtest import backtest
from jiuhuang.dash import display_backtesting
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

jh_data = JiuhuangData()


# 自定义策略, 需要集成Strategy基类，必须实现_execute_one方法(入参：pandas.DataFrame, 输出：pandas.DataFram, 输出需要包含buy_signal和sell_signal两个列)
# jiuhuang会默认使用多进程并行进行回测， 所以速度很快
class MyStrategy(Strategy):
    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """对单个标的生成海龟策略的买卖信号"""
        data = data.copy()

        # 计算滚动最高价和最低价
        data["entry_high"] = (
            data["high"].rolling(window=self.entry_window, min_periods=1).max()
        )
        data["exit_low"] = (
            data["low"].rolling(window=self.exit_window, min_periods=1).min()
        )

        # Generate signals
        data["buy_signal"] = (data["close"] > data["entry_high"].shift(1)).astype(int)
        data["sell_signal"] = (data["close"] < data["exit_low"].shift(1)).astype(int)

        # Clean up temporary columns
        data = data.drop(["entry_high", "exit_low"], axis=1)
        return data

def main():
    strategies = {
        "海龟(测试)": MyStrategy(),
        "保持持有": StrategyBuyAndHold(), # 导入jiuhuang定义的strategy
    }

    # 获取股票数据， 正式交易场景下需要结合一些选股逻辑
    symbols = [
        "600135",  # Example stock
        "000001",  # Ping An Bank
        "600036",  # China Merchants Bank
        "600519",  # Kweichow Moutai
        "000858",  # Wuliangye Yibin
        "601318",  # China Ping An
        "000002",  # Vanke A
        "600030",  # CITIC Securities
        "600000",  # SAIC Motor
        "600016",  # Minsheng Bank
        "600048",  # Poly Development
        "600887",  # Inner Mongolia Yili
        "601166",  # Industrial Bank
        "601601",  # China Pacific Insurance
        "601628",  # China Life Insurance
        "601857",  # PetroChina
        "601939",  # China Construction Bank
        "601988",  # Bank of China
        "000063",  # ZTE Corporation
        "000333",  # Midea Group
        "000425",  # XCMG Machinery
        "000568",  # Luzhou Laojiao
        "000651",  # Gree Electric
        "000725",  # BOE Technology
        "000776",  # GF Securities
        "000895",  # Yurun Food
        "002027",  # Focus Media
        "002142",  # Ningbo Bank
        "002230",  # iFLYTEK
        "002415",  # Hikvision
    ]
    stock_price = jh_data.get_data(
        DataTypes.STOCK_ZH_A_HIST_QFQ,
        start="2024-12-25",
        end="2026-03-11",
        symbol=",".join(symbols),
    )
    stock_info = jh_data.get_data(
        DataTypes.STOCK_INDIVIDUAL_INFO_EM,
    )
    trading_history, backtest_perf = backtest(
        strategies,
        stock_price,
        stock_info,
    ) # trading_history为交易历史(明细数据), backtest_perf为回测结果(总览数据)

    # 展示回测面板， 注意trading_history和backtest_perf在数量过大的情况下可能会非常慢
    display_backtesting(trading_history, backtest_perf)


if __name__ == "__main__":
    main()
