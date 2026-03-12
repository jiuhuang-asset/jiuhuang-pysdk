# quick start
from jiuhuang.data import JiuhuangData, DataTypes
from jiuhuang.strategy import *
from jiuhuang.backtest import backtest
from jiuhuang.dash import display_backtesting
import warnings

# 安装(升级)方式 pip install -U jiuhuang  

warnings.filterwarnings("ignore")

jh = JiuhuangData(api_url="https://data.jiuhuang.xyz", api_key="你的API KEY")
# 建议通过设置环境变API_URL和API_KEY, 而不是用明文， API申请：https://jiuhuang.xyz
# jh = JiuhuangData()

def main():
    strategies = {
        "海龟": StrategyTurtle(entry_window=20, exit_window=10),
        "移动均线交叉": StrategyMovingAverageCrossover(12, 24),
    }

    # 获取股票数据
    symbols = [
        "000001",  # Ping An Bank
        "600036",  # China Merchants Bank
        "600519",  # Kweichow Moutai
        "000858",  # Wuliangye Yibin
        "601318",  # China Ping An
        "000002",  # Vanke A
    ]
    stock_price = jh.get_data(
        DataTypes.STOCK_ZH_A_HIST_QFQ, # 前复权股票数据
        start="2024-12-25",
        end="2026-03-11",
        symbol=",".join(symbols),
    )
    stock_info = jh.get_data(
        DataTypes.STOCK_INDIVIDUAL_INFO_EM,
    )
    trading_history, backtest_perf = backtest(
        strategies,
        stock_price,
        stock_info,
    )
    # 展示回测面板
    display_backtesting(trading_history, backtest_perf)

if __name__ == "__main__":
    main()