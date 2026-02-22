from jiuhuang.data import JiuhuangData
from jiuhuang.strategy import *
from jiuhuang.backtest import backtest
from jiuhuang.metrics import *
from jiuhuang.dash import display_backtesting
import warnings

warnings.filterwarnings("ignore")

jh_data = JiuhuangData(sync=False)


def main():
    strategies = {
        "保持持有": StrategyBuyAndHold(),
        "海龟": StrategyTurtle(entry_window=20, exit_window=10),
        "量价趋势": StrategyVolumeTrend(),
        "量价背离": StrategyVolumeDivergence(volume_trend_threshold=0.05),
        "均值回归": StrategyMeanReversion(deviation_threshold=0.05),
        "移动均线交叉": StrategyMovingAverageCrossover(10, 20),
    }

    # 获取股票数据
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
        "stock_zh_a_hist_d",
        start_date="2025-10-06",
        end_date="2026-02-06",
    ).query("symbol in @symbols")
    
    stock_info = jh_data.get_data(
        "stock_individual_info_em"
    )

    backtest_result_df, plot_data = backtest(
        strategies,
        stock_price,
        stock_info,
        return_plot_data=True,
        commission_rate=0.00002,
        use_next_day_return=True,
    )

    # breakpoint()
    # print(backtest_result_df)
    display_backtesting(plot_data, backtest_result_df)


if __name__ == "__main__":
    import time

    start = time.time()
    main()
    end = time.time()
    print(f"耗时: {end - start}")
