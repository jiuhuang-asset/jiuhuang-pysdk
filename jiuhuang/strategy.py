import pandas as pd
from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Execute the actual strategy logic."""
        pass

    def __call__(self, price: pd.DataFrame) -> pd.DataFrame:
        result = self._execute_strategy(price)
        return self._validate_output(result)

    def _validate_output(self, result: pd.DataFrame) -> pd.DataFrame:
        """Validate that buy and sell signals are not simultaneously 1."""
        if "buy_signal" in result.columns and "sell_signal" in result.columns:
            simultaneous_mask = (result["buy_signal"] == 1) & (result["sell_signal"] == 1)
            if simultaneous_mask.any():
                print(f"Warning: {simultaneous_mask.sum()} simultaneous buy/sell signals found. Resolving...")
                result.loc[simultaneous_mask, "sell_signal"] = 0
        return result



class StrategyTurtle(Strategy):
    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals based on turtle trading strategy.

        The strategy buys when price breaks above the highest high in entry_window,
        and sells when price falls below the lowest low in exit_window.

        Args:
            price: DataFrame with stock price data

        Returns:
            DataFrame with added 'buy_signal' and 'sell_signal' columns
        """
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Calculate rolling high for entry and low for exit
        result_df["entry_high"] = (
            result_df.groupby("symbol")["high"]
            .rolling(window=self.entry_window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

        result_df["exit_low"] = (
            result_df.groupby("symbol")["low"]
            .rolling(window=self.exit_window, min_periods=1)
            .min()
            .reset_index(level=0, drop=True)
        )

        # Generate signals
        result_df["buy_signal"] = (
            result_df["close"] > result_df["entry_high"].shift(1)
        ).astype(int)
        result_df["sell_signal"] = (
            result_df["close"] < result_df["exit_low"].shift(1)
        ).astype(int)

        # Clean up temporary columns
        result_df = result_df.drop(["entry_high", "exit_low"], axis=1)

        return result_df




class StrategyMovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals based on moving average crossover strategy.

        The strategy buys when the short-term moving average crosses above the long-term moving average,
        and sells when it crosses below.

        Args:
            price: DataFrame with stock price data

        Returns:
            DataFrame with added 'buy_signal' and 'sell_signal' columns
        """
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        result_df["short_ma"] = result_df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=self.short_window, min_periods=1).mean()
        )
        result_df["long_ma"] = result_df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=self.long_window, min_periods=1).mean()
        )

        result_df["buy_signal"] = (
            (result_df["short_ma"] > result_df["long_ma"])
            & (result_df["short_ma"].shift(1) <= result_df["long_ma"].shift(1))
        ).astype(int)

        result_df["sell_signal"] = (
            (result_df["short_ma"] < result_df["long_ma"])
            & (result_df["short_ma"].shift(1) >= result_df["long_ma"].shift(1))
        ).astype(int)

        result_df = result_df.drop(["short_ma", "long_ma"], axis=1)
        return result_df



class StrategyBuyAndHold(Strategy):
    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and hold signals - buy on first day and hold forever.

        The strategy buys on the first day and maintains the position for all subsequent days.

        Args:
            price: DataFrame with stock price data

        Returns:
            DataFrame with added 'buy_signal' and 'sell_signal' columns
        """
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Create buy signal only on the first occurrence of each symbol
        result_df["buy_signal"] = 0
        result_df["sell_signal"] = 0

        # For each symbol, set buy signal to 1 on the first day
        for symbol in result_df["symbol"].unique():
            symbol_mask = result_df["symbol"] == symbol
            first_idx = result_df[symbol_mask].index[0]
            result_df.loc[first_idx, "buy_signal"] = 1

        return result_df



class StrategyVolumeTrend(Strategy):
    def __init__(
        self,
        ma_window: int = 20,
        volume_window: int = 20,
        volume_threshold: float = 1.2,
        volume_trend_threshold: float = 0.1,
        price_change_threshold: float = 0.02,
    ):
        self.ma_window = ma_window
        self.volume_window = volume_window
        self.volume_threshold = (
            volume_threshold  # Threshold for volume to be considered high
        )
        self.volume_trend_threshold = (
            volume_trend_threshold  # Minimum volume trend change
        )
        self.price_change_threshold = (
            price_change_threshold  # Minimum price change for signal
        )

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Volume-based trend following strategy."""
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Calculate moving averages for trend identification
        result_df["ma"] = result_df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=self.ma_window, min_periods=1).mean()
        )

        # Calculate average volume for volume confirmation
        result_df["avg_volume"] = result_df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(window=self.volume_window, min_periods=1).mean()
        )

        # Calculate volume trend as percentage change
        result_df["volume_trend"] = result_df.groupby("symbol")[
            "avg_volume"
        ].pct_change()

        # Identify trends with volume confirmation
        # Buy when price is above MA, volume is above average, and volume trend is positive
        result_df["buy_signal"] = (
            (result_df["close"] > result_df["ma"])
            & (result_df["volume"] > result_df["avg_volume"] * self.volume_threshold)
            & (result_df["volume_trend"] > self.volume_trend_threshold)
        ).astype(int)

        # Sell when price is below MA, volume is below average, or volume dries up significantly
        result_df["sell_signal"] = (
            (result_df["close"] < result_df["ma"])
            | (result_df["volume"] < result_df["avg_volume"] * 0.8)
            | (result_df["volume_trend"] < -self.volume_trend_threshold)
        ).astype(int)

        return result_df



class StrategyVolumeDivergence(Strategy):
    def __init__(
        self,
        rsi_window: int = 14,
        volume_window: int = 20,
        volume_trend_threshold: float = 0.05,
        price_change_threshold: float = 0.02,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
    ):
        self.rsi_window = rsi_window
        self.volume_window = volume_window
        self.volume_trend_threshold = (
            volume_trend_threshold  # Minimum volume trend change
        )
        self.price_change_threshold = (
            price_change_threshold  # Minimum price change for divergence
        )
        self.rsi_oversold = rsi_oversold  # RSI level considered oversold
        self.rsi_overbought = rsi_overbought  # RSI level considered overbought

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Strategy based on price-volume divergence."""
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Calculate RSI for price trend
        delta = result_df.groupby("symbol")["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        result_df["rsi"] = 100 - (100 / (1 + rs))

        # Calculate volume momentum
        result_df["avg_volume"] = result_df.groupby("symbol")["volume"].transform(
            lambda x: x.rolling(window=self.volume_window, min_periods=1).mean()
        )
        result_df["volume_trend"] = result_df.groupby("symbol")[
            "avg_volume"
        ].pct_change()

        # 买入信号：价格超卖 + 成交量增加 + 价格大幅下跌
        result_df["buy_signal"] = (
            (result_df["rsi"] < self.rsi_oversold)
            & (result_df["volume_trend"] > self.volume_trend_threshold)  # Oversold
            & (  # Volume increasing
                result_df["pct_chg"] < -self.price_change_threshold
            )  # Price declining significantly
        ).astype(int)

        # 卖出信号：价格超买 + 成交量减少 + 价格大幅上涨
        result_df["sell_signal"] = (
            (result_df["rsi"] > self.rsi_overbought)
            & (result_df["volume_trend"] < -self.volume_trend_threshold)  # Overbought
            & (  # Volume decreasing
                result_df["pct_chg"] > self.price_change_threshold
            )  # Price rising significantly
        ).astype(int)

        return result_df


class StrategyMeanReversion(Strategy):
    def __init__(
        self,
        ma_window: int = 20,
        deviation_threshold: float = 0.02,
        rsi_window: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
    ):
        """
        均值回归策略 - 当价格偏离移动平均线过多时进行反向交易
        同时结合RSI指标确认超买超卖状态

        Args:
            ma_window: 移动平均线窗口
            deviation_threshold: 价格偏离移动平均线的阈值（如0.02表示2%）
            rsi_window: RSI计算窗口
            rsi_oversold: RSI超卖阈值
            rsi_overbought: RSI超买阈值
        """
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """生成均值回归策略的买卖信号

        Args:
            price: 包含股票价格数据的DataFrame

        Returns:
            添加了'buy_signal'和'sell_signal'列的DataFrame
        """
        result_df = price.copy()
        result_df = result_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        # 计算移动平均线
        result_df["ma"] = result_df.groupby("symbol")["close"].transform(
            lambda x: x.rolling(window=self.ma_window, min_periods=1).mean()
        )

        # 计算价格偏离移动平均线的百分比
        result_df["price_deviation"] = (
            result_df["close"] - result_df["ma"]
        ) / result_df["ma"]

        # 计算RSI
        delta = result_df.groupby("symbol")["close"].diff()
        gain = (
            delta.where(delta > 0, 0)
            .rolling(window=self.rsi_window, min_periods=1)
            .mean()
        )
        loss = (
            (-delta.where(delta < 0, 0))
            .rolling(window=self.rsi_window, min_periods=1)
            .mean()
        )
        rs = gain / loss
        result_df["rsi"] = 100 - (100 / (1 + rs))

        # 买入信号：价格显著低于移动平均线（超卖）且RSI也显示超卖
        result_df["buy_signal"] = (
            (result_df["price_deviation"] < -self.deviation_threshold)
            & (result_df["rsi"] < self.rsi_oversold)  # 价格显著低于均线  # RSI显示超卖
        ).astype(int)

        # 卖出信号：价格显著高于移动平均线（超买）且RSI也显示超买
        result_df["sell_signal"] = (
            (result_df["price_deviation"] > self.deviation_threshold)
            & (result_df["rsi"] > self.rsi_overbought)  # 价格显著高于均线  # RSI显示超买
        ).astype(int)

        # 清理临时列
        result_df = result_df.drop(["ma", "price_deviation", "rsi"], axis=1)

        return result_df

