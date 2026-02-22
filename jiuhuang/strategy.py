import pandas as pd
from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import os

# Get the number of CPUs minus one
n_jobs = max(1, os.cpu_count() - 1)


class Strategy(ABC):
    @abstractmethod
    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Execute the actual strategy logic for a single data."""
        pass

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        """Execute the strategy logic in parallel across groups."""
        # Sort and reset index
        price = price.sort_values(["symbol", "date"]).reset_index(drop=True)

        # Group by symbol and apply parallel processing
        grouped = price.groupby("symbol")
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._execute_one)(data) for _, data in grouped
        )

        # Concatenate results
        result_df = pd.concat(results, ignore_index=True)
        return self._validate_output(result_df)

    def _validate_output(self, result: pd.DataFrame) -> pd.DataFrame:
        """Validate that buy and sell signals are not simultaneously 1."""
        if "buy_signal" in result.columns and "sell_signal" in result.columns:
            simultaneous_mask = (result["buy_signal"] == 1) & (
                result["sell_signal"] == 1
            )
            if simultaneous_mask.any():
                print(
                    f"Warning: {simultaneous_mask.sum()} simultaneous buy/sell signals found. Resolving..."
                )
                result.loc[simultaneous_mask, "sell_signal"] = 0
        return result

    def __call__(self, price: pd.DataFrame) -> pd.DataFrame:
        result = self._execute_strategy(price)
        return self._validate_output(result)


class StrategyTurtle(Strategy):
    def __init__(self, entry_window: int = 20, exit_window: int = 10):
        self.entry_window = entry_window
        self.exit_window = exit_window

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals for a single symbol."""
        data = data.copy()

        # Calculate rolling high for entry and low for exit
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


class StrategyMovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and sell signals for a single symbol."""
        data = data.copy()

        data["short_ma"] = (
            data["close"].rolling(window=self.short_window, min_periods=1).mean()
        )
        data["long_ma"] = (
            data["close"].rolling(window=self.long_window, min_periods=1).mean()
        )

        data["buy_signal"] = (
            (data["short_ma"] > data["long_ma"])
            & (data["short_ma"].shift(1) <= data["long_ma"].shift(1))
        ).astype(int)

        data["sell_signal"] = (
            (data["short_ma"] < data["long_ma"])
            & (data["short_ma"].shift(1) >= data["long_ma"].shift(1))
        ).astype(int)

        data = data.drop(["short_ma", "long_ma"], axis=1)
        return data


class StrategyBuyAndHold(Strategy):
    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate buy and hold signals for a single symbol."""
        data = data.copy()

        # Create buy signal only on the first occurrence
        data["buy_signal"] = 0
        data["sell_signal"] = 0
        data.iloc[0, data.columns.get_loc("buy_signal")] = 1

        return data


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
        self.volume_threshold = volume_threshold
        self.volume_trend_threshold = volume_trend_threshold
        self.price_change_threshold = price_change_threshold

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Volume-based trend following strategy for a single symbol."""
        data = data.copy()

        data["ma"] = data["close"].rolling(window=self.ma_window, min_periods=1).mean()
        data["avg_volume"] = (
            data["volume"].rolling(window=self.volume_window, min_periods=1).mean()
        )
        data["volume_trend"] = data["avg_volume"].pct_change()

        data["buy_signal"] = (
            (data["close"] > data["ma"])
            & (data["volume"] > data["avg_volume"] * self.volume_threshold)
            & (data["volume_trend"] > self.volume_trend_threshold)
        ).astype(int)

        data["sell_signal"] = (
            (data["close"] < data["ma"])
            | (data["volume"] < data["avg_volume"] * 0.8)
            | (data["volume_trend"] < -self.volume_trend_threshold)
        ).astype(int)

        return data


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
        self.volume_trend_threshold = volume_trend_threshold
        self.price_change_threshold = price_change_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Strategy based on price-volume divergence for a single symbol."""
        data = data.copy()

        delta = data["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        data["rsi"] = 100 - (100 / (1 + rs))

        data["avg_volume"] = (
            data["volume"].rolling(window=self.volume_window, min_periods=1).mean()
        )
        data["volume_trend"] = data["avg_volume"].pct_change()

        data["buy_signal"] = (
            (data["rsi"] < self.rsi_oversold)
            & (data["volume_trend"] > self.volume_trend_threshold)
            & (data["pct_chg"] < -self.price_change_threshold)
        ).astype(int)

        data["sell_signal"] = (
            (data["rsi"] > self.rsi_overbought)
            & (data["volume_trend"] < -self.volume_trend_threshold)
            & (data["pct_chg"] > self.price_change_threshold)
        ).astype(int)

        return data


class StrategyMeanReversion(Strategy):
    def __init__(
        self,
        ma_window: int = 20,
        deviation_threshold: float = 0.02,
        rsi_window: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70,
    ):
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _execute_one(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals for a single symbol."""
        data = data.copy()

        data["ma"] = data["close"].rolling(window=self.ma_window, min_periods=1).mean()
        data["price_deviation"] = (data["close"] - data["ma"]) / data["ma"]

        delta = data["close"].diff()
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
        data["rsi"] = 100 - (100 / (1 + rs))

        data["buy_signal"] = (
            (data["price_deviation"] < -self.deviation_threshold)
            & (data["rsi"] < self.rsi_oversold)
        ).astype(int)

        data["sell_signal"] = (
            (data["price_deviation"] > self.deviation_threshold)
            & (data["rsi"] > self.rsi_overbought)
        ).astype(int)

        data = data.drop(["ma", "price_deviation", "rsi"], axis=1)
        return data
