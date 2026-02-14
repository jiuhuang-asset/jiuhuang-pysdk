import pandas as pd
import duckdb
from abc import ABC, abstractmethod



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
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        # 窗口函数获取前n天的最高/最低价（不含当日）
        query = f"""
        WITH base AS (
            SELECT *,
                MAX(high) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN {self.entry_window} PRECEDING AND 1 PRECEDING) as entry_high,
                MIN(low) OVER (PARTITION BY symbol ORDER BY date 
                    ROWS BETWEEN {self.exit_window} PRECEDING AND 1 PRECEDING) as exit_low
            FROM df_price
        )
        SELECT * EXCLUDE(entry_high, exit_low),
            CASE WHEN close > entry_high THEN 1 ELSE 0 END as buy_signal,
            CASE WHEN close < exit_low THEN 1 ELSE 0 END as sell_signal
        FROM base
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result

class StrategyMovingAverageCrossover(Strategy):
    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        query = f"""
        WITH ma_calc AS (
            SELECT *,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS {self.short_window-1} PRECEDING) as short_ma,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS {self.long_window-1} PRECEDING) as long_ma
            FROM df_price
        ),
        signals AS (
            SELECT *,
                LAG(short_ma) OVER (PARTITION BY symbol ORDER BY date) as prev_short,
                LAG(long_ma) OVER (PARTITION BY symbol ORDER BY date) as prev_long
            FROM ma_calc
        )
        SELECT * EXCLUDE(short_ma, long_ma, prev_short, prev_long),
            CASE WHEN short_ma > long_ma AND prev_short <= prev_long THEN 1 ELSE 0 END as buy_signal,
            CASE WHEN short_ma < long_ma AND prev_short >= prev_long THEN 1 ELSE 0 END as sell_signal
        FROM signals
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result

class StrategyBuyAndHold(Strategy):
    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        # 使用 ROW_NUMBER 替代 Python 循环，性能大幅提升
        query = """
        SELECT *,
            CASE WHEN ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) = 1 THEN 1 ELSE 0 END as buy_signal,
            0 as sell_signal
        FROM df_price
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result

class StrategyVolumeTrend(Strategy):
    def __init__(self, ma_window: int = 20, volume_window: int = 20, volume_threshold: float = 1.2, 
                 volume_trend_threshold: float = 0.1, price_change_threshold: float = 0.02):
        self.ma_window = ma_window
        self.volume_window = volume_window
        self.volume_threshold = volume_threshold
        self.volume_trend_threshold = volume_trend_threshold

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        query = f"""
        WITH metrics AS (
            SELECT *,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS {self.ma_window-1} PRECEDING) as ma,
                AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS {self.volume_window-1} PRECEDING) as avg_volume
            FROM df_price
        ),
        trends AS (
            SELECT *,
                (avg_volume - LAG(avg_volume) OVER (PARTITION BY symbol ORDER BY date)) / 
                    NULLIF(LAG(avg_volume) OVER (PARTITION BY symbol ORDER BY date), 0) as volume_trend
            FROM metrics
        )
        SELECT * EXCLUDE(ma, avg_volume, volume_trend),
            CASE WHEN close > ma AND volume > avg_volume * {self.volume_threshold} 
                 AND volume_trend > {self.volume_trend_threshold} THEN 1 ELSE 0 END as buy_signal,
            CASE WHEN close < ma OR volume < avg_volume * 0.8 
                 OR volume_trend < -{self.volume_trend_threshold} THEN 1 ELSE 0 END as sell_signal
        FROM trends
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result

class StrategyVolumeDivergence(Strategy):
    def __init__(self, rsi_window: int = 14, volume_window: int = 20, volume_trend_threshold: float = 0.05,
                 price_change_threshold: float = 0.02, rsi_oversold: float = 30, rsi_overbought: float = 70):
        self.rsi_window = rsi_window
        self.volume_window = volume_window
        self.volume_trend_threshold = volume_trend_threshold
        self.price_change_threshold = price_change_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        query = f"""
        WITH diffs AS (
            SELECT *,
                close - LAG(close) OVER (PARTITION BY symbol ORDER BY date) as diff,
                AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS {self.volume_window-1} PRECEDING) as avg_volume
            FROM df_price
        ),
        rsi_calc AS (
            SELECT *,
                AVG(CASE WHEN diff > 0 THEN diff ELSE 0 END) OVER (PARTITION BY symbol ORDER BY date ROWS {self.rsi_window-1} PRECEDING) as gain,
                AVG(CASE WHEN diff < 0 THEN -diff ELSE 0 END) OVER (PARTITION BY symbol ORDER BY date ROWS {self.rsi_window-1} PRECEDING) as loss,
                (avg_volume - LAG(avg_volume) OVER (PARTITION BY symbol ORDER BY date)) / 
                    NULLIF(LAG(avg_volume) OVER (PARTITION BY symbol ORDER BY date), 0) as volume_trend
            FROM diffs
        )
        SELECT * EXCLUDE(diff, avg_volume, gain, loss, volume_trend),
            CASE WHEN (100 - (100 / (1 + (gain / NULLIF(loss, 0))))) < {self.rsi_oversold} 
                 AND volume_trend > {self.volume_trend_threshold} 
                 AND pct_chg < -{self.price_change_threshold} THEN 1 ELSE 0 END as buy_signal,
            CASE WHEN (100 - (100 / (1 + (gain / NULLIF(loss, 0))))) > {self.rsi_overbought} 
                 AND volume_trend < -{self.volume_trend_threshold} 
                 AND pct_chg > {self.price_change_threshold} THEN 1 ELSE 0 END as sell_signal
        FROM rsi_calc
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result

class StrategyMeanReversion(Strategy):
    def __init__(self, ma_window: int = 20, deviation_threshold: float = 0.02, rsi_window: int = 14,
                 rsi_oversold: int = 30, rsi_overbought: int = 70):
        self.ma_window = ma_window
        self.deviation_threshold = deviation_threshold
        self.rsi_window = rsi_window
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def _execute_strategy(self, price: pd.DataFrame) -> pd.DataFrame:
        con = duckdb.connect(database=':memory:')
        con.register('df_price', price)
        
        query = f"""
        WITH basic_metrics AS (
            SELECT *,
                AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS {self.ma_window-1} PRECEDING) as ma,
                close - LAG(close) OVER (PARTITION BY symbol ORDER BY date) as diff
            FROM df_price
        ),
        indicators AS (
            SELECT *,
                (close - ma) / ma as price_deviation,
                AVG(CASE WHEN diff > 0 THEN diff ELSE 0 END) OVER (PARTITION BY symbol ORDER BY date ROWS {self.rsi_window-1} PRECEDING) as gain,
                AVG(CASE WHEN diff < 0 THEN -diff ELSE 0 END) OVER (PARTITION BY symbol ORDER BY date ROWS {self.rsi_window-1} PRECEDING) as loss
            FROM basic_metrics
        )
        SELECT * EXCLUDE(ma, diff, price_deviation, gain, loss),
            CASE WHEN price_deviation < -{self.deviation_threshold} 
                 AND (100 - (100 / (1 + (gain / NULLIF(loss, 0))))) < {self.rsi_oversold} THEN 1 ELSE 0 END as buy_signal,
            CASE WHEN price_deviation > {self.deviation_threshold} 
                 AND (100 - (100 / (1 + (gain / NULLIF(loss, 0))))) > {self.rsi_overbought} THEN 1 ELSE 0 END as sell_signal
        FROM indicators
        ORDER BY symbol, date
        """
        result = con.execute(query).df()
        con.close()
        return result