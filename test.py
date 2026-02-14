import webview
import pandas as pd
import numpy as np
import json

# --- 模拟数据生成 ---
def get_mock_df():
    symbols = ['000001', '000002', '600000']
    strategies = ['turtle', 'sma', 'mean_reversion']
    data = []
    for s in symbols:
        for st in strategies:
            dates = pd.date_range("2023-01-01", periods=100)
            base = np.random.randint(10, 100)
            prices = base + np.cumsum(np.random.randn(100))
            for i, d in enumerate(dates):
                data.append({
                    'date': d.strftime('%Y-%m-%d'),
                    'symbol': s, 'strategy': st,
                    'open': prices[i], 'close': prices[i] + np.random.randn(),
                    'high': prices[i] + 2, 'low': prices[i] - 2,
                    'buy_signal': 1 if np.random.rand() > 0.95 else 0,
                    'sell_signal': 1 if np.random.rand() > 0.95 else 0
                })
    return pd.DataFrame(data)

def get_performance_df():
    data = {
        'symbol': ['000001', '000002', '600000'],
        'turtle': [0.01, -0.21, 0.05],
        'sma': [0.06, -0.12, 0.08],
        'mean_reversion': [0.03, -0.12, -0.02]
    }
    return pd.DataFrame(data)

class Api:
    def init_data(self):
        df = get_mock_df()
        print(len(df))
        perf = get_performance_df()
        return {
            'main_data': df.to_dict(orient='records'),
            'perf_data': perf.to_dict(orient='records')
        }

if __name__ == '__main__':
    api = Api()
    # 允许 Python 与 JS 交互
    window = webview.create_window('量化策略看板', 'index.html', js_api=api)
    webview.start()