import webview
import pandas as pd
import os 

class BacktestingView:
    def __init__(self, main_data:pd.DataFrame, perf_data:pd.DataFrame):
        self.main_data = main_data[["symbol", "date", "open","high", "low", "close", "volume", "buy_signal", "sell_signal", "strategy"]].assign(date=lambda x: x['date'].dt.strftime('%Y-%m-%d')).to_dict(orient='records')        
        
        self.perf_data = perf_data.reset_index().to_dict(orient='records')
    def init_data(self):
        return {
            'main_data': self.main_data,
            'perf_data': self.perf_data
        }
    

def display_backtesting(main_data, pref_data):
    api = BacktestingView(main_data, pref_data)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(current_dir, 'front_src', 'bt.html')

    window = webview.create_window('回测结果看板', html_path, js_api=api)
    webview.start()
