import time
from jiuhuang.data import JiuhuangData


jiuhuang = JiuhuangData(api_key="pat_3jVobVyI0kgCt87xKrrlfjkAHgxUI4WI", sync=True)

# data_types = jiuhuang.get_offline_data_types()


start = time.time()
stock_price = jiuhuang.get_data(
    "stock_zh_a_hist_m",
    # symbol="600135",
    start_date="2025-02-01",
    end_date="2026-02-11",
    stream=True,
)
print(stock_price)
end = time.time()

print(f"耗时：{end - start}s")
