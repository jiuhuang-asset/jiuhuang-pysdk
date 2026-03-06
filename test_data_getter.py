from jiuhuang.data import JiuhuangData
import time


jh_data = JiuhuangData(sync=False)

start = time.time()
data = jh_data.get_data("stock_zh_a_hist_d", start="2026-01-01", end="2026-03-03")
data = jh_data.get_data("stock_individual_info_em")
print(data)
end = print(time.time() - start)
