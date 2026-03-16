import time
from jiuhuang.data import JiuhuangData, DataTypes
from pprint import pprint as print

jd = JiuhuangData()
# results = jd.search_data("A股 股价 前复权", top_n=5)

# print(results)


jd.describe_data(DataTypes.STOCK_CYQ_EM)