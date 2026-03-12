from jiuhuang.data import JiuhuangData,DataTypes
from pprint import pprint 

jh = JiuhuangData()

md = jh.describe_data(DataTypes.STOCK_ZH_A_HIST)

print(md)