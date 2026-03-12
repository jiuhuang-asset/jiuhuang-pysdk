# 单一symbol股价获取
from jiuhuang.data import JiuhuangData, DataTypes

jh = JiuhuangData()

# 单一股票数据获取
stock_price = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_D_QFQ, # 前复权
    start="2025-01-01",
    end="2026-02-06",
    symbol="000001",
)

print

# 多只股票数据获取
symbols = [
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

# jiuhuang 兼容了很多akshare数据类型, DataTypes(枚举类)对应了ak.xxxx()
# 比如akshare获取股票日K线数据akshare.stock_zh_a_hist(symbol="000001", start_date="2025-01-01", end_date="2026-02-06", adjust="qfq")
# 不同点在于， akshare通过adjust参数控制复权方式，而jiuhuang通过DataTypes(枚举类)有无后缀进行区分  
# 另外jiuhuang中的输出DataFrame 都是字段命名标准化后的英文字段名
stock_price = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_QFQ,  # 前复权
    start="2025-01-01",
    end="2026-02-06",
    symbol=",".join(symbols), # 多只股票使用英文逗号分隔
)

print(stock_price)


""" 输出结果示例
 date        symbol  open     close    high  low    volume  amount
 ─────────────────────────────────────────────────────────────────
 2017-03-01  000001  1575.20  1575.20  0.83  0.10   1.63    0.21
 2017-03-02  000001  1578.45  1565.45  1.24  -0.62  -9.75   0.24
 2017-03-03  000001  1562.20  1560.57  0.73  -0.31  -4.88   0.20
"""
