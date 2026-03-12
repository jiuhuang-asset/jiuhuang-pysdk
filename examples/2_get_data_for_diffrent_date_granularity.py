#  为不同的时间颗粒度数据报表限定时间查询范围
from jiuhuang.data import JiuhuangData, DataTypes
from datetime import datetime, timedelta

jh = JiuhuangData()

# akshare的时间范围控制参数一般就是start_date和end_date, 而且格式通常是YYYMMDD
# 而jiuhuang通过统一的start和end参数来筛选数据范围(不同数据源时间颗粒度不同， 不适合都使用date后缀)
# 日数据， 使用时间格式YYYY-MM-DD
stock_price = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_QFQ,  # 前复权
    start="2025-01-01",
    end="2026-02-06",
    symbol="000568", # 多只股票使用英文逗号分隔
)

print(stock_price)

# 月数据, 使用时间格式YYYY-MM
cpi = jh.get_data(
    DataTypes.MACRO_CHINA_CPI, # cpi
    start="2025-01",
    end="2026-02",
)

print(cpi)

# 分钟级数据, 使用时间格式YYYY-MM-DD HH:MM:SS
now = datetime.now()

price_realtime = jh.get_data(
    DataTypes.STOCK_ZH_A_SPOT,  # 股票分钟线
    start=now - timedelta(minutes=10).strftime("%Y-%m-%d %H:%M:%S"), 
    end=now.strftime("%Y-%m-%d %H:%M:%S"),
    symbol="000001",
)
print(price_realtime)


# 如果使用了错误的数据格jiuhuang会抛出异常