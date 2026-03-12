# 搜索数据 & 数据详细说明
from jiuhuang.data import JiuhuangData,DataTypes
from pprint import pprint as print

# 搜索数据
jh = JiuhuangData()
results = jh.search_data("A股 股价 前复权", top_n=5)  #  中文名搜索， top_n返回搜索数量
# results = jd.search_data("zh_a_hist_qfq")  #  数据名模糊搜索,不区分大小写

print(results)

# 数据详细说明(调用以及输出)
description_md = jh.describe_data(DataTypes.STOCK_ZH_A_HIST_D_QFQ)

"""控制台输出示例(markdown):
## stock_zh_a_hist-A股股价日数据(不复权)


### 接口描述

接口: stock_zh_a_hist
目标地址: https://quote.eastmoney.com/concept/sh603777.html?from=classic(示例)
描述: 东方财富-沪深京 A 股日频率数据; 历史数据按日频率更新, 当日收盘价请在收盘后获取
限量: 单次返回指定沪深京 A 股上市公司、指定周期和指定日期间的历史行情日频率数据

### 输入参数

| 参数名 | 类型 | 描述 |
|---|---|---|
| symbol | str | symbol='603777'; 股票代码可以在 ak.stock_zh_a_spot_em() 中获取 |
| start | str | 开始日期，格式为YYYY-mm-dd, 示例2025-01-01 |
| end | str | 结束日期，格式为YYYY-mm-dd, 示例2025-01-01 |

### 输出参数

| 字段名 | 类型 | 描述 |
|---|---|---|
| date | object | 日期, 交易日 |
| symbol | object | 股票代码, 不带市场标识的股票代码 |
| open | float64 | 开盘, 开盘价 |
| close | float64 | 收盘, 收盘价 |
| high | float64 | 最高, 最高价 |
| low | float64 | 最低, 最低价 |
| volume | int64 | 成交量, 注意单位: 手 |
| amount | float64 | 成交额, 注意单位: 元 |
| amplitude | float64 | 振幅, 注意单位: % |
| pct_chg | float64 | 涨跌幅, 注意单位: % |
| chg | float64 | 涨跌额, 注意单位: 元 |
| turnover_rate | float64 | 换手率, 注意单位: % |

### 代码示例 (Jiuhuang)

```python
from jiuhuang.data import JiuhuangData, DataTypes

jh_data = JiuhuangData()

data = jh_data.get_data(
    DataTypes.STOCK_ZH_A_HIST,
    symbol="000001"
)

print(data)
```

### 数据示例

| date | symbol | open | close | high | low | volume | amount |
|---|---|---|---|---|---|---|---|
| 2017-03-01 | 000001 | 1575.20 | 1575.20 | 0.83 | 0.10 | 1.63 | 0.21 |
| 2017-03-02 | 000001 | 1578.45 | 1565.45 | 1.24 | -0.62 | -9.75 | 0.24 |
| 2017-03-03 | 000001 | 1562.20 | 1560.57 | 0.73 | -0.31 | -4.88 | 0.20 |
"""