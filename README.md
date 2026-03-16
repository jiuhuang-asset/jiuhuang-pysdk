![banner](./assets/banner.png)
# jiuhuang-pysdk

jiuhuang（韭皇）是一个**免费**,高性能,简洁易用的金融数据获取和回测框架。

- **官网&API申请**: https://jiuhuang.xyz  
- **文档地址**: https://doc.jiuhuang.xyz

## 亮点

- **丰富的数据源**：兼容akshare多种数据类型，支持获取A股、基金、宏观等数据
- **统一的数据接口**：通过 `DataTypes` 枚举类统一管理数据类型，输出字段名标准化为英文字段名
- **多时间颗粒度支持**：支持日、周、月级别数据，以及分钟级实时数据
- **智能数据搜索**：内置中文语义搜索功能，快速找到所需数据类型
- **多进程并行回测**：内置多进程并行计算，回测速度极快
- **内置多种策略**：提供海龟交易、均线交叉、RSI、布林带、动量等 11+ 种经典策略
- **可视化回测仪表盘**：交互式图表展示回测结果，支持交易历史、策略分布、排名对比等
- **易于扩展**：支持自定义策略
- **MCP Server 支持**：可作为 MCP 服务器运行，AI Agent 可通过标准协议调用
- **Claude Code Skill 集成**：提供 Skill 配置，Claude Code 可直接使用 

## 快速开始

### 安装
uv安装(推荐)
```bash
uv add jiuhuang-pysdk
# 或者uv pip install -U jiuhuang-pysdk
```

pip安装
```bash
pip install -U jiuhuang-pysdk
```


### 回测示例

```python
from jiuhuang.data import JiuhuangData, DataTypes
from jiuhuang.strategy import *
from jiuhuang.backtest import backtest
from jiuhuang.dash import display_backtesting
import warnings

warnings.filterwarnings("ignore")

# 方式一：使用环境变量（推荐）
# 设置 API_URL 和 API_KEY 环境变量
# API_KEY申请地址：https://jiuhuang.xyz
jh = JiuhuangData()  
# 方式二：直接传入参数
jh = JiuhuangData(api_url="https://data.jiuhuang.xyz", api_key="你的API KEY")


# 定义策略（可使用内置策略或自定义策略）
strategies = {
    "海龟": StrategyTurtle(entry_window=20, exit_window=10),
    "移动均线交叉": StrategyMovingAverageCrossover(12, 24),
    "买入持有": StrategyBuyAndHold(),
}

# 获取数据
symbols = ["000001", "600036", "600519", "000858", "601318", "000002"]
stock_price = jh.get_data(
    DataTypes.STOCK_ZH_A_HIST_QFQ,
    start="2024-12-25",
    end="2026-03-11",
    symbol=",".join(symbols),
)
stock_info = jh.get_data(DataTypes.STOCK_INDIVIDUAL_INFO_EM)

# 执行回测
trading_history, backtest_perf = backtest(
    strategies,
    stock_price,
    stock_info,
)

# 展示回测仪表盘
display_backtesting(trading_history, backtest_perf)
```

### 回测仪表盘预览

| 策略对比 | 策略分布 |
|---------|---------|
| ![策略对比](./assets/strat_compare_resized.png) | ![策略分布](./assets/strat_dist_resized.png) |

| 交易历史 | 策略排名 |
|---------|---------|
| ![交易历史](./assets/trading_history_resized.png) | ![策略排名](./assets/strat_ranking_resized.png) |


## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
