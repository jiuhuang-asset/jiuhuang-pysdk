"""
Jiuhuang MCP Server - 使用 FastMCP 实现 MCP 协议

提供以下工具:
- search_data: 搜索可用数据类型
- describe_data: 获取数据类型详细说明（含输入参数）
- get_data: 通用数据获取工具

使用流程:
    1. 使用 search_data 搜索需要的数据类型
    2. 使用 describe_data 查看数据详情和参数
    3. 使用 get_data 获取数据

运行方式:
    python -m jiuhuang.mcp

或使用 MCP 客户端连接:
    from jiuhuang.mcp import mcp
"""

import os
import re
from typing import Optional

from fastmcp import FastMCP
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 创建 MCP Server
mcp = FastMCP("Jiuhuang Data API")

# 全局 JiuhuangData 实例
_jh_instance = None


def _get_jiuhuang():
    """获取或创建 JiuhuangData 实例"""
    global _jh_instance
    if _jh_instance is None:
        from jiuhuang.data import JiuhuangData
        _jh_instance = JiuhuangData()
    return _jh_instance


def _parse_input_params(markdown_doc: str) -> list[dict]:
    """
    从 describe_data 返回的 markdown 文档中解析输入参数。

    Args:
        markdown_doc: describe_data 返回的 markdown 文档

    Returns:
        参数列表，每项包含 name, type, description
    """
    params = []
    # 匹配参数表格行 - 格式: | 参数名 | 类型 | 描述 |
    # 先找到输入参数部分
    pattern = r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(.+?)\s*\|'

    # 直接在整个文档中搜索
    matches = re.findall(pattern, markdown_doc)
    for name, param_type, desc in matches:
        if name not in ['字段名', '参数名', 'date', 'symbol', 'open', 'close', 'high', 'low', 'volume', 'amount']:
            params.append({
                "name": name.strip(),
                "type": param_type.strip(),
                "description": desc.strip()
            })

    return params


@mcp.tool()
async def search_data(
    keyword: str,
    top_n: int = 5
) -> list[dict[str, str]]:
    """
    搜索数据接口，根据关键词搜索可用的数据类型。

    使用流程:
        1. 用 keyword 搜索需要的数据类型
        2. 查看返回的 DataType 名称
        3. 使用 describe_data 获取详细信息

    Args:
        keyword: 搜索关键词，例如 "A股 股价 前复权" 或 "zh_a_hist_qfq"
        top_n: 返回结果数量，默认 5

    Returns:
        匹配的 DataTypes 列表，每项包含类型名称和中文描述

    Examples:
        >>> search_data("A股 股价 前复权", top_n=5)
        >>> search_data("zh_a_hist_qfq")
    """
    jh = _get_jiuhuang()
    results = jh.search_data(keyword=keyword, top_n=top_n)
    return results


@mcp.tool()
async def describe_data(
    data_type: str
) -> dict:
    """
    获取数据详细说明，包括接口描述、输入参数、输出参数和代码示例。

    使用流程:
        1. 先用 search_data 找到需要的数据类型
        2. 用 describe_data 查看详细说明
        3. 从返回的 input_params 中了解需要传入哪些参数

    Args:
        data_type: 数据类型名称，例如 "stock_zh_a_hist" 或 "STOCK_ZH_A_HIST_QFQ"

    Returns:
        包含详细信息的字典:
        - description: 接口描述
        - input_params: 输入参数列表 (name, type, description)
        - output_fields: 输出字段列表 (name, type, description)
        - code_example: 代码示例

    Examples:
        >>> describe_data("STOCK_ZH_A_HIST_QFQ")
    """
    from jiuhuang.data import DataTypes

    # 转换为枚举（尝试多种格式）
    dt = None
    for fmt in [data_type.lower(), f"{data_type.lower()}_qfq", f"{data_type.lower()}_hfq"]:
        try:
            dt = DataTypes(fmt)
            break
        except ValueError:
            continue

    if dt is None:
        return {
            "error": f"Unknown data type '{data_type}'",
            "suggestion": "Use search_data to find available data types"
        }

    jh = _get_jiuhuang()
    markdown_doc = jh.describe_data(dt)

    # 解析输入参数
    input_params = _parse_input_params(markdown_doc)

    # 提取输出字段（简化处理）
    output_fields = []
    output_section_match = re.search(r'### 输出参数\s*\|[\s\-]+\|([\s\S]*?)(?=###|$)', markdown_doc)
    if output_section_match:
        pattern = r'\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(.+?)\s*\|'
        matches = re.findall(pattern, output_section_match.group(1))
        for name, field_type, desc in matches:
            output_fields.append({
                "name": name.strip(),
                "type": field_type.strip(),
                "description": desc.strip()
            })

    # 提取代码示例
    code_example = ""
    code_match = re.search(r'```python\n([\s\S]*?)```', markdown_doc)
    if code_match:
        code_example = code_match.group(1).strip()

    return {
        "data_type": dt.value,
        "input_params": input_params,
        "output_fields": output_fields[:10],  # 限制数量
        "code_example": code_example,
        "full_doc": markdown_doc
    }


@mcp.tool()
async def get_data(
    data_type: str,
    params: Optional[str] = None
) -> dict:
    """
    通用数据获取工具。

    使用流程:
        1. 使用 search_data 搜索数据类型
        2. 使用 describe_data 查看需要的参数
        3. 传入 data_type 和对应参数获取数据

    Args:
        data_type: 数据类型，例如 "stock_zh_a_hist_qfq", "macro_china_cpi" 等
        params: JSON 格式参数字符串，例如 '{"symbol": "000001", "start": "2025-01-01", "end": "2025-01-10"}'

    Returns:
        包含数据的字典:
        - data: 数据列表
        - columns: 字段名列表
        - count: 数据条数

    Examples:
        # 获取股票数据
        >>> get_data("stock_zh_a_hist_qfq", '{"symbol": "000001", "start": "2025-01-01", "end": "2025-01-10"}')

        # 获取CPI数据
        >>> get_data("macro_china_cpi", '{"start": "2025-01", "end": "2025-12"}')

        # 获取股票基本信息
        >>> get_data("stock_individual_info_em", '{"symbol": "000001"}')
    """
    import json
    from jiuhuang.data import DataTypes

    # 解析参数
    kwargs = {}
    if params:
        try:
            kwargs = json.loads(params)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON params format"}

    # 转换为枚举
    dt = None
    for fmt in [data_type.lower(), f"{data_type.lower()}_qfq", f"{data_type.lower()}_hfq"]:
        try:
            dt = DataTypes(fmt)
            break
        except ValueError:
            continue

    if dt is None:
        return {
            "error": f"Unknown data type '{data_type}'",
            "suggestion": "Use search_data to find available data types"
        }

    jh = _get_jiuhuang()

    try:
        df = jh.get_data(dt, **kwargs)

        if df is None or df.empty:
            return {"error": "No data returned", "data": []}

        return {
            "data": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "count": len(df)
        }
    except Exception as e:
        return {"error": str(e), "data_type": data_type, "params": kwargs}


def run():
    """运行 MCP Server"""
    # 检查环境变量
    api_key = os.getenv("JIUHUANG_API_KEY")
    api_url = os.getenv("JIUHUANG_API_URL")

    if not api_key:
        print("Warning: JIUHUANG_API_KEY environment variable not set")
    if not api_url:
        print("Warning: JIUHUANG_API_URL environment variable not set")

    print("Starting Jiuhuang MCP Server...")
    print(f"API URL: {api_url or 'https://data.jiuhuang.xyz'}")
    print("\nAvailable tools:")
    print("  - search_data: 搜索数据类型")
    print("  - describe_data: 获取数据类型详细说明")
    print("  - get_data: 通用数据获取")
    mcp.run()


if __name__ == "__main__":
    run()
