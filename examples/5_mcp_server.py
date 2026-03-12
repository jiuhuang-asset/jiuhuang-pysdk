# Jiuhuang MCP Server 使用示例
#
# 运行 MCP Server:
#   python -m jiuhuang.mcp
#
# 或者使用 uvx 运行:
#   uvx python -m jiuhuang.mcp
#
# MCP Server 会通过 stdio 通信，需要设置环境变量:
#   JIUHUANG_API_KEY=your_api_key
#   JIUHUANG_API_URL=https://data.jiuhuang.xyz
#
# 使用流程:
#   1. 使用 search_data 搜索需要的数据类型
#   2. 使用 describe_data 查看数据详情和需要的参数
#   3. 使用 get_data 获取数据
#
# 可用工具:
#   - search_data(keyword, top_n): 搜索数据类型
#   - describe_data(data_type): 获取数据类型详细说明
#   - get_data(data_type, params): 通用数据获取

from jiuhuang.mcp import mcp
import asyncio


async def main():
    # 1. 搜索数据
    print("=== 1. search_data ===")
    result = await mcp.call_tool("search_data", {"keyword": "A股 股价 前复权", "top_n": 3})
    print(result.structured_content)

    # 2. 获取数据说明 (会返回 input_params，告诉你需要什么参数)
    print("\n=== 2. describe_data ===")
    result = await mcp.call_tool("describe_data", {"data_type": "stock_zh_a_hist_qfq"})
    data = result.structured_content
    print("data_type:", data.get("data_type"))
    print("input_params:", data.get("input_params", []))
    print("code_example:", data.get("code_example", ""))

    # 3. 获取数据 (根据 describe_data 返回的参数传入)
    print("\n=== 3. get_data ===")
    result = await mcp.call_tool("get_data", {
        "data_type": "stock_zh_a_hist_qfq",
        "params": '{"symbol": "000001", "start": "2025-01-01", "end": "2025-01-10"}'
    })
    data = result.structured_content
    print("count:", data.get("count"))
    print("columns:", data.get("columns", []))
    print("data sample:", data.get("data", [])[:2])

    # 4. 获取其他类型数据
    print("\n=== 4. get_macro_cpi ===")
    result = await mcp.call_tool("get_data", {
        "data_type": "macro_china_cpi",
        "params": '{"start": "2025-01", "end": "2025-12"}'
    })
    data = result.structured_content
    print("count:", data.get("count"))
    print("data:", data.get("data", [])[:2])


if __name__ == "__main__":
    asyncio.run(main())
