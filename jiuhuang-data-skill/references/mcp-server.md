# MCP Server

JiuHuang provides an MCP server for AI agent integration.

## Setup

```bash
# Set environment variables
export JIUHUANG_API_KEY=your_api_key
export JIUHUANG_API_URL=https://data.jiuhuang.xyz

# Run MCP server
python -m jiuhuang.mcp
```

## MCP Tools

The server provides 3 tools:

### 1. search_data

Search available data types:

```python
# MCP call
await mcp.call_tool("search_data", {
    "keyword": "A股 股价",
    "top_n": 5
})
```

### 2. describe_data

Get data type details with input parameters:

```python
# MCP call
await mcp.call_tool("describe_data", {
    "data_type": "stock_zh_a_hist_qfq"
})
```

Returns:
- `data_type`: The data type name
- `input_params`: List of required parameters with types
- `output_fields`: Output field list
- `code_example`: Usage code example
- `full_doc`: Full markdown documentation

### 3. get_data

Fetch data:

```python
# MCP call
await mcp.call_tool("get_data", {
    "data_type": "stock_zh_a_hist_qfq",
    "params": '{"symbol": "000001", "start": "2025-01-01", "end": "2025-01-10"}'
})
```

## Workflow for AI Agents

1. Use `search_data` to find relevant data types
2. Use `describe_data` to understand required parameters
3. Use `get_data` to fetch the data

## Direct Python Usage

You can also use the MCP server directly in Python:

```python
from jiuhuang.mcp import mcp
import asyncio

async def main():
    # Search
    result = await mcp.call_tool("search_data", {"keyword": "A股", "top_n": 3})

    # Describe
    result = await mcp.call_tool("describe_data", {"data_type": "stock_zh_a_hist_qfq"})

    # Get data
    result = await mcp.call_tool("get_data", {
        "data_type": "stock_zh_a_hist_qfq",
        "params": '{"symbol": "000001", "start": "2025-01-01", "end": "2025-01-10"}'
    })

asyncio.run(main())
```
