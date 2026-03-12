import json
import re
import httpx
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime
from rich.markdown import Markdown
from .data_types import DataTypes, get_data_rename_mapping, get_table_comment

console = Console()

# 参数重命名字典：将 akshare 参数名映射到 jiuhuang 参数名
PARAM_RENAME_MAP = {
    "start_date": "start",
    "end_date": "end",
}

# 需要排除的参数列表（jiuhuang 不需要这些参数）
STOP_ARGUMENTS = ["adjust", "period", "timeout"]


def raise_err_with_details(response, read_body: bool = True) -> None:
    """
    检查响应状态，如果出错则抛出包含具体错误信息的异常。

    替代 response.raise_for_status()，能解析响应体中的错误信息。

    Args:
        response: httpx.Response 对象
        read_body: 是否读取响应体获取错误详情。流式响应应设为 False
    """
    if response.status_code >= 400:
        error_msg = f"HTTP {response.status_code}"
        # 流式响应需要特殊处理：先消费流或使用 text 属性
        if read_body:
            try:
                # 对于流式响应，需要读取完整内容
                if hasattr(response, "stream"):
                    # 流式响应：先读取完整内容
                    error_body = response.read().decode("utf-8")
                else:
                    error_body = response.text
                error_data = json.loads(error_body)
                error_msg = error_data.get("detail", error_body)
            except (json.JSONDecodeError, ValueError):
                error_msg = (
                    error_body if "error_body" in dir() else response.text or error_msg
                )
        raise Exception(f"API error {response.status_code}: {error_msg}")


def rprint(label: str, content: str, add_datetime: bool = True):
    if add_datetime:
        template = (
            "[dim]{}[/dim] [bold blue]{}[/bold blue]: [bold green]{}[/bold green]"
        )
        args = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label, content]
    else:
        template = "[bold blue][{}][/bold blue]: [bold green]{}[/bold green]"
        args = [label, content]

    console.print(template.format(*args))


def highlight_table(data: pd.DataFrame, title: str):
    console.print(f"[bold blue]{title}[/bold blue]")
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("Index", style="dim")
    for col in data.columns:
        table.add_column(str(col), style="dim")

    # Process each row to highlight max and min values
    for idx, row in data.iterrows():
        row_cells = [str(idx)]

        # Find max and min values in the current row (excluding index column)
        numeric_values = pd.to_numeric(row, errors="coerce")
        max_val = numeric_values.max()
        min_val = numeric_values.min()

        for col in data.columns:
            val = row[col]
            str_val = str(round(val, 2))

            # Check if the value is max or min and apply appropriate style
            if pd.api.types.is_numeric_dtype(data[col]) and pd.notna(val):
                if val == max_val:
                    row_cells.append(f"[bold green]{str_val}[/bold green]")
                elif val == min_val:
                    row_cells.append(f"[bold red]{str_val}[/bold red]")
                else:
                    row_cells.append(str_val)
            else:
                # For non-numeric values, just add as is
                row_cells.append(str_val)

        table.add_row(*row_cells)

    console.print(table)


# 辅助函数
def filter_buy_signal_by_fundamental_condition(
    fundamental_data: pd.DataFrame, conditions: str
):
    def filter_func(candidates: pd.DataFrame) -> pd.DataFrame:
        if fundamental_data.empty or conditions == "":
            return candidates

        assert "symbol" in fundamental_data.columns
        data = pd.merge(candidates, fundamental_data, on="symbol", how="left")
        try:
            data = data.query(conditions)
            return data

        except Exception as e:
            rprint(
                "Error",
                "Data Filtering error applying condition'{}': {}".format(
                    conditions, str(e)
                ),
            )
            return candidates

    return filter_func


def filter_st(fundamental_data: pd.DataFrame):
    def filter_func(candidates: pd.DataFrame) -> pd.DataFrame:
        if fundamental_data.empty:
            return candidates

        assert "symbol" in fundamental_data.columns
        data = pd.merge(
            candidates,
            fundamental_data[["symbol", "name"]],
            on="symbol",
            how="left",
        )
        try:
            data = data[~data["name"].str.contains("ST", case=True)]
            return data

        except Exception as e:
            rprint("Error", f"Data Filtering error applying ST filtering: {e.args}")
            return candidates

    return filter_func


def filter_buy_signal_by_industry(
    fundamental_data: pd.DataFrame,
    max_candidate_per_industry: int = 2,
):
    def filter_func(candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        result = candidates.copy()

        # Add industry info if missing
        if "industry" not in result.columns:
            if not fundamental_data.empty and "industry" in fundamental_data.columns:
                result = result.merge(
                    fundamental_data[["symbol", "industry"]],
                    on="symbol",
                    how="left",
                )
            else:
                rprint("Warning:", "No industry information available")
                return result

        # Simple and efficient: group by industry and take top N by score
        result_filtered = (
            result.sort_values("score", ascending=False)
            .groupby("industry")
            .head(max_candidate_per_industry)
            .reset_index(drop=True)
        )

        rprint(
            "Info:",
            f"Industry risk filter: {len(result)} → {len(result_filtered)} candidates",
        )

        # Clean up temporary industry column
        if (
            "industry" in result_filtered.columns
            and "industry" not in candidates.columns
        ):
            result_filtered = result_filtered.drop("industry", axis=1)

        return result_filtered

    return filter_func


# 实现一个标准化
def normalize_score(score_series: pd.Series) -> pd.Series:
    """
    Normalize a score series to have a mean of 1.

    Parameters:
    score_series (pd.Series): Series containing the scores to be normalized.

    Returns:
    pd.Series: Normalized score series with mean = 1.
    """
    # Calculate the mean of the current scores
    mean_score = score_series.mean()

    # Normalize the scores so that the mean becomes 1
    normalized_scores = score_series / mean_score

    return normalized_scores


def adjust_buy_score_by_fundamentals(
    fundamental_data: pd.DataFrame,
    adjust_factors: list[str],
    default_rank_factor: float = 0.1,
):
    def adjust_score_func(candidates: pd.DataFrame):
        result = candidates.copy()

        for factor_name in adjust_factors:
            factor_col = f"{factor_name}_rank_in_industry"
            if factor_col not in fundamental_data.columns:
                rprint(
                    label="Warning:",
                    content=f"Factor {factor_col} not found in fundamental data",
                )
                continue

            # 合并基本面数据
            result = result.merge(
                fundamental_data[["symbol", factor_col]].rename(
                    columns={factor_col: f"{factor_name}_rank"}
                ),
                on="symbol",
                how="left",
            )

            # 计算调整因子（排名越高，因子越接近1）
            result[f"{factor_name}_factor"] = (
                1 - result[f"{factor_name}_rank"]
            ).fillna(default_rank_factor)

            # 应用调整
            result["score"] = (
                result["score"] * result[f"{factor_name}_factor"]
            )  # factor 是0到1
            result["score"] = normalize_score(result["score"])
            # 清理临时列
            result = result.drop(
                [f"{factor_name}_rank", f"{factor_name}_factor"], axis=1
            )

        return result

    return adjust_score_func


def parse_akshare_doc(html_content: str, block_id: str, index_name: str) -> dict:
    """解析 HTML 内容，提取 section 各种字段"""
    # 提取 section 内容
    section_pattern = (
        rf'<section[^>]*id=["\']?{re.escape(block_id)}["\']?[^>]*>(.*?)</section>'
    )
    match = re.search(section_pattern, html_content, re.DOTALL)

    if not match:
        section_pattern = rf'<div[^>]*id=["\']?{block_id}["\']?class=["\']?section["\']?[^>]*>(.*?)</div>'
        match = re.search(section_pattern, html_content, re.DOTALL)

    result = {
        "title": index_name,
        "doc_url": "",
        "description": "",
        "limit": "",
        "input_params": [],
        "output_params": [],
        "code_example": "",
        "data_sample": [],
    }

    if match:
        section_html = match.group(1)
        # 处理代码块
        section_html = re.sub(
            r"<pre><code>(.*?)</code></pre>",
            lambda m: f"\n```\n{m.group(1)}\n```\n",
            section_html,
            flags=re.DOTALL,
        )
        # 移除 HTML 标签
        text = re.sub(r"<[^>]+>", "", section_html)
        # 解码 HTML 实体
        text = text.replace("&nbsp;", " ")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&amp;", "&")
        text = text.replace("&#x2F;", "/")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        lines = text.split("\n")

        # 解析各部分
        current_section = None
        param_table = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "输入参数" in line:
                current_section = "input"
                param_table = []
            elif "输出参数" in line:
                current_section = "output"
                param_table = []
            elif "接口示例" in line:
                current_section = "code"
                result["code_example"] = ""
            elif "数据示例" in line:
                current_section = "data"
                result["data_sample"] = []
            elif current_section == "input" or current_section == "output":
                # 解析参数表格
                if line and not line.startswith("---"):
                    param_table.append(line)
                    if len(param_table) >= 3:
                        # 提取参数名、类型、描述
                        parts = [p.strip() for p in param_table]
                        if len(parts) >= 3 and parts[0] not in ["名称", "类型", "描述"]:
                            param_dict = {
                                "name": parts[0],
                                "type": parts[1],
                                "desc": " ".join(parts[2:]),
                            }
                            if current_section == "input":
                                result["input_params"].append(param_dict)
                            else:
                                result["output_params"].append(param_dict)
                        param_table = []
                elif line.startswith("---"):
                    param_table = []
            elif current_section == "code":
                result["code_example"] += line + "\n"
            elif current_section == "data":
                if line and not line.startswith("..."):
                    result["data_sample"].append(line)

        # 提取描述
        desc_lines = []
        for line in lines:
            if any(x in line for x in ["接口", "目标地址", "描述", "限量"]):
                desc_lines.append(line.strip())
            elif desc_lines and not any(
                x in line for x in ["输入参数", "输出参数", "接口示例", "数据示例"]
            ):
                desc_lines.append(line.strip())
            elif "输入参数" in line:
                break

        result["description"] = "\n".join(desc_lines[:5])

    return result


def fetch_akshare_doc_structured(index_name: str) -> dict:
    """
    从 akshare 官网动态抓取相应数据的介绍，返回结构化数据

    Args:
        index_name: akshare 接口名称，如 fund_info_index_em

    Returns:
        包含各部分数据的字典
    """
    search_url = "https://akshare.akfamily.xyz/_/api/v3/search/"
    params = {"q": f"project:akshare/latest {index_name}"}
    headers = {
        "accept": "*/*",
        "referer": "https://akshare.akfamily.xyz/data/index.html",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    with httpx.Client() as client:
        search_response = client.get(search_url, params=params, headers=headers)
        search_response.raise_for_status()
        search_data = search_response.json()

        results = search_data.get("results", [])
        if not results:
            return {"error": f"未找到 {index_name} 的相关文档"}

        # 优先查找 URL 中包含 /data/ 的结果
        first_result = None
        for r in results:
            path = r.get("path", "")
            if "/data/" in path:
                first_result = r
                break

        # 如果没有找到 /data/ 的结果，使用第一个
        if not first_result:
            first_result = results[0]

        blocks = first_result.get("blocks", [])
        if not blocks:
            return {"error": f"未找到 {index_name} 的相关文档"}

        block = blocks[0]
        doc_path = first_result.get("path", "")
        block_id = block.get("id", "")
        doc_url = f"https://akshare.akfamily.xyz{doc_path}#{block_id}"

        # 请求文档页面
        doc_response = client.get(doc_url, headers=headers)
        doc_response.raise_for_status()
        html_content = doc_response.text

        # 解析 HTML
        return parse_akshare_doc(html_content, block_id, index_name)


def fetch_akshare_doc(index_name: str) -> str:
    """
    从 akshare 官网动态抓取相应数据的介绍（文本格式）

    Args:
        index_name: akshare 接口名称，如 fund_info_index_em

    Returns:
        格式化后的数据介绍文本
    """
    data = fetch_akshare_doc_structured(index_name)

    if "error" in data:
        return data["error"]

    # 转换为文本格式
    lines = [f"=== {data['title']} ===", "", f"文档地址: {data['doc_url']}", ""]

    if data.get("description"):
        lines.append(data["description"])
        lines.append("")

    if data.get("input_params"):
        lines.append("输入参数:")
        for p in data["input_params"]:
            lines.append(f"  {p['name']} ({p['type']}): {p['desc']}")
        lines.append("")

    if data.get("output_params"):
        lines.append("输出参数:")
        for p in data["output_params"]:
            lines.append(f"  {p['name']} ({p['type']}): {p['desc']}")
        lines.append("")

    if data.get("code_example"):
        lines.append("接口示例:")
        lines.append(data["code_example"])

    if data.get("data_sample"):
        lines.append("数据示例:")
        for row in data["data_sample"][:10]:
            lines.append(row)

    return "\n".join(lines)


def  generate_markdown_doc(data: dict, rename_mapping: dict, index_name: str, dt_field_type: str = None, data_type=None) -> str:
    """
    生成 markdown 格式的文档字符串

    Args:
        data: fetch_akshare_doc_structured 返回的结构化数据
        rename_mapping: 字段重命名字典
        index_name: 接口名称
        dt_field_type: 时间字段类型（dt/date/ym）
        data_type: DataTypes 枚举（可选）

    Returns:
        markdown 格式的文档字符串
    """
    data_type = DataTypes(data["title"])
    lines = []

    # 获取时间字段名称
    actual_dt_field = None
    if data_type:
        try:
            from .data_types import get_table_dt_field
            actual_dt_field = get_table_dt_field(data_type)
        except ImportError:
            pass

    # 如果没有传入 dt_field_type，尝试从 data_type 获取
    if not dt_field_type and actual_dt_field:
        dt_field_type = actual_dt_field

    # 时间字段格式映射（包含格式和示例）
    dt_format_map = {
        "dt": ("YYYY-mm-dd HH:MM:SS", "2025-01-01 10:30:00"),
        "date": ("YYYY-mm-dd", "2025-01-01"),
        "ym": ("YYYY-mm", "2025-01"),
    }
    dt_format, dt_example = dt_format_map.get(dt_field_type, ("YYYY-mm-dd", "2025-01-01")) if dt_field_type else (None, None)

    # 标题
    title = f"{index_name}-{get_table_comment(data_type)}"
    if index_name.endswith("_qfq"):
        title += "(前复权)"
    elif index_name.endswith("_hfq"):
        title += "(后复权)"
    lines.append(f"## {title}")
    lines.append("")
    # lines.append(f"**文档地址**: {data.get('doc_url', '')}")
    lines.append("")

    # 描述
    if data.get("description"):
        lines.append("### 接口描述")
        lines.append("")
        lines.append(data["description"])
        lines.append("")

    # 输入参数
    input_params = list(data.get("input_params", [])) if data.get("input_params") else []

    # 如果有 dt_field_type，更新或添加 start/end 参数的描述
    if dt_field_type and dt_format and dt_example:
        # 查找并更新 start_date 参数
        for p in input_params:
            if p.get("name") == "start_date":
                p["name"] = "start"
                p["desc"] = f"开始日期，格式为{dt_format}, 示例{dt_example}"
            elif p.get("name") == "end_date":
                p["name"] = "end"
                p["desc"] = f"结束日期，格式为{dt_format}, 示例{dt_example}"
        # 如果没有 start_date/end_date，添加它们
        has_start = any(p.get("name") == "start" for p in input_params)
        has_end = any(p.get("name") == "end" for p in input_params)
        if not has_start:
            input_params.append({
                "name": "start",
                "type": "str",
                "desc": f"开始日期，格式为{dt_format}, 示例{dt_example}"
            })
        if not has_end:
            input_params.append({
                "name": "end",
                "type": "str",
                "desc": f"结束日期，格式为{dt_format}, 示例{dt_example}"
            })

    # 过滤输入参数：参数名必须是英文、长度<12、不在STOP_ARGUMENTS中
    valid_input_params = []
    for p in input_params:
        param_name = p.get("name", "")
        # 跳过无效的参数名
        if not param_name or param_name == "-":
            continue
        # 参数名必须是英文且长度<12
        if not param_name.isascii() or len(param_name) >= 12:
            continue
        # 排除 STOP_ARGUMENTS 中的参数
        if param_name in STOP_ARGUMENTS:
            continue
        valid_input_params.append(p)

    if valid_input_params:
        lines.append("### 输入参数")
        lines.append("")
        lines.append("| 参数名 | 类型 | 描述 |")
        lines.append("|---|---|---|")
        for p in valid_input_params:
            param_name = p.get("name", "")
            param_name = PARAM_RENAME_MAP.get(param_name, param_name)
            desc = p.get("desc", "").replace("|", "\\|")
            lines.append(f"| {param_name} | {p.get('type', '')} | {desc} |")
        lines.append("")

    # 输出参数
    zh_to_en =  get_data_rename_mapping(data_type)
    if data.get("output_params"):

        lines.append("### 输出参数")
        lines.append("")
        lines.append("| 字段名 | 类型 | 描述 |")
        lines.append("|---|---|---|")
        for p in data["output_params"]:
            zh_name = p.get("name", "")
            en_name = zh_to_en.get(zh_name, "-")
            # 字段名保持英文字段名
            # 描述 = {中文字段说明}, {原来akshare官网的输出字段描述}
            original_desc = p.get("desc", "")
            desc = f"{zh_name}"
            if original_desc and original_desc != "-":
                desc = f"{zh_name}, {original_desc}"
            lines.append(f"| {en_name} | {p.get('type', '')} | {desc} |")
        lines.append("")

    # 代码示例
    code_example = generate_jiuhuang_code(index_name, data.get("input_params", []), STOP_ARGUMENTS, dt_field_type)
    if code_example:
        lines.append("### 代码示例 (Jiuhuang)")
        lines.append("")
        lines.append("```python")
        lines.append(code_example)
        lines.append("```")
        lines.append("")

    # 数据示例
    if data.get("data_sample") and rename_mapping:
        lines.append("### 数据示例")
        lines.append("")

        # 解析数据行（最多3行）
        sample_rows = []
        for row in data["data_sample"][1:4]:  # 只取3行
            cols = [c.strip() for c in row.split() if c.strip()]
            if cols:
                # 跳过第一列（index），保留后面列
                sample_rows.append(cols[1:])

        if not sample_rows:
            sample_rows = []

        # 找出哪些列有...并排除
        col_has_ellipsis = set()
        for row in sample_rows:
            for i, c in enumerate(row):
                if c == "...":
                    col_has_ellipsis.add(i)

        # 过滤掉包含...的列
        filtered_rows = []
        for row in sample_rows:
            filtered_row = [c for i, c in enumerate(row) if i not in col_has_ellipsis]
            filtered_rows.append(filtered_row)

        # 使用默认表头
        default_headers = ["date", "symbol", "open", "close", "high", "low", "volume", "amount"]
        num_cols = len(filtered_rows[0]) if filtered_rows else 8
        converted_headers = default_headers[:num_cols]

        # 生成表头
        lines.append("| " + " | ".join(converted_headers) + " |")
        lines.append("|" + "|".join(["---" for _ in converted_headers]) + "|")

        # 生成数据行（过滤后的）
        for row in filtered_rows:
            lines.append("| " + " | ".join(row) + " |")

        lines.append("")

    return "\n".join(lines)


def pretty_print_doc(data: dict, rename_mapping: dict, index_name: str, data_type=None) -> str:
    """
    美化打印 akshare 文档数据

    Args:
        data: fetch_akshare_doc_structured 返回的结构化数据
        rename_mapping: 字段重命名字典
        index_name: 接口名称
        data_type: DataTypes 枚举（可选）

    Returns:
        markdown 格式的文档字符串
    """
    # 获取时间字段类型
    dt_field_type = None
    if data_type:
        try:
            from .data_types import get_table_dt_field
            dt_field_type = get_table_dt_field(data_type)
        except ImportError:
            pass

    # 时间字段格式映射（包含格式和示例）
    dt_format_map = {
        "dt": ("YYYY-mm-dd HH:MM:SS", "2025-01-01 10:30:00"),
        "date": ("YYYY-mm-dd", "2025-01-01"),
        "ym": ("YYYY-mm", "2025-01"),
    }
    dt_format, dt_example = dt_format_map.get(dt_field_type, ("YYYY-mm-dd", "2025-01-01")) if dt_field_type else (None, None)

    # 先生成 markdown 字符串
    markdown_doc = generate_markdown_doc(data, rename_mapping, index_name, dt_field_type, data_type)

    # 再打印到控制台
    console = Console()

    console.print(Markdown(markdown_doc))

    return markdown_doc

  


def generate_jiuhuang_code(
    index_name: str, input_params: list, stop_arguments: list = None, dt_field_type: str = None
) -> str:
    """生成 jiuhuang 代码示例

    Args:
        index_name: 接口名称
        input_params: 输入参数列表
        stop_arguments: 要排除的参数列表
        dt_field_type: 时间字段类型（dt/date/ym），用于生成正确格式的 start/end 示例值
    """
    if stop_arguments is None:
        stop_arguments = STOP_ARGUMENTS

    # 时间字段格式映射
    dt_format_map = {
        "dt": "YYYY-mm-dd HH:MM:SS",
        "date": "YYYY-mm-dd",
        "ym": "YYYY-mm",
    }
    dt_format = dt_format_map.get(dt_field_type, "YYYY-mm-dd") if dt_field_type else "YYYY-mm-dd"

    lines = []
    lines.append("from jiuhuang.data import JiuhuangData, DataTypes")
    lines.append("")
    lines.append("jh_data = JiuhuangData()")
    lines.append("")

    # 构建参数
    params = []

    # 检查是否已有 start_date 或 end_date 参数
    has_start = any(p.get("name") in ["start_date", "start"] for p in input_params)
    has_end = any(p.get("name") in ["end_date", "end"] for p in input_params)

    # 示例日期值映射（基于 dt_field_type）
    example_date_map = {
        "dt": "2025-01-01 00:00:00",
        "date": "2025-01-01",
        "ym": "2025-01",
    }
    example_date = example_date_map.get(dt_field_type, "2025-01-01") if dt_field_type else "2025-01-01"

    # 如果有 dt_field_type 且原始文档没有 start/end，添加它们
    if dt_field_type and not has_start:
        params.append(f'start="{example_date}"')
    if dt_field_type and not has_end:
        params.append(f'end="{example_date}"')

    for p in input_params:
        param_name = p.get("name", "")
        # 跳过 stop_arguments 中的参数
        if param_name in stop_arguments:
            continue
        if param_name and param_name != "-":
            # 使用 PARAM_RENAME_MAP 转换参数名
            param_name = PARAM_RENAME_MAP.get(param_name, param_name)

            # 根据参数名生成示例值
            if param_name == "symbol":
                params.append(f'{param_name}="000001"')
            elif param_name in ["start", "end", "date"]:
                # 避免重复添加 start/end
                if param_name == "start" and has_start:
                    continue
                if param_name == "end" and has_end:
                    continue
                params.append(f'{param_name}="{dt_format}"')

    # 生成调用代码
    # 从 index_name 获取 DataTypes 枚举名称
    enum_name = index_name.upper()

    lines.append("data = jh_data.get_data(")
    lines.append(f"    DataTypes.{enum_name},")

    if params:
        for i, p in enumerate(params):
            if i < len(params) - 1:
                lines.append(f"    {p},")
            else:
                lines.append(f"    {p}")
    lines.append(")")

    lines.append("")
    lines.append("print(data)")

    return "\n".join(lines)


# ============ 搜索相关函数 ============

# 数据类型前缀映射
DATA_TYPE_PREFIX = {
    "STOCK": "stock_",
    "FACTOR": "factor_",
    "FUND": "fund_",
    "INDEX": "index_",
    "MACRO": "macro_",
}

# 每种数据类型对应的优先匹配关键词
PRIORITY_KEYWORDS = {
    "STOCK": ["股票", "股价", "A股", "港股", "美股", "上证", "深证", "创业板", "科创板", "北交所", "沪深", "大盘", "个股", "涨停", "跌停", "复权", "未复权"],
    "FACTOR": ["因子", "暴露", "收益率", "因子收益", "因子暴露", "Alpha", "Beta", "风险因子"],
    "FUND": ["基金", "ETF", "FOF", "LOF", "私募", "公募", "持仓", "净值"],
    "INDEX": ["指数", "股指", "指数行情", "PMI", "纳指", "标普", "道琼斯", "恒生", "日经"],
    "MACRO": ["宏观", "GDP", "CPI", "PPI", "利率", "货币", "M2", "社融", "通胀", "降息", "加息", "外汇", "黄金", "农产品", "大宗商品"],
}


def _levenshtein_distance(s1: str, s2: str) -> int:
    """计算两个字符串之间的编辑距离"""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def _get_type_from_prefix(dt: DataTypes) -> str | None:
    """根据 DataTypes 的前缀确定数据类型"""
    for type_name, prefix in DATA_TYPE_PREFIX.items():
        if dt.value.startswith(prefix):
            return type_name
    return None


def _match_type_priority(keyword: str) -> list[str]:
    """根据关键词确定优先匹配的数据类型列表"""
    keyword_lower = keyword.lower()
    matched_types = []
    for type_name, keywords in PRIORITY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in keyword_lower:
                matched_types.append(type_name)
                break
    return matched_types


def _calc_edit_distance_score(keyword: str, name: str, comment: str) -> float:
    """使用编辑距离计算相似度分数"""
    keyword_lower = keyword.lower()
    name_lower = name.lower()
    comment_lower = comment.lower()

    # 1. 完全匹配
    if keyword_lower == name_lower:
        return 1.0

    # 2. 名称以关键词开头（正向包含：keyword in name）
    if keyword_lower in name_lower:
        return 0.9

    # 3. 名称被关键词包含（反向包含：name in keyword，即关键词是名称的前缀+更多）
    if name_lower in keyword_lower:
        return 0.85

    # 4. 注释包含匹配
    if keyword_lower in comment_lower:
        return 0.8

    # 5. 编辑距离计算（名称）
    max_len_name = max(len(keyword_lower), len(name_lower))
    if max_len_name > 0:
        name_score = 1 - (_levenshtein_distance(keyword_lower, name_lower) / max_len_name)
        if name_score >= 0.5:
            return name_score * 0.7

    # 6. 编辑距离计算（注释，取前50字符）
    comment_short = comment_lower[:50]
    max_len_comment = max(len(keyword_lower), len(comment_short))
    if max_len_comment > 0:
        comment_score = 1 - (_levenshtein_distance(keyword_lower, comment_short) / max_len_comment)
        if comment_score >= 0.5:
            return comment_score * 0.5

    # 7. 短关键词任意匹配
    if len(keyword_lower) >= 3:
        return 0.2

    return 0


def search_datatypes(keyword: str, top_n: int = 10) -> list[DataTypes]:
    """
    根据关键词搜索 DataTypes，按匹配度排序返回最相似的若干结果。

    Args:
        keyword: 搜索关键词
        top_n: 返回结果数量

    Returns:
        按匹配度从高到低排序的 DataTypes 列表
    """
    if not keyword or not keyword.strip():
        return []

    keyword = keyword.strip()

    # 确定优先类型
    priority_types = _match_type_priority(keyword)

    # 计算相似度
    results = []
    for dt in DataTypes:
        comment = get_table_comment(dt)
        score = _calc_edit_distance_score(keyword, dt.value, comment)
        if score > 0:
            dt_type = _get_type_from_prefix(dt)
            if priority_types and dt_type in priority_types:
                score += 0.5
            results.append((dt, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return [dt for dt, _ in results[:top_n]]
