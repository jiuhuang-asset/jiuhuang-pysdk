import re
import httpx
import json
import sys
import pandas as pd
from os import getenv
from dotenv import load_dotenv
import os
import duckdb
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich import print as rprint
from .utils import (
    raise_err_with_details,
    fetch_akshare_doc_structured,
    pretty_print_doc,
    search_datatypes,
)
from .data_types import (
    DataTypes,
    get_table_fields,
    get_table_unique_keys,
    get_data_rename_mapping,
    get_table_comment
)

load_dotenv()

__all__ = ["JiuhuangData", "DataTypes"]


class JiuhuangData:
    def __init__(
        self,
        api_key: str = getenv("JIUHUANG_API_KEY"),
        api_url: str = getenv("JIUHUANG_API_URL", "https://data.jiuhuang.xyz"),
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._prepare_client(api_key)
        self._cache = _DataCache(jd=self)

    def _prepare_client(self, api_key: str):
        client = httpx.Client(timeout=180)
        client.headers.update({"Authorization": f"Bearer {api_key}"})
        self._client = client

    def search_data(self, keyword: str, top_n: int = 5) -> list[dict[str, str]]:
        """
        根据关键词搜索 DataTypes，按匹配度排序返回最相似的若干结果。

        Args:
            keyword: 搜索关键词
            top_n: 返回结果数量，默认 5

        Returns:
            按匹配度从高到低排序的 DataTypes 列表
        """
        dts = search_datatypes(keyword, top_n)

        return [{d.value.upper(): get_table_comment(d)} for d in dts]

    def describe_data(self, data_type: DataTypes, return_md:bool=False) -> str:
        """
        从 akshare 官网动态抓取相应数据的介绍，使用 rich 打印，同时返回 markdown 格式文档

        Args:
            data_type: 数据类型
            return_md: 是否返回 markdown 格式文档

        Returns:
            markdown 格式的文档字符串
        """
        from rich.console import Console

        console = Console()
        index_name = data_type.value

        # 处理 _qfq/_hfq 后缀
        base_index_name = index_name
        if index_name.endswith("_qfq") or index_name.endswith("_hfq"):
            base_index_name = index_name.rsplit("_", 1)[0]

        data = fetch_akshare_doc_structured(base_index_name)

        if "error" in data:
            console.print(f"[red]{data['error']}[/red]")
            return ""

        # 获取字段映射
        rename_mapping = get_data_rename_mapping(data_type)

        # 调用 pretty_print_doc 打印，并获取返回的 markdown
        markdown_doc = pretty_print_doc(data, rename_mapping, index_name, data_type)

        if return_md:
            return markdown_doc

    def get_data_total(
        self,
        data_type: DataTypes,
        **kwargs,
    ):
        payload = {
            "data_type": data_type.value,
        }
        payload.update(kwargs)
        response = self._client.post(f"{self.api_url}/data-offline/total", json=payload)

        raise_err_with_details(response)
        total = response.json()["data"]
        try:
            total = int(total)
        except Exception as e:
            raise e

        return total

    def get_data(
        self,
        data_type: DataTypes,
        remote: bool = False,
        **kwargs,
    ):
        """
        从 JiuHuang API 获取离线数据，根据指定参数获取历史数据。

        Args:
            data_type: Data
            remote: 是否强制从远程获取
        """
        data = None
        remote_data_count = self.get_data_total(data_type=data_type, **kwargs)
        if remote_data_count == 0:
            rprint("[bold yellow]没有数据, 请检查参数")
            return
        if not remote:
            kw = {k: v for k, v in kwargs.items() if k != "remote"}
            data = self._cache.get_data(data_type, **kw)
            if len(data) == remote_data_count:
                return data
        
        rprint(f"[cyan]Pulling data from JiuHuang API...[/cyan]")
        url = f"{self.api_url}/data-offline/"
        payload = {
            "data_type": data_type.value,
        }
        payload.update(kwargs)
        all_data = []
        with self._client.stream("POST", url, json=payload) as response:
            raise_err_with_details(response, read_body=True)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            ) as progress:
                task_id = progress.add_task(
                    f"[cyan]Downloading {data_type}...", total=remote_data_count
                )
                for chunk in response.iter_lines():
                    if chunk:
                        try:
                            resp = json.loads(chunk)
                            data = resp["data"]
                            all_data.extend(data)
                            progress.update(task_id, completed=len(all_data))
                        except json.JSONDecodeError as e:
                            raise Exception("获取数据失败[Json解码错误]")
        data = pd.DataFrame(all_data)
        self._cache.bulk_import(data_type, data)
        kwargs.pop("remote", None)
        return self._cache.get_data(data_type, **kwargs)


class _DataCache:
    def __init__(self, jd: JiuhuangData):
        self.cache_dir = os.path.expanduser("~/.jiuhuang")
        self.cache_db_path = os.path.join(self.cache_dir, "cache_data.db")
        self._jd = jd
        self._initialize_cache()

    def _initialize_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            conn = duckdb.connect(self.cache_db_path)
        except:
            rprint("[red]Warning: 请确保只有单个正在运行的JiuhuangData实例[/red]")
            sys.exit(1)

        conn.close()

    def _init_table(self, data_type: DataTypes):
        """按需初始化表：如果表不存在则从API获取DDL并创建"""
        table_name = data_type.value
        conn = duckdb.connect(self.cache_db_path)
        try:
            # 检查表是否已存在
            result = conn.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0]
            if result > 0:
                conn.close()
                return

            # 表不存在，从API获取DDL
            resp = self._jd._client.get(
                self._jd.api_url + "/data-offline/ddl",
                params={"data_types": [data_type.value]},
            )
            if resp.status_code == 200:
                ddl_dict = resp.json()["data"]
                ddl = ddl_dict.get(data_type.value)
                if ddl:
                    # 创建 sequence（如果DDL中有）
                    seq_match = re.search(r"nextval\('(\w+)'\)\)?", ddl)
                    if seq_match:
                        seq_name = seq_match.group(1)
                        try:
                            conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq_name}")
                        except Exception:
                            pass
                    # 执行DDL创建表
                    conn.execute(ddl)
        finally:
            conn.close()

    def _build_filter_sql(self, data_type: DataTypes, kwargs: dict) -> str:
        """构建 WHERE 条件SQL片段"""
        # 验证日期参数格式
        _validate_date(kwargs.get("start"), "start")
        _validate_date(kwargs.get("end"), "end")
        table_fields = get_table_fields(data_type)
        sql = "WHERE 1=1"

        if "start" in kwargs and kwargs["start"] and "date" in table_fields:
            sql += f" AND date >= '{kwargs['start']}'"

        if "end" in kwargs and kwargs["end"] and "date" in table_fields:
            sql += f" AND date <= '{kwargs['end']}'"

        if "symbol" in kwargs and kwargs["symbol"] and "symbol" in table_fields:
            symbol_value = kwargs["symbol"]
            # 验证 symbol 格式（长度 <= 12）
            _validate_symbol(symbol_value)
            # 处理逗号分隔的 symbol 字符串
            if isinstance(symbol_value, str) and "," in symbol_value:
                symbols = [s.strip() for s in symbol_value.split(",")]
                symbol_list = ", ".join([f"'{s}'" for s in symbols])
                sql += f" AND symbol IN ({symbol_list})"
            else:
                # 单个 symbol，直接使用
                sql += f" AND symbol = '{symbol_value}'"
        return sql

    def get_data(self, data_type: DataTypes, **kwargs):
        self._init_table(data_type)

        table_name = data_type.value
        where_sql = self._build_filter_sql(data_type, kwargs)
        sql = f"SELECT * FROM {table_name} {where_sql}"
        conn = duckdb.connect(self.cache_db_path)
        data = conn.sql(sql).to_df()
        conn.close()
        return data

    def get_data_total(self, data_type: DataTypes, **kwargs):
        self._init_table(data_type)

        table_name = data_type.value
        where_sql = self._build_filter_sql(data_type, kwargs)
        sql = f"SELECT count(*) FROM {table_name} {where_sql}"

        conn = duckdb.connect(self.cache_db_path)
        count = conn.execute(sql).fetchone()[0]
        conn.close()
        return count

    def bulk_import(self, data_type: DataTypes, data: pd.DataFrame):
        """批量导入数据到缓存（支持 upsert）

        Args:
            data_type: 表名
            data: pandas DataFrame
        """
        if data.empty:
            return
        table_name = data_type.value
        data = data.replace("NaN", None)

        # 获取 unique_keys
        unique_keys = get_table_unique_keys(data_type)
        if unique_keys:
            data = data.drop_duplicates(subset=unique_keys, keep="first")

        conn = duckdb.connect(self.cache_db_path)
        try:
            # 注册临时 DataFrame
            conn.register("temp_df", data)

            # 获取表字段
            table_fields = get_table_fields(data_type)

            if unique_keys:
                # Upsert 模式：使用 MERGE 语句
                self._bulk_upsert_df(conn, table_name, unique_keys, table_fields, data)
                rprint(f"[green]Successfully upserted {len(data)} records into {table_name}")
            else:
                self._bulk_insert_df(conn, table_name, table_fields, data)
                rprint(f"[green]Successfully inserted {len(data)} records into {table_name}")

        except Exception as e:
            rprint(f"[bold red]Error importing data: {e}")
            raise
        finally:
            conn.unregister("temp_df")
            conn.close()

    def _bulk_insert_df(
        self,
        conn,
        table_name: str,
        table_fields: list,
        data: pd.DataFrame,
    ):
        """简单批量插入（无冲突处理）"""
        # 只包含存在于 data 中的列
        data_columns = list(data.columns)
        insert_columns = [col for col in table_fields if col in data_columns]

        if not insert_columns:
            return

        column_list = ", ".join([f'"{col}"' for col in insert_columns])
        copy_sql = f"""
            INSERT INTO {table_name} ({column_list})
            SELECT {column_list} FROM temp_df
        """
        conn.execute(copy_sql)

    def _bulk_upsert_df(
        self,
        conn,
        table_name: str,
        unique_keys: list,
        table_fields: list,
        data: pd.DataFrame,
    ):
        """Upsert 数据到 DuckDB（基于唯一键的冲突解决）"""
        # 只包含存在于 data 中的列
        data_columns = list(data.columns)
        insert_columns = [col for col in table_fields if col in data_columns]

        if not insert_columns:
            return

        column_list = ", ".join([f'"{col}"' for col in insert_columns])

        # 构建更新列（非唯一键列）
        update_columns = [col for col in insert_columns if col not in unique_keys]

        if update_columns:
            # 使用 EXCLUDED 关键字引用新数据
            update_set_clause = ", ".join(
                [f'EXCLUDED."{col}"' for col in update_columns]
            )
            update_column_names = ", ".join([f'"{col}"' for col in update_columns])

            merge_sql = f"""
                INSERT INTO {table_name} ({column_list})
                SELECT {column_list} FROM temp_df
                ON CONFLICT ({', '.join([f'"{key}"' for key in unique_keys])})
                DO UPDATE SET ({update_column_names}) = ({update_set_clause})
            """
        else:
            # 如果没有非键列需要更新，直接忽略冲突
            merge_sql = f"""
                INSERT INTO {table_name} ({column_list})
                SELECT {column_list} FROM temp_df
                ON CONFLICT ({', '.join([f'"{key}"' for key in unique_keys])})
                DO NOTHING
            """

        try:
            conn.execute(merge_sql)
        except Exception as e:
            rprint(f"[bold red]Error executing merge statement: {e}")
            # 回退到 INSERT ... WHERE NOT EXISTS 方式
            self._fallback_upsert(conn, table_name, unique_keys, insert_columns)

    def _fallback_upsert(
        self,
        conn,
        table_name: str,
        unique_keys: list,
        insert_columns: list,
    ):
        """回退的 upsert 实现（使用 WHERE NOT EXISTS）"""
        column_list = ", ".join([f'"{col}"' for col in insert_columns])

        # 构建 WHERE NOT EXISTS 条件
        where_conditions = []
        for key in unique_keys:
            where_conditions.append(f't1."{key}" = t2."{key}"')
        where_clause = " AND ".join(where_conditions)

        insert_sql = f"""
            INSERT INTO {table_name} ({column_list})
            SELECT {column_list} FROM temp_df t1
            WHERE NOT EXISTS (
                SELECT 1 FROM {table_name} t2
                WHERE {where_clause}
            )
        """
        conn.execute(insert_sql)


def _validate_date(date_str: str, param_name: str) -> None:
    """验证日期参数格式，无效则抛出 ValueError"""
    # 日期格式验证正则: YYYY-MM-DD
    _DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if not date_str:
        return
    if not _DATE_PATTERN.match(date_str):
        raise ValueError(
            f"Invalid date format for '{param_name}': '{date_str}'. Expected format: YYYY-MM-DD (e.g., '2025-01-01')"
        )


def _validate_symbol(symbol_value: str) -> None:
    """验证 symbol 参数格式，无效则抛出 ValueError"""
    if not symbol_value:
        return

    # 如果是逗号分隔的字符串，验证每个 symbol
    if isinstance(symbol_value, str) and "," in symbol_value:
        symbols = [s.strip() for s in symbol_value.split(",")]
        for sym in symbols:
            if len(sym) > 12:
                raise ValueError(
                    f"Invalid symbol length: '{sym}' (length={len(sym)}). Symbol length must be <= 12"
                )
    else:
        # 单个 symbol
        if len(symbol_value) > 12:
            raise ValueError(
                f"Invalid symbol length: '{symbol_value}' (length={len(symbol_value)}). Symbol length must be <= 12"
            )
