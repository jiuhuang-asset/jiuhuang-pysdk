import re
import httpx
import json
import sys
import pandas as pd
from os import getenv
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os
import duckdb
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich import print as rprint
from .utils import raise_err_with_details


load_dotenv()

__all__ = ["JiuhuangData"]


_CACHE_LOOKBACK_DAYS = {
    "stock_info_a_code_name": None,
    "stock_zh_a_hist_d": 365 * 5,
    "stock_zh_a_hist_d_qfq": 365 * 5,
    "stock_zh_a_hist_d_hfq": 365 * 5,
    "stock_zcfz_em": None,
    "stock_lrb_em": None,
    "stock_xjll_em": None,
}

_CACHE_TABLE_UNIQUE_KEYS = {
    "stock_info_a_code_name": ["code"],
    "stock_zh_a_hist_d": ["date", "symbol"],
    "stock_zh_a_hist_d_qfq": ["date", "symbol"],
    "stock_zh_a_hist_d_hfq": ["date", "symbol"],
    "stock_zcfz_em": ["date", "symbol"],
    "stock_lrb_em": ["date", "symbol"],
    "stock_xjll_em": ["date", "symbol"],
}


class JiuhuangData:
    def __init__(
        self,
        api_key: str = getenv("JIUHUANG_API_KEY"),
        api_url: str = getenv("JIUHUANG_API_URL"),
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._prepare_client(api_key)
        self._cache = _DataCache(jd=self)

    def _prepare_client(self, api_key: str):
        client = httpx.Client(timeout=180)
        client.headers.update({"Authorization": f"Bearer {api_key}"})
        self._client = client

    def get_data_types(self):
        """
        Retrieve available offline data types from the JiuHuang API.

        This method sends a GET request to the data-offline/data_types endpoint
        to fetch a list of available data types that can be retrieved.

        :return: JSON response containing available data types
        :rtype: dict
        :raises httpx.HTTPError: If the HTTP request fails
        """
        url = f"{self.api_url}/data-offline/data_types"
        response = self._client.get(url)
        raise_err_with_details(response)
        dts = response.json()["data"]
        return dts

    def get_data_total(
        self,
        data_type: str,
        **kwargs,
    ):
        payload = {
            "data_type": data_type,
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
        data_type: str,
        remote: bool = False,
        **kwargs,
    ):
        """
        Retrieve offline data from the JiuHuang API.

        This method sends a POST request to the data-offline endpoint to fetch
        historical data based on the specified parameters.

        :param data_type: Type of data to retrieve (e.g., 'index_global_hist_em')
        :param remote: Force fetch data form remote
        """
        data = None
        if not remote:
            kw = {k: v for k, v in kwargs.items() if k != "remote"}
            data = self._cache.get_data(data_type, **kw)
            remote_data_count = self.get_data_total(data_type=data_type, **kwargs)
            if not data.empty and (len(data) == remote_data_count):
                return data

        url = f"{self.api_url}/data-offline/"
        payload = {
            "data_type": data_type,
        }
        payload.update(kwargs)
        all_data = []
        with self._client.stream("POST", url, json=payload) as response:
            raise_err_with_details(response, read_body=True)
            for chunk in response.iter_lines():
                if chunk:
                    try:
                        resp = json.loads(chunk)
                        data = resp["data"]
                        all_data.extend(data)
                    except json.JSONDecodeError as e:
                        raise Exception("获取数据失败[Json解码错误]")
        data = pd.DataFrame(all_data)
        self._cache.bulk_import(data_type, data)
        return data


class _DataCache:
    def __init__(self, jd: JiuhuangData):
        self.cache_dir = os.path.expanduser("~/.jiuhuang")
        self.cache_db_path = os.path.join(self.cache_dir, "cache_data.db")
        self._jd = jd
        self._table_fields = {}
        self._initialize_cache()

    def _initialize_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)

        try:
            conn = duckdb.connect(self.cache_db_path)
        except:
            rprint("[red]Warning: 请确保只有单个正在运行的JiuhuangData实例[/red]")
            sys.exit(1)

        data_types = list(_CACHE_LOOKBACK_DAYS.keys())

        # 批量获取 DDL
        resp = self._jd._client.get(
            self._jd.api_url + "/data-offline/ddl", params={"data_types": data_types}
        )
        if resp.status_code == 200:
            ddl_dict = resp.json()["data"]
            for ddl in ddl_dict.values():
                if ddl:
                    seq_match = re.search(r"nextval\('(\w+)'\)\)?", ddl)
                    if seq_match:
                        seq_name = seq_match.group(1)
                        try:
                            conn.execute(f"CREATE SEQUENCE IF NOT EXISTS {seq_name}")
                        except Exception:
                            pass
                    conn.execute(ddl)

        # 批量获取 table_fields
        resp = self._jd._client.get(
            self._jd.api_url + "/data-offline/table_fields",
            params={"data_types": data_types},
        )
        if resp.status_code == 200:
            fields_dict = resp.json()["data"]
            self._table_fields.update(fields_dict)

        conn.close()

    def _build_filter_sql(self, table_name: str, kwargs: dict) -> str:
        """构建 WHERE 条件SQL片段"""
        # 验证日期参数格式
        _validate_date(kwargs.get("start"), "start")
        _validate_date(kwargs.get("end"), "end")

        table_fields = self._table_fields[table_name]
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

    def get_data(self, data_type, **kwargs):
        table_name = data_type
        if table_name not in _CACHE_LOOKBACK_DAYS:
            return pd.DataFrame()

        where_sql = self._build_filter_sql(table_name, kwargs)
        sql = f"SELECT * FROM {table_name} {where_sql}"
        conn = duckdb.connect(self.cache_db_path)
        data = conn.sql(sql).to_df()
        conn.close()
        return data

    def get_data_total(self, data_type: str, **kwargs):
        table_name = data_type
        where_sql = self._build_filter_sql(table_name, kwargs)
        sql = f"SELECT count(*) FROM {table_name} {where_sql}"

        conn = duckdb.connect(self.cache_db_path)
        count = conn.execute(sql).fetchone()[0]
        conn.close()
        return count

    def bulk_import(self, data_type: str, data: pd.DataFrame):
        """批量导入数据到缓存（支持 upsert）

        Args:
            data_type: 表名
            data: pandas DataFrame
        """
        if data_type not in _CACHE_LOOKBACK_DAYS or data.empty:
            return
        table_name = data_type
        data = data.replace("NaN", None)

        # 获取 unique_keys
        unique_keys = _CACHE_TABLE_UNIQUE_KEYS.get(data_type)

        conn = duckdb.connect(self.cache_db_path)

        try:
            # 注册临时 DataFrame
            conn.register("temp_df", data)

            # 获取表字段
            table_fields = self._table_fields.get(table_name, list(data.columns))

            if unique_keys:
                # Upsert 模式：使用 MERGE 语句
                self._bulk_upsert_df(conn, table_name, unique_keys, table_fields, data)
                print(f"Successfully upserted {len(data)} records into {table_name}")
            else:
                # 插入模式：使用 COPY
                self._bulk_insert_df(conn, table_name, table_fields, data)
                print(f"Successfully inserted {len(data)} records into {table_name}")

        except Exception as e:
            print(f"Error importing data: {e}")
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
            print(f"Error executing merge statement: {e}")
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


class _DataReconciler:
    def __init__(self, local_data_getter, remote_data_getter):
        self.local_getter = local_data_getter
        self.remote_getter = remote_data_getter
        self.window_size = 30

    def reconcile_all(self):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:
            for data_type in _CACHE_LOOKBACK_DAYS.keys():
                self._reconcile_table(data_type, progress)

    def _reconcile_table_full(self, data_type, progress):
        task_id = progress.add_task(f"Full Sync {data_type}", total=1)
        local_total = self.local_getter.get_data_total(data_type)
        remote_total = self.remote_getter.get_data_total(data_type)

        if local_total == remote_total:
            progress.update(
                task_id,
                completed=1,
                description=f"[green]✔ {data_type}",
            )
            return

        progress.console.print(
            f"[yellow]‼ {data_type} mismatch ({local_total} vs {remote_total}). Syncing...[/yellow]"
        )

        # 获取远程全量数据
        remote_data = self.remote_getter.get_data(data_type, remote=True)

        if remote_data is not None and not remote_data.empty:
            actual_remote_total = len(remote_data)
            progress.console.print(
                f"[cyan]→ Fetched {actual_remote_total} rows from remote for {data_type}[/cyan]"
            )

            # 删除本地全量数据
            conn = duckdb.connect(self.local_getter.cache_db_path)
            conn.execute(f"DELETE FROM {data_type}")
            conn.close()

            progress.update(task_id, advance=0.5)

            # 导入远程数据
            self.local_getter.bulk_import(data_type, remote_data)

            progress.update(
                task_id,
                completed=1,
                description=f"[green]✔ {data_type} Full Sync Complete",
            )
        else:
            progress.update(
                task_id,
                completed=1,
                description=f"[yellow]⚠ {data_type} No remote data found",
            )

    def _reconcile_table(self, data_type, progress):
        """
        对账逻辑:

        全量对比 → 有差异 →
            从最近 window (30天) 开始滑动检查
            ↓
            发现差异 → 删除本地 window 数据 → 同步 remote 数据
            ↓
            继续往前下一个 window (30天)
            ↓
            直到 limit_date
        """
        today = datetime.today()
        lookback_days = _CACHE_LOOKBACK_DAYS.get(data_type)

        # 如果 lookback_days 为 None，执行全量同步
        if lookback_days is None:
            self._reconcile_table_full(data_type, progress)
            return

        limit_date = today - timedelta(days=lookback_days)
        start_limit_str = limit_date.strftime("%Y-%m-%d")
        end_today_str = today.strftime("%Y-%m-%d")

        task_id = progress.add_task(f"Reconciling {data_type}", total=lookback_days)

        # --- Step 1: 全量对比整个区间 ---
        g_local = self.local_getter.get_data_total(
            data_type, start=start_limit_str, end=end_today_str
        )
        g_remote = self.remote_getter.get_data_total(
            data_type, start=start_limit_str, end=end_today_str
        )

        progress.update(task_id, advance=max(1, lookback_days * 0.05))

        if g_local == g_remote:
            progress.update(
                task_id,
                completed=lookback_days,
                description=f"[green]✔ {data_type}",
            )
            return

        progress.console.print(
            f"[yellow]‼ {data_type} mismatch ({g_local} vs {g_remote}). Starting window scan...[/yellow]"
        )

        # --- Step 2: 固定窗口滑动检测 ---
        window_days = self.window_size
        current_end_dt = today

        while current_end_dt >= limit_date:
            # 计算窗口起始日期（闭区间，所以 -1）
            current_start_dt = current_end_dt - timedelta(days=window_days - 1)

            # 如果窗口超出 limit_date，调整为 limit_date
            if current_start_dt < limit_date:
                current_start_dt = limit_date
                window_days = (current_end_dt - current_start_dt).days + 1

            start_str = current_start_dt.strftime("%Y-%m-%d")
            end_str = current_end_dt.strftime("%Y-%m-%d")

            # 对比这个窗口的数据量
            l_total = self.local_getter.get_data_total(
                data_type, start=start_str, end=end_str
            )
            r_total = self.remote_getter.get_data_total(
                data_type, start=start_str, end=end_str
            )

            days_in_window = (current_end_dt - current_start_dt).days + 1
            progress.update(task_id, advance=days_in_window)

            if l_total != r_total:

                # 同步这个窗口的数据
                remote_data = self.remote_getter.get_data(
                    data_type, start=start_str, end=end_str, remote=True
                )

                if remote_data is not None and not remote_data.empty:
                    self.local_getter.bulk_import(data_type, remote_data)

            # 窗口向前滑动（闭区间不重叠：下一个窗口从 start-1 开始）
            current_end_dt = current_start_dt - timedelta(days=1)

        progress.update(task_id, description=f"[green]✔ {data_type} Fixed and Synced")


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
