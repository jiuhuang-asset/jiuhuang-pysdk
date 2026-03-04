import re
import httpx
import json
import tempfile
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
    "stock_zcfz_em": 365 * 5,
    "stock_lrb_em": 365 * 5,
    "stock_xjll_em": 365 * 5,  
}

class JiuhuangData:
    def __init__(
        self,
        api_key: str = getenv("JIUHUANG_API_KEY"),
        api_url: str = getenv("JIUHUANG_API_URL"),
        sync: bool = True,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._prepare_client(api_key)
        self._cache = _DataCache(jd=self)
        if sync:
            _DataReconciler(self._cache, self).reconcile_all()

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
        print("DEBUG", data_type)
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
            data = self._cache.get_data(
                data_type, **kw
            )

            if not data.empty:
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

        return pd.DataFrame(all_data)


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
                    seq_match = re.search(r"nextval\('(\w+)'\)\)?" , ddl)
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
        table_fields = self._table_fields[table_name]
        sql = "WHERE 1=1"

        if "start" in kwargs and kwargs["start"] and "date" in table_fields:
            sql += f" AND date >= '{kwargs['start']}'"

        if "end" in kwargs and kwargs["end"] and "date" in table_fields:
            sql += f" AND date <= '{kwargs['end']}'"

        if "symbol" in kwargs and kwargs["symbol"] and "symbol" in table_fields:
            sql += f" AND symbol = '{kwargs['symbol']}'"

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
        """批量导入数据到缓存

        Args:
            data_type: 表名
            data: pandas DataFrame
        """
        if data.empty:
            return

        table_name = data_type
        data = data.replace("NaN", None)
        conn = duckdb.connect(self.cache_db_path)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            temp_path = f.name

        try:
            # 写入 Parquet 文件
            data.to_parquet(temp_path, index=False)

            # 使用 COPY 高效导入
            copy_sql = f"COPY {table_name} FROM '{temp_path}' (FORMAT PARQUET)"
            conn.execute(copy_sql)
        finally:
            # 清理临时文件
            import os
            os.unlink(temp_path)

        conn.close()


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
        """
        全量同步逻辑 (当 lookback_days 为 None 时):
        - 对比本地和远程数据总量
        - 如果一致则跳过，否则获取远程全量数据同步
        """
        task_id = progress.add_task(f"Full Sync {data_type}", total=1)
        local_total = self.local_getter.get_data_total(data_type)
        remote_total = self.remote_getter.get_data_total(data_type)

        if local_total == remote_total:
            progress.update(
                task_id,
                completed=1,
                description=f"[green]✔ {data_type} Already Consistent ({local_total} rows)",
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

        task_id = progress.add_task(
            f"Reconciling {data_type}", total=lookback_days
        )

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
                description=f"[green]✔ {data_type} Already Consistent",
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
                # 先删除本地 window 期间的数据，再下载 remote 数据导入
                conn = duckdb.connect(self.local_getter.cache_db_path)
                conn.execute(
                    f"DELETE FROM {data_type} WHERE date >= '{start_str}' AND date <= '{end_str}'"
                )
                conn.close()

                # 同步这个窗口的数据
                remote_data = self.remote_getter.get_data(
                    data_type, start=start_str, end=end_str, remote=True
                )

                if remote_data is not None and not remote_data.empty:
                    self.local_getter.bulk_import(data_type, remote_data)

            # 窗口向前滑动（闭区间不重叠：下一个窗口从 start-1 开始）
            current_end_dt = current_start_dt - timedelta(days=1)

        progress.update(task_id, description=f"[green]✔ {data_type} Fixed and Synced")
