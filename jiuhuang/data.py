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

load_dotenv()

__all__ = ["JiuhuangData"]

_CACHE_TABLES = {"stock_zh_a_hist_d": ["symbol", "date"]}


class JiuhuangData:
    def __init__(
        self, api_key: str = getenv("JIUHUANG_API_KEY") , api_url: str = getenv("JIUHUANG_API_URL"), sync: bool = True
    ):
        self.api_key = api_key
        self.api_url = api_url
        self._prepare_client(api_key)
        self._cache = _DataCache(jd=self)
        if sync:
            # _reconcile(self._cache, self)
            _DataReconciler(self._cache, self).reconcile_all(_CACHE_TABLES)

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
        response.raise_for_status()
        dts = response.json()["data"]
        return dts

    def _get_data_all(
        self,
        data_type: str,
        symbol: str = "",
        start_date: str = "",
        end_date: str = "",
    ):
        url = f"{self.api_url}/data-offline/"
        payload = {
            "data_type": data_type,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "batch_size": 1000,
            "stream": False,
        }

        response = self._client.post(url, json=payload)
        response.raise_for_status()
        dts = response.json()["data"]
        return dts

    def get_data_total(
        self,
        data_type: str,
        symbol: str = "",
        start_date: str = "",
        end_date: str = "",
    ):
        payload = {
            "data_type": data_type,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
        }

        response = self._client.post(f"{self.api_url}/data-offline/total", json=payload)

        response.raise_for_status()
        total = response.json()["data"]
        try:
            total = int(total)
        except Exception as e:
            raise e

        return total

    def get_data(
        self,
        data_type: str,
        symbol: str = "",
        start_date: str = "",
        end_date: str = "",
        remote: bool = getenv("JIUHUANG_DATA_MODE", "").lower() == "remote",
        **kwargs,
    ):
        """
        Retrieve offline data from the JiuHuang API.

        This method sends a POST request to the data-offline endpoint to fetch
        historical data based on the specified parameters.

        :param data_type: Type of data to retrieve (e.g., 'index_global_hist_em')
        :param symbol: Symbol or identifier for the data series
        :param start_date: Start date for the data range (format: YYYY-MM-DD)
        :param end_date: End date for the data range (format: YYYY-MM-DD)
        :param remote: Force fetch data form remote
        """
        data = None
        if not remote:
            kw = {k: v for k, v in kwargs.items() if k != "remote"}
            data = self._cache.get_data(
                data_type, symbol=symbol, start_date=start_date, end_date=end_date, **kw
            )

        if not data.empty:
            return data

        url = f"{self.api_url}/data-offline/"
        payload = {
            "data_type": data_type,
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "batch_size": 1000,
            "stream": True,
        }
        all_data = []
        with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
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
        self._initialize_cache()

    def _initialize_cache(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        try:
            conn = duckdb.connect(self.cache_db_path)
        except:
            rprint("[red]Warning: 请确保只有单个正在运行的JiuhuangData实例[/red]")
            sys.exit(1)
        for t in _CACHE_TABLES.keys():
            resp = self._jd._client.get(
                self._jd.api_url + "/data-offline/ddl", params={"data_type": t}
            )
            # breakpoint()
            if resp.status_code == 200:
                ddl = resp.json()["data"]
                conn.execute(ddl)
        conn.close()

    def get_data(self, data_type, **kwargs):
        table_name = data_type
        if table_name not in _CACHE_TABLES:
            return pd.DataFrame()

        sql = f"SELECT * FROM {table_name} WHERE 1=1"

        if "start_date" in kwargs and kwargs["start_date"]:
            sql += f" AND date >= '{kwargs['start_date']}'"

        if "end_date" in kwargs and kwargs["end_date"]:
            sql += f" AND date <= '{kwargs['end_date']}'"

        if "symbol" in kwargs and kwargs["symbol"]:
            sql += f" AND symbol = '{kwargs['symbol']}'"

        conn = duckdb.connect(self.cache_db_path)
        data = conn.sql(sql).to_df()
        conn.close()
        return data

    def get_data_total(self, data_type: str, **kwargs):
        conn = duckdb.connect(self.cache_db_path)
        table_name = data_type
        sql = f"SELECT count(*) FROM {table_name} WHERE 1=1"

        if "start_date" in kwargs and kwargs["start_date"]:
            sql += f" AND date >= '{kwargs['start_date']}'"

        if "end_date" in kwargs and kwargs["end_date"]:
            sql += f" AND date <= '{kwargs['end_date']}'"

        if "symbol" in kwargs and kwargs["symbol"]:
            sql += f" AND symbol = '{kwargs['symbol']}'"

        count = conn.execute(sql).fetchone()[0]
        conn.close()
        return count

    def upsert_data(self, data_type: str, data: pd.DataFrame, unique_keys: list[str]):
        conn = duckdb.connect(self.cache_db_path)
        table_name = data_type
        conn.register("temp_df", data)
        columns = list(data.columns)
        column_list = ", ".join([f'"{col}"' for col in columns])

        # Build WHERE NOT EXISTS clause to avoid conflicts
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
        conn.unregister("temp_df")
        conn.close()


class _DataReconciler:
    def __init__(self, local_data_getter, remote_data_getter):
        self.local_getter = local_data_getter
        self.remote_getter = remote_data_getter
        self.max_lookback_days = 365 * 5
        self.max_window = 90

    def reconcile_all(self, cache_tables):
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            expand=True,
        ) as progress:

            for data_type in cache_tables.keys():
                self._reconcile_table(data_type, progress, cache_tables.get(data_type))

    def _reconcile_table(self, data_type, progress, unique_keys):
        today = datetime.today()
        limit_date = today - timedelta(days=self.max_lookback_days)
        start_limit_str = limit_date.strftime("%Y-%m-%d")
        end_today_str = today.strftime("%Y-%m-%d")

        # --- 优化策略 1: 全局总量预检 ---
        # progress.console.print(f"[bold blue]🔍 Pre-checking {data_type}...[/bold blue]")

        g_local = self.local_getter.get_data_total(
            data_type, start_date=start_limit_str, end_date=end_today_str
        )
        g_remote = self.remote_getter.get_data_total(
            data_type, start_date=start_limit_str, end_date=end_today_str
        )

        task_id = progress.add_task(
            f"Reconciling {data_type}", total=self.max_lookback_days
        )

        if g_local == g_remote:
            # 总量一致，直接完成任务
            progress.update(
                task_id,
                completed=self.max_lookback_days,
                description=f"[green]✔ {data_type} Already Consistent (Total: {g_local})",
            )
            return

        # --- 如果总量不一致，进入滑动窗口逻辑 ---
        progress.console.print(
            f"[yellow]‼ {data_type} mismatch detected ({g_local} vs {g_remote}). Starting deep sync...[/yellow]"
        )

        current_end_dt = today
        window_days = 1

        while current_end_dt >= limit_date:
            current_start_dt = current_end_dt - timedelta(days=window_days - 1)
            if current_start_dt < limit_date:
                current_start_dt = limit_date

            start_str = current_start_dt.strftime("%Y-%m-%d")
            end_str = current_end_dt.strftime("%Y-%m-%d")

            l_total = self.local_getter.get_data_total(
                data_type, start_date=start_str, end_date=end_str
            )
            r_total = self.remote_getter.get_data_total(
                data_type, start_date=start_str, end_date=end_str
            )

            days_processed = (current_end_dt - current_start_dt).days + 1

            if l_total != r_total:
                # 只有不一致时才输出，且增加具体差异信息
                progress.console.print(
                    f"  [red]diff[/red] {start_str} to {end_str}: L{l_total}/R{r_total}"
                )
                remote_data = self.remote_getter.get_data(
                    data_type, start_date=start_str, end_date=end_str, remote=True
                )

                if remote_data is not None and not remote_data.empty:
                    self.local_getter.upsert_data(
                        data_type, remote_data, unique_keys=unique_keys
                    )

                current_end_dt = current_start_dt - timedelta(days=1)
                window_days = 1  # 遇到错误收缩窗口
            else:
                current_end_dt = current_start_dt - timedelta(days=1)
                window_days = min(window_days * 2, self.max_window)

            progress.update(task_id, advance=days_processed)

        progress.update(task_id, description=f"[green]✔ {data_type} Fixed and Synced")
