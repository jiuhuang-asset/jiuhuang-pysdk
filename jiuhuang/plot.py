import pandas as pd
import mplfinance as mpf
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def plot(
    data: pd.DataFrame,
    strategy_name: str,
    symbol: str,
    save_path: str = "",
    bt_result: pd.DataFrame = None,
    **kwargs,
):
    plot_data = data[data.strategy == strategy_name]
    plot_data = plot_data[plot_data.symbol == symbol]
    if plot_data.empty:
        raise ValueError("No data to plot")

    plot_data.index = pd.to_datetime(plot_data.date)

    add_plots = []

    # Add buy signals as upward triangles
    if "buy_signal" in plot_data.columns:
        buy_mask = plot_data["buy_signal"] == 1
        if buy_mask.any():
            buy_series = pd.Series(
                data=float("nan"), index=plot_data.index, dtype=float
            )
            buy_series.loc[buy_mask] = plot_data.loc[buy_mask, "close"].values
            add_plots.append(
                mpf.make_addplot(
                    buy_series,
                    type="scatter",
                    marker="^",
                    markersize=100,
                    color="green",
                    secondary_y=False,
                )
            )

    # Add sell signals as downward triangles
    if "sell_signal" in plot_data.columns:
        sell_mask = plot_data["sell_signal"] == 1
        if sell_mask.any():
            sell_series = pd.Series(
                data=float("nan"), index=plot_data.index, dtype=float
            )
            sell_series.loc[sell_mask] = plot_data.loc[sell_mask, "close"].values
            add_plots.append(
                mpf.make_addplot(
                    sell_series,
                    type="scatter",
                    marker="v",
                    markersize=100,
                    color="red",
                    secondary_y=False,
                )
            )

    # Add position markers
    if "position" in plot_data.columns:
        position_mask = plot_data["position"] == 1
        if position_mask.any():
            pos_series = pd.Series(
                data=float("nan"), index=plot_data.index, dtype=float
            )
            pos_series.loc[position_mask] = plot_data.loc[position_mask, "close"].values
            add_plots.append(
                mpf.make_addplot(
                    pos_series,
                    type="scatter",
                    marker="+",
                    markersize=100,
                    color="blue",
                    secondary_y=False,
                )
            )

    # Pass the additional plots to mpf.plot()
    if save_path:
        savefig_dict = {
            "fname": save_path,
            "bbox_inches": "tight",
            "pad_inches": 0.05,  # Very small padding
            "dpi": kwargs.get("dpi", 300),  # You can adjust DPI as needed
        }

        fig, axes = mpf.plot(
            plot_data,
            addplot=add_plots,
            returnfig=True,  # 返回figure对象以便进一步定制
            figratio=(4, 3),  # Adjust aspect ratio
            figscale=1.0,  # Control overall scale
            **kwargs,
        )

        # 在图的右上角添加注释
        if bt_result is not None:
            annotation = bt_result.loc[bt_result.index == symbol, strategy_name].values[
                0
            ]
            # 获取主轴
            main_ax = axes[0] if isinstance(axes, (list, tuple)) else axes
            # 在右上角添加文本
            main_ax.text(
                0.99,
                0.99,
                str(annotation),
                transform=main_ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12,
            )  # 设置较小的字体

        # 保存图像
        fig.savefig(**savefig_dict)
        plt.close(fig)  # 关闭图形以释放内存
    else:
        fig, axes = mpf.plot(plot_data, addplot=add_plots, returnfig=True, **kwargs)

        # 在图的右上角添加注释
        if bt_result is not None:
            annotation = bt_result.loc[bt_result.index == symbol, strategy_name].values[
                0
            ]
            # 获取主轴
            main_ax = axes[0] if isinstance(axes, (list, tuple)) else axes
            # 在右上角添加文本
            main_ax.text(
                0.99,
                0.99,
                str(annotation),
                transform=main_ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                fontsize=12,
            )  # 设置较小的字体

        plt.show()


def facet_plot(
    data: pd.DataFrame,
    strategy_names: list[str],
    symbol: str,
    cols: int,
    bt_result: pd.DataFrame = None,
    **kwargs,
):

    with tempfile.TemporaryDirectory() as temp_dir:
        plots = []
        for strategy_name in strategy_names:
            plot_path = f"{temp_dir}/{strategy_name}.png"
            plot(
                data,
                strategy_name,
                symbol,
                save_path=f"{plot_path}",
                bt_result=bt_result,
                **kwargs,
            )
            plots.append(plot_path)

        # Calculate rows needed based on number of plots and columns
        n_plots = len(plots)
        rows = (n_plots + cols - 1) // cols  # Ceiling division

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))

        # Handle case where there's only one subplot
        if n_plots == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if cols > 1 else [axes]
        else:
            axes = axes.flatten()

        # Read and display each image in the subplot
        for i, plot_path in enumerate(plots):
            img = Image.open(plot_path)
            axes[i].imshow(np.asarray(img))
            axes[i].axis("off")  # Turn off axis
            # Extract title from the filename or strategy name
            strategy_name = strategy_names[i]
            axes[i].set_title(strategy_name)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()
