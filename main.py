# -*- coding: utf-8 -*-
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ===================== 配置 =====================
ROOT_DIR = r"D:/ALL"                     # <- 改成你的数据根目录
OUTPUT_DIR = r"./visualization_all_Character"  # 输出目录
N = 300                                   # 归一化采样点个数（统计曲线分辨率）

# ===================== 工具函数 =====================
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一为小写、去空白"""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _time_from_df(df: pd.DataFrame) -> np.ndarray:
    """
    选用时间列并转为“相对秒”（从0开始）：
      corrected_wall_clock_ms(ms) -> wall_clock_ms(ms) -> event_timestamp_ns(ns) -> 行号
    """
    ldf = _lower_cols(df)
    if "corrected_wall_clock_ms" in ldf:
        t = ldf["corrected_wall_clock_ms"].to_numpy("float64")
        return (t - t[0]) / 1e3
    if "wall_clock_ms" in ldf:
        t = ldf["wall_clock_ms"].to_numpy("float64")
        return (t - t[0]) / 1e3
    if "event_timestamp_ns" in ldf:
        t = ldf["event_timestamp_ns"].to_numpy("float64")
        return (t - t[0]) / 1e9
    # 兜底：用行号
    return np.arange(len(ldf), dtype="float64")

def find_csv_in_folder(folder: Path) -> Path | None:
    """在单个子文件夹里找到需要的 CSV；优先与文件夹同名，其次任取第一个"""
    csvs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ".csv"]
    if not csvs:
        return None
    for f in csvs:
        if f.stem == folder.name:
            return f
    return csvs[0]

def parse_action_from_folder(folder_name: str) -> str | None:
    """从文件夹名解析动作：'..._draw3_...' -> 'draw3'"""
    m = re.search(r"draw\s*(_)?\s*(\d+)", folder_name, re.IGNORECASE)
    if m:
        return f"draw{int(m.group(2))}"
    return None

def read_single_combined_csv(path: Path):
    """
    读取一个包含 accel_x/y/z 和 gyro_x/y/z 的 CSV。
    返回：accel_time, accel_x, accel_y, accel_z, gyro_time, gyro_x, gyro_y, gyro_z（全部 numpy 数组）
    """
    df = pd.read_csv(path)
    # 列名做成小写，避免大小写不一致
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = {"accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} 缺少必要列：{sorted(missing)}")

    # 🔑 时间戳选择顺序（毫秒优先）
    for c, unit in [
        ("corrected_wall_clock_ms", "ms"),
        ("wall_clock_ms", "ms"),
        ("event_timestamp_ns", "ns"),
        ("timestamp", "auto"),
        ("time", "auto")
    ]:
        if c in df.columns:
            t = df[c].to_numpy(dtype=float)
            if unit == "ns":
                t = (t - t[0]) / 1e9   # 纳秒 → 秒
            elif unit == "ms":
                t = (t - t[0]) / 1e3   # 毫秒 → 秒
            else:
                # 粗略判断
                span = abs(t.max() - t.min())
                if span >= 1e14:
                    t = (t - t[0]) / 1e9
                elif span >= 1e11:
                    t = (t - t[0]) / 1e3
                else:
                    t = np.arange(len(df), dtype=float)
            accel_time = gyro_time = t
            break
    else:
        # 没有任何时间列，就用行号
        t = np.arange(len(df), dtype=float)
        accel_time = gyro_time = t

    accel_x = df["accel_x"].to_numpy(float)
    accel_y = df["accel_y"].to_numpy(float)
    accel_z = df["accel_z"].to_numpy(float)
    gyro_x  = df["gyro_x"].to_numpy(float)
    gyro_y  = df["gyro_y"].to_numpy(float)
    gyro_z  = df["gyro_z"].to_numpy(float)
    return accel_time, accel_x, accel_y, accel_z, gyro_time, gyro_x, gyro_y, gyro_z


# ===================== 画图样式 =====================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
})

# ===================== 统计可视化 =====================
def visualize_sensor_data_by_category(category_data: dict, save_root: Path, n_points: int = N):
    """
    每个动作 -> 一张图；背景淡线 + 均值曲线 + ±1σ 填充带
    category_data: {action_name: [(seq_name, data_dict), ...]}
    data_dict keys: accel_time, accel_x/y/z, gyro_time, gyro_x/y/z (pandas.Series)
    """
    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    bg_alpha = 0.03
    mean_lw = 2.2
    fill_alpha = 0.15

    for act, seq_list in category_data.items():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle(f"Action: {act}", fontsize=18)
        for ax in axes:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        acc_stack = {a: [] for a in "xyz"}
        gyr_stack = {a: [] for a in "xyz"}

        # ---------- 背景线 + 堆栈收集（统一到 0~1 归一化时间） ----------
        for _, d in seq_list:
            # 相对秒
            t_acc = d["accel_time"].to_numpy() - d["accel_time"].iloc[0]
            t_gyr = d["gyro_time"].to_numpy() - d["gyro_time"].iloc[0]

            # 防止跨度为 0
            Tacc = float(t_acc[-1]) if len(t_acc) > 1 else 1.0
            Tgyr = float(t_gyr[-1]) if len(t_gyr) > 1 else 1.0

            # 归一化到 0~1
            t_acc_norm = t_acc / Tacc
            t_gyr_norm = t_gyr / Tgyr

            # 统一的归一化采样点
            t_new = np.linspace(0, 1, n_points)

            for axis in "xyz":
                # 背景线（归一化时间）
                axes[0].plot(t_acc_norm, d[f"accel_{axis}"], color=colors[axis], alpha=bg_alpha, lw=1)
                axes[1].plot(t_gyr_norm, d[f"gyro_{axis}"],  color=colors[axis], alpha=bg_alpha, lw=1)

                # 对齐到公共时间轴
                acc_stack[axis].append(np.interp(t_new, t_acc_norm, d[f"accel_{axis}"].to_numpy()))
                gyr_stack[axis].append(np.interp(t_new, t_gyr_norm, d[f"gyro_{axis}"].to_numpy()))

        # ---------- 均值 ± 1σ ----------
        t_norm = np.linspace(0, 1, n_points)
        for axis in "xyz":
            acc_arr = np.vstack(acc_stack[axis]) if acc_stack[axis] else np.empty((0, n_points))
            gyr_arr = np.vstack(gyr_stack[axis]) if gyr_stack[axis] else np.empty((0, n_points))
            if acc_arr.size == 0 or gyr_arr.size == 0:
                continue

            # 平滑均值（窗口 11、三次多项式；需 n_points>=11）
            acc_mean = savgol_filter(acc_arr.mean(0), 11, 3) if n_points >= 11 else acc_arr.mean(0)
            gyr_mean = savgol_filter(gyr_arr.mean(0), 11, 3) if n_points >= 11 else gyr_arr.mean(0)

            acc_std = acc_arr.std(0)
            gyr_std = gyr_arr.std(0)

            axes[0].fill_between(t_norm, acc_mean - acc_std, acc_mean + acc_std, color=colors[axis], alpha=fill_alpha)
            axes[1].fill_between(t_norm, gyr_mean - gyr_std, gyr_mean + gyr_std, color=colors[axis], alpha=fill_alpha)

            axes[0].plot(t_norm, acc_mean, color=colors[axis], lw=mean_lw, label=f"{axis.upper()}-axis")
            axes[1].plot(t_norm, gyr_mean, color=colors[axis], lw=mean_lw)

        axes[0].set_title("Accelerometer", fontsize=18)
        axes[1].set_title("Gyroscope", fontsize=18)
        for ax in axes:
            ax.set_xlabel("Normalized Time", fontsize=18)
        axes[0].set_ylabel("Accel (m/s²)", fontsize=18)
        axes[1].set_ylabel("Angular (rad/s)", fontsize=18)
        axes[0].legend(frameon=False, loc="upper right")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = Path(OUTPUT_DIR) / f"{act}.png"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✅ Saved  {out_path}")

# ===================== 主流程（扁平目录） =====================
def visualize_all_flat(base_dir: str):
    base_path = Path(base_dir)
    out_root = Path(OUTPUT_DIR)

    if out_root.exists():
        shutil.rmtree(out_root)
        print("🗑️ 已清空旧的可视化目录")
    out_root.mkdir(parents=True, exist_ok=True)

    all_category_data: dict[str, list] = {}  # {action: [(seq_name, data_dict), ...]}

    # 遍历一层子文件夹
    for folder in sorted([d for d in base_path.iterdir() if d.is_dir()]):
        csv_path = find_csv_in_folder(folder)
        if csv_path is None:
            continue
        try:
            (accel_time, accel_x, accel_y, accel_z,
             gyro_time,  gyro_x,  gyro_y,  gyro_z) = read_single_combined_csv(csv_path)
        except Exception as e:
            print(f"⚠️ 读取失败：{csv_path} -> {e}")
            continue

        action_name = parse_action_from_folder(folder.name) or "unknown"
        seq_name = folder.name

        all_category_data.setdefault(action_name, []).append((
            seq_name,
            {
                "accel_time": pd.Series(accel_time),
                "accel_x": pd.Series(accel_x),
                "accel_y": pd.Series(accel_y),
                "accel_z": pd.Series(accel_z),
                "gyro_time": pd.Series(gyro_time),
                "gyro_x": pd.Series(gyro_x),
                "gyro_y": pd.Series(gyro_y),
                "gyro_z": pd.Series(gyro_z),
            }
        ))

    if all_category_data:
        visualize_sensor_data_by_category(all_category_data, out_root, n_points=N)
        print(f"🎉 全部图像已保存至：{out_root.resolve()}")
    else:
        print("⚠️ 未找到任何可用的 CSV 数据")

# ===================== 入口 =====================
def main():
    visualize_all_flat(ROOT_DIR)

if __name__ == "__main__":
    main()
