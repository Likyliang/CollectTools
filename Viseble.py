# viz_actions.py
# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d  # 用于把不同长度的序列插值到同一长度

# =============== 1) 扫描目录结构 ===============
def traverse_directory(root_dir: str):
    """
    返回结构:
    {
      collection_id: {
        action_category: [action_name1, action_name2, ...],
        ...
      },
      ...
    }
    """
    structure = {}
    for collection_id in os.listdir(root_dir):
        collection_path = os.path.join(root_dir, collection_id)
        if not os.path.isdir(collection_path):
            continue
        structure[collection_id] = {}
        for action_category in os.listdir(collection_path):
            category_path = os.path.join(collection_path, action_category)
            if not os.path.isdir(category_path):
                continue
            structure[collection_id][action_category] = []
            for action_name in os.listdir(category_path):
                action_path = os.path.join(category_path, action_name)
                if not os.path.isdir(action_path):
                    continue
                structure[collection_id][action_category].append(action_name)
    return structure


# =============== 2) 选择采集号 ===============
def select_collection_id(structure: dict):
    all_collection_ids = sorted(list(structure.keys()))
    print("所有采集号：")
    for idx, collection_id in enumerate(all_collection_ids, start=1):
        print(f"{idx}: {collection_id}")
    choice = input("请输入采集号序号（输入 0 绘制所有采集号汇总）：").strip()
    if choice == "0":
        return 0
    try:
        return all_collection_ids[int(choice) - 1]
    except Exception:
        print("输入无效，默认绘制所有采集号汇总。")
        return 0


# =============== 3) 读取“单 CSV（含 accel/gyro 六列）” ===============
def read_sensor_data(csv_path: Path):
    """
    读取一个 sequence 下的单 CSV，返回字典：
    {
      'time': 序列行号（当作时间）,
      'accel_x/y/z', 'gyro_x/y/z': 每列的 pandas.Series
    }
    """
    df = pd.read_csv(csv_path)
    # 必要列检查
    required = {"accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"}
    cols_lower = {c.strip().lower() for c in df.columns}
    missing = required - cols_lower
    if missing:
        raise ValueError(f"CSV 缺少必要列 {sorted(missing)}: {csv_path}")

    # 用行号当时间戳，保持简单稳妥
    df = df.reset_index().rename(columns={"index": "time"})

    return {
        "time": df["time"],
        "accel_x": df["accel_x"], "accel_y": df["accel_y"], "accel_z": df["accel_z"],
        "gyro_x":  df["gyro_x"],  "gyro_y":  df["gyro_y"],  "gyro_z":  df["gyro_z"],
    }


# =============== 4) 每个采集号的详细图（每条序列一行 × 4列） ===============
def visualize_sensor_data_by_category(category_data: dict, save_root: Path):
    """
    category_data 结构：
    {
      category_name: {
        seq_name: {
           'time', 'accel_x/y/z', 'gyro_x/y/z'
        }, ...
      }, ...
    }
    为每个“动作名”生成一张大图：每条序列一行，共 4 列子图
    """
    for category_name, sequences in category_data.items():
        category_dir = save_root / category_name
        category_dir.mkdir(parents=True, exist_ok=True)

        # 按动作名分组（seq_name 的左侧部分）
        action_groups = {}
        for seq_name, data in sequences.items():
            action_name = seq_name.split("-")[0]
            action_groups.setdefault(action_name, []).append((seq_name, data))

        for action_name, action_data in action_groups.items():
            num_rows = len(action_data)
            num_cols = 4  # Accel三轴 / Gyro三轴 / Accel模长 / Gyro模长
            plt.figure(figsize=(6 * num_cols, 2.5 * num_rows))
            plt.suptitle(f"Category: {category_name} | Action: {action_name}", fontsize=14)

            for i, (seq_name, data) in enumerate(action_data):
                # Accel三轴
                ax = plt.subplot(num_rows, num_cols, i * num_cols + 1)
                ax.plot(data['time'], data['accel_x'], label='X')
                ax.plot(data['time'], data['accel_y'], label='Y')
                ax.plot(data['time'], data['accel_z'], label='Z')
                ax.set_title(f"{seq_name} - Accel")
                ax.set_xlabel("Time (index)")
                ax.set_ylabel("m/s²")
                ax.legend(fontsize=8)

                # Gyro三轴
                ax = plt.subplot(num_rows, num_cols, i * num_cols + 2)
                ax.plot(data['time'], data['gyro_x'], label='X')
                ax.plot(data['time'], data['gyro_y'], label='Y')
                ax.plot(data['time'], data['gyro_z'], label='Z')
                ax.set_title(f"{seq_name} - Gyro")
                ax.set_xlabel("Time (index)")
                ax.set_ylabel("rad/s")
                ax.legend(fontsize=8)

                # Accel 模长
                accel_norm = np.sqrt(np.array(data['accel_x'])**2 +
                                     np.array(data['accel_y'])**2 +
                                     np.array(data['accel_z'])**2)
                ax = plt.subplot(num_rows, num_cols, i * num_cols + 3)
                ax.plot(data['time'], accel_norm, color='tab:blue')
                ax.set_title(f"{seq_name} - Accel Norm")
                ax.set_xlabel("Time (index)")
                ax.set_ylabel("Magnitude")

                # Gyro 模长
                gyro_norm = np.sqrt(np.array(data['gyro_x'])**2 +
                                    np.array(data['gyro_y'])**2 +
                                    np.array(data['gyro_z'])**2)
                ax = plt.subplot(num_rows, num_cols, i * num_cols + 4)
                ax.plot(data['time'], gyro_norm, color='tab:orange')
                ax.set_title(f"{seq_name} - Gyro Norm")
                ax.set_xlabel("Time (index)")
                ax.set_ylabel("Magnitude")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path = category_dir / f"{category_name}_{action_name}.png"
            plt.savefig(out_path, dpi=120)
            plt.close()
            print(f"已保存：{out_path}")


# =============== 5) 画指定采集号的所有动作序列 ===============
def visualize_collection_data(root_dir: str, structure: dict, collection_id: str):
    """
    针对一个采集号，收集所有序列（单CSV），并调用上面的函数出图。
    目录假设：root/collection_id/category/action_name/device/sequence/sequence.csv
    """
    save_root = Path(collection_id)
    if save_root.exists():
        replace = input(f"采集号文件夹 '{collection_id}' 已存在，是否清空？(y/n): ").lower()
        if replace == 'y':
            shutil.rmtree(save_root)
            print(f"已清空文件夹: {collection_id}")
        else:
            return
    save_root.mkdir(parents=True, exist_ok=True)

    category_data = {}

    for collection_i, categories in structure.items():
        if collection_i != collection_id:
            continue
        for action_category, actions in categories.items():
            category_data.setdefault(action_category, {})
            for action_name in actions:
                # root/collection/action_category/action_name/*
                action_path = Path(root_dir) / collection_id / action_category / action_name
                if not action_path.exists():
                    continue
                for device_folder in action_path.iterdir():
                    if not device_folder.is_dir():
                        continue
                    for sequence_folder in device_folder.iterdir():
                        if not sequence_folder.is_dir():
                            continue

                        # 找单CSV（忽略 .txt/.log）
                        csv_files = [f for f in sequence_folder.glob("*.csv")]
                        if len(csv_files) == 0:
                            continue
                        # 如有多个，优先选“与序列同名”的，否则取第一个
                        chosen = None
                        for f in csv_files:
                            if f.stem == sequence_folder.name:
                                chosen = f; break
                        if chosen is None:
                            chosen = csv_files[0]

                        try:
                            data = read_sensor_data(chosen)
                        except Exception as e:
                            print(f"[跳过] 读取失败 {chosen}: {e}")
                            continue

                        seq_name = f"{action_name}-{sequence_folder.name}"
                        category_data[action_category][seq_name] = data

    if any(category_data.values()):
        visualize_sensor_data_by_category(category_data, save_root)
        print(f"可视化结果已保存至：{save_root.absolute()}")
    else:
        print(f"警告：未找到采集号 {collection_id} 的数据。")


# =============== 6) 汇总所有采集号：动作级均值±标准差曲线（accel_x / gyro_x） ===============
def visualize_all_actions(root_dir: str, structure: dict):
    """
    对每个动作 action_name，收集所有序列的 accel_x / gyro_x，
    统一到同一长度后画均值±标准差，便于快速比较动作形态。
    """
    save_root = Path("All_Actions")
    if save_root.exists():
        replace = input(f"'All_Actions' 文件夹已存在，是否清空？(y/n): ").lower()
        if replace == 'y':
            shutil.rmtree(save_root)
            print("已清空文件夹: All_Actions")
        else:
            return
    save_root.mkdir(parents=True, exist_ok=True)

    # 聚合所有动作的所有序列
    action_data = {}  # {action_name: {'accel_x': [(t, v), ...], 'gyro_x': [(t, v), ...]}}

    for collection_id, categories in structure.items():
        for category, actions in categories.items():
            for action_name in actions:
                action_path = Path(root_dir) / collection_id / category / action_name
                if not action_path.exists():
                    continue
                for device_folder in action_path.iterdir():
                    if not device_folder.is_dir():
                        continue
                    for sequence_folder in device_folder.iterdir():
                        if not sequence_folder.is_dir():
                            continue
                        csv_files = [f for f in sequence_folder.glob("*.csv")]
                        if not csv_files:
                            continue
                        # 选择一个 CSV
                        chosen = None
                        for f in csv_files:
                            if f.stem == sequence_folder.name:
                                chosen = f; break
                        if chosen is None:
                            chosen = csv_files[0]

                        try:
                            d = read_sensor_data(chosen)
                        except Exception:
                            continue

                        # 时间归零（从 0 开始），仅用于插值对齐
                        t = np.arange(len(d['time']), dtype=float)
                        action_data.setdefault(action_name, {'accel_x': [], 'gyro_x': []})
                        action_data[action_name]['accel_x'].append((t, np.asarray(d['accel_x'], float)))
                        action_data[action_name]['gyro_x'].append((t, np.asarray(d['gyro_x'],  float)))

    # 出图：对每个动作画两行（accel_x / gyro_x）
    for action_name, sensor in action_data.items():
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        plt.suptitle(f"Action: {action_name} (mean ± std)", fontsize=15)

        def plot_mean_std(ax, data_list, title, ylabel):
            # 统一长度：取所有序列中最短长度 minL，按该长度生成公共时间轴
            if not data_list:
                ax.set_title(title + " (no data)"); return
            minL = min(len(v) for _, v in data_list)
            if minL < 5:  # 太短就不画
                ax.set_title(title + " (too short)"); return
            common_t = np.linspace(0, minL - 1, minL)
            aligned = []
            for t, v in data_list:
                # 按各自长度 -> 公共时间轴线性插值
                if len(v) < 2:
                    continue
                f = interp1d(np.linspace(0, minL - 1, len(v)), v[:len(v)], kind='linear', fill_value='extrapolate')
                aligned.append(f(common_t))
            if not aligned:
                ax.set_title(title + " (no aligned data)"); return
            arr = np.vstack(aligned)  # [N, minL]
            mean = arr.mean(axis=0)
            std  = arr.std(axis=0)
            ax.plot(common_t, mean)
            ax.fill_between(common_t, mean - std, mean + std, alpha=0.25)
            ax.set_title(title); ax.set_xlabel("Normalized Time"); ax.set_ylabel(ylabel)

        plot_mean_std(axes[0], sensor['accel_x'], "Accelerometer X (mean ± std)", "m/s²")
        plot_mean_std(axes[1], sensor['gyro_x'],  "Gyroscope X (mean ± std)",     "rad/s")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = save_root / f"{action_name}_mean_std.png"
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"已保存汇总动作图：{out_path}")


# =============== 7) 主程序入口 ===============
def main():
    # ↓↓↓ 把这里改成你的数据根目录 ↓↓↓
    root_dir = r"D:/ALL"

    structure = traverse_directory(root_dir)
    print("目录结构遍历完成！")

    collection_id = select_collection_id(structure)
    print(f"选择的采集号：{collection_id}")

    if collection_id == 0:
        visualize_all_actions(root_dir, structure)
    else:
        visualize_collection_data(root_dir, structure, str(collection_id))


if __name__ == "__main__":
    main()
