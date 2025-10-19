import pandas as pd
import re
import os

# --- 文件路径已根据您上传的文件名硬编码 ---
SESSION_LOG_FILE = 'log_20251019_1331_007_坐姿休息_FINAL.txt'
LEFT_ACC_CSV = '007_坐姿休息_Left_20251019_133148_acc.csv'
LEFT_GYR_CSV = '007_坐姿休息_Left_20251019_133148_gyr.csv'
RIGHT_ACC_CSV = '007_坐姿休息_Right_20251019_133148_acc.csv'
RIGHT_GYR_CSV = '007_坐姿休息_Right_20251019_133148_gyr.csv'


def parse_log_file(log_path):
    """解析日志文件，提取所有关键的同步信息。"""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"错误: 日志文件未找到 -> {log_path}")

    sync_info = {'mac_to_hand': {}, 'hand_to_mac': {}, 'started': {}, 'stopped': {}, 'user_click_ts': None}
    mac_pattern = r'\[([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\]'
    hand_pattern = re.compile(rf'{mac_pattern} Set session ctx: .*?, hand=(Left|Right)')
    started_pattern = re.compile(rf'{mac_pattern} 收到消息: \'COLLECTION_STARTED:(\d+)\'')
    stopped_pattern = re.compile(rf'{mac_pattern} 收到消息: \'COLLECTION_STOPPED:(\d+)\'')
    start_cmd_pattern = re.compile(r'CMD_SEND \| .*? 发送指令: START, 发送时间戳: (\d+)')

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    start_cmd_match = start_cmd_pattern.search(content)
    if start_cmd_match:
        sync_info['user_click_ts'] = int(start_cmd_match.group(1))
    else:
        raise ValueError("日志中未找到'START'指令的发送时间戳。")

    for hand_match in hand_pattern.finditer(content):
        mac, hand = hand_match.groups()
        sync_info['mac_to_hand'][mac] = hand.lower()
        sync_info['hand_to_mac'][hand.lower()] = mac

    for started_match in started_pattern.finditer(content):
        mac, ts = started_match.groups()
        hand = sync_info['mac_to_hand'].get(mac)
        if hand: sync_info['started'][hand] = int(ts)

    for stopped_match in stopped_pattern.finditer(content):
        mac, ts = stopped_match.groups()
        hand = sync_info['mac_to_hand'].get(mac)
        if hand: sync_info['stopped'][hand] = int(ts)

    return sync_info


def load_and_prepare_hand_data(acc_path, gyr_path, suffix):
    """加载、合并单只手的所有传感器数据，并添加帧号。"""
    df_acc = pd.read_csv(acc_path)
    df_gyr = pd.read_csv(gyr_path)
    df_acc.columns = df_acc.columns.str.strip()
    df_gyr.columns = df_gyr.columns.str.strip()

    df_acc[f'frame_index_acc_{suffix}'] = df_acc.index + 2
    df_gyr[f'frame_index_gyr_{suffix}'] = df_gyr.index + 2

    # 重命名XYZ列以避免合并冲突
    acc_cols = {col: f'acc_{col}_{suffix}' for col in ['x', 'y', 'z']}
    gyr_cols = {col: f'gyr_{col}_{suffix}' for col in ['x', 'y', 'z']}
    df_acc.rename(columns=acc_cols, inplace=True)
    df_gyr.rename(columns=gyr_cols, inplace=True)

    # 内部对齐ACC和GYR
    df_hand = pd.merge_asof(
        df_acc.sort_values('wall_ms'),
        df_gyr.sort_values('wall_ms'),
        on='wall_ms', direction='nearest', tolerance=5
    )
    return df_hand.sort_values('wall_ms').reset_index(drop=True)


def main():
    """主执行函数"""
    print("--- 开始处理 '007_坐姿休息' 会话数据 (原始数据匹配方案) ---")

    # 1. 解析日志
    print("1. 正在解析日志...")
    sync_info = parse_log_file(SESSION_LOG_FILE)

    # 2. 计算共同时间窗口用于裁剪
    master_start = max(sync_info['started']['left'], sync_info['started']['right'])
    master_end = min(sync_info['stopped']['left'], sync_info['stopped']['right'])
    print(f"   - 共同起点: {master_start}, 共同终点: {master_end}")

    # 3. 加载并合并每只手的数据
    print("\n2. 正在加载并合并传感器数据...")
    df_left = load_and_prepare_hand_data(LEFT_ACC_CSV, LEFT_GYR_CSV, 'left')
    df_right = load_and_prepare_hand_data(RIGHT_ACC_CSV, RIGHT_GYR_CSV, 'right')
    print(f"   - [成功] 左手表内部对齐后: {len(df_left)} 行")
    print(f"   - [成功] 右手表内部对齐后: {len(df_right)} 行")

    # 4. 裁剪到共同时间窗口
    print("\n3. 正在根据共同时间窗口裁剪数据...")
    df_left_cut = df_left[(df_left['wall_ms'] >= master_start) & (df_left['wall_ms'] <= master_end)]
    df_right_cut = df_right[(df_right['wall_ms'] >= master_start) & (df_right['wall_ms'] <= master_end)]
    print(f"   - 左手表裁剪后: {len(df_left_cut)} 行")
    print(f"   - 右手表裁剪后: {len(df_right_cut)} 行")

    # 5. 跨手表对齐与舍弃 (使用10ms容差)
    print("\n4. 正在使用 10ms 容差进行跨手表对齐...")
    aligned_df = pd.merge_asof(
        df_left_cut,
        df_right_cut,
        on='wall_ms',
        direction='nearest',
        tolerance=10  # <-- 核心修改：使用您建议的10ms容差
    )
    aligned_df.dropna(inplace=True)
    print(f"   - [成功] 对齐完成，生成了 {len(aligned_df)} 行共同数据。")

    # 6. 生成最终输出列
    print("\n5. 正在生成最终的输出列...")
    aligned_df.rename(columns={'wall_ms': 'aligned_timestamp_ms'}, inplace=True)
    aligned_df['relative_user_click_ms'] = aligned_df['aligned_timestamp_ms'] - sync_info['user_click_ts']

    # 7. 保存结果
    output_filename = "aligned_session_007_坐姿休息_raw_match.csv"
    print(f"\n6. 正在保存结果到文件: {output_filename} ...")

    # 重新组织列顺序以获得最佳可读性
    final_cols_order = [
        'aligned_timestamp_ms', 'relative_user_click_ms',
        'frame_index_acc_left', 'frame_index_gyr_left',
        'frame_index_acc_right', 'frame_index_gyr_right',
        'acc_x_left', 'acc_y_left', 'acc_z_left',
        'gyr_x_left', 'gyr_y_left', 'gyr_z_left',
        'acc_x_right', 'acc_y_right', 'acc_z_right',
        'gyr_x_right', 'gyr_y_right', 'gyr_z_right',
    ]
    final_cols = [col for col in final_cols_order if col in aligned_df.columns]

    aligned_df[final_cols].to_csv(output_filename, index=False, float_format='%.6f')

    print(f"\n--- [完成] 数据对齐成功！ ---")
    print(f"请查看新文件: {output_filename}")


if __name__ == '__main__':
    main()