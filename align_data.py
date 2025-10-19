# align_all_for_date.py
# 批量：按日期遍历 SESSIONS_ROOT/<DATE> 下所有会话目录，完成左右手对齐并把结果写回各会话目录。
# 用法：
#   交互：python align_all_for_date.py  ← 按提示输入日期
#   传参：python align_all_for_date.py 20251019
# 可调：
#   - SESSIONS_ROOT：会话根目录（你之前 organize 的输出根）
#   - TOLERANCE_MS：跨手表 merge_asof 容差（默认 10ms）
# 产物：
#   - 每个会话目录：aligned_<ID>_<动作>_tol{TOL}ms.csv + aligned_<ID>_<动作>_summary.txt
#   - 日期目录：aligned_summary_<DATE>.csv 汇总表

import pandas as pd
import re
import sys
from pathlib import Path
from typing import Tuple, Optional
import datetime
import traceback

# ====== 只需改这里 ======
SESSIONS_ROOT = Path(r"D:\Data\Watch_Data_sessions")  # 组织后的会话根目录
TOLERANCE_MS  = 10  # 跨手表 merge_asof 容差
# =======================

LOG_TXT_PATTERN = re.compile(
    r"^log_(?P<date>\d{8})_(?P<hm>\d{4})_(?P<id>\d{3})_(?P<action>.+?)_(FINAL|TEMP)\.txt$",
    re.IGNORECASE
)
CSV_NAME_PATTERN = re.compile(
    r"^(?P<id>\d{3})_(?P<action>.+?)_(?P<side>Left|Right)_(?P<date>\d{8})_(?P<hms>\d{6})_(?P<type>acc|gyr)\.csv$",
    re.IGNORECASE
)

def resolve_date(argv) -> str:
    def _today(): return datetime.datetime.now().strftime("%Y%m%d")
    if len(argv) > 1:
        cand = argv[1].strip()
        if not re.fullmatch(r"\d{8}", cand):
            print(f"[ERROR] 日期格式应为 yyyyMMdd：{cand}")
            sys.exit(1)
        return cand
    raw = input("请输入要处理的日期 (yyyyMMdd)，回车=今天：").strip()
    if raw == "": return _today()
    if not re.fullmatch(r"\d{8}", raw):
        print("[ERROR] 日期格式不正确。"); sys.exit(1)
    return raw

def find_unique_file(session: Path, patterns) -> Path:
    if isinstance(patterns, str):
        patterns = [patterns]
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(session.glob(pat)))
    if not candidates:
        raise FileNotFoundError(f"在 {session} 未找到：{patterns}")
    finals = [p for p in candidates if p.name.upper().endswith("_FINAL.TXT")]
    if finals:
        return finals[0]
    return candidates[-1]

def autodetect_files(session: Path):
    log_txt = find_unique_file(session, "log_*.txt")
    mlog = LOG_TXT_PATTERN.match(log_txt.name)
    if not mlog:
        raise ValueError(f"日志命名不符合规范：{log_txt.name}")
    cid = mlog.group("id")
    action = mlog.group("action")

    left_acc  = find_unique_file(session, "*_Left_*_acc.csv")
    left_gyr  = find_unique_file(session, "*_Left_*_gyr.csv")
    right_acc = find_unique_file(session, "*_Right_*_acc.csv")
    right_gyr = find_unique_file(session, "*_Right_*_gyr.csv")

    # 一致性提示（不强制）
    for p in [left_acc, left_gyr, right_acc, right_gyr]:
        m = CSV_NAME_PATTERN.match(p.name)
        if not m:
            print(f"[WARN] CSV 命名非标准：{p.name}")
            continue
        if m.group("id") != cid or m.group("action") != action:
            print(f"[WARN] {p.name} 的采集号/动作与日志不完全一致（继续处理）")

    return log_txt, left_acc, left_gyr, right_acc, right_gyr, cid, action

def parse_log_file(log_path: Path):
    sync = {'mac_to_hand': {}, 'hand_to_mac': {}, 'started': {}, 'stopped': {}, 'user_click_ts': None}
    mac_pat = r'\[([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})\]'
    hand_re = re.compile(rf'{mac_pat} Set session ctx: .*?, hand=(Left|Right)')
    start_re = re.compile(rf'{mac_pat} 收到消息: \'COLLECTION_STARTED:(\d+)\'')
    stop_re  = re.compile(rf'{mac_pat} 收到消息: \'COLLECTION_STOPPED:(\d+)\'')
    click_re = re.compile(r'CMD_SEND \| .*? 发送指令: START, 发送时间戳: (\d+)')

    content = log_path.read_text(encoding='utf-8', errors='ignore')

    m = click_re.search(content)
    if m:
        sync['user_click_ts'] = int(m.group(1))
    else:
        raise ValueError("日志中未找到 START 指令时间戳。")

    for hm in hand_re.finditer(content):
        mac, hand = hm.groups()
        hand = hand.lower()
        sync['mac_to_hand'][mac] = hand
        sync['hand_to_mac'][hand] = mac

    for sm in start_re.finditer(content):
        mac, ts = sm.groups()
        hand = sync['mac_to_hand'].get(mac)
        if hand: sync['started'][hand] = int(ts)

    for sm in stop_re.finditer(content):
        mac, ts = sm.groups()
        hand = sync['mac_to_hand'].get(mac)
        if hand: sync['stopped'][hand] = int(ts)

    return sync

def load_and_prepare_hand(acc_path: Path, gyr_path: Path, suffix: str) -> pd.DataFrame:
    df_acc = pd.read_csv(acc_path)
    df_gyr = pd.read_csv(gyr_path)
    df_acc.columns = df_acc.columns.str.strip()
    df_gyr.columns = df_gyr.columns.str.strip()

    df_acc[f'frame_index_acc_{suffix}'] = df_acc.index + 2
    df_gyr[f'frame_index_gyr_{suffix}'] = df_gyr.index + 2

    df_acc.rename(columns={c: f'acc_{c}_{suffix}' for c in ['x','y','z']}, inplace=True)
    df_gyr.rename(columns={c: f'gyr_{c}_{suffix}' for c in ['x','y','z']}, inplace=True)

    df_hand = pd.merge_asof(
        df_acc.sort_values('wall_ms'),
        df_gyr.sort_values('wall_ms'),
        on='wall_ms', direction='nearest', tolerance=5
    )
    return df_hand.sort_values('wall_ms').reset_index(drop=True)

def process_one_session(session_dir: Path, date: str) -> dict:
    """返回统计信息字典；如失败，抛异常到上层捕获。"""
    log_txt, left_acc, left_gyr, right_acc, right_gyr, cid, action = autodetect_files(session_dir)
    sync = parse_log_file(log_txt)

    left_start  = sync['started'].get('left',  -10**18)
    right_start = sync['started'].get('right', -10**18)
    left_stop   = sync['stopped'].get('left',   10**18)
    right_stop  = sync['stopped'].get('right',  10**18)

    master_start = max(left_start, right_start)
    master_end   = min(left_stop, right_stop)

    df_left  = load_and_prepare_hand(left_acc,  left_gyr,  'left')
    df_right = load_and_prepare_hand(right_acc, right_gyr, 'right')

    if master_start <= master_end:
        df_left  = df_left[(df_left['wall_ms'] >= master_start) & (df_left['wall_ms'] <= master_end)]
        df_right = df_right[(df_right['wall_ms'] >= master_start) & (df_right['wall_ms'] <= master_end)]

    aligned = pd.merge_asof(
        df_left.sort_values('wall_ms'),
        df_right.sort_values('wall_ms'),
        on='wall_ms', direction='nearest', tolerance=TOLERANCE_MS
    ).dropna()

    aligned.rename(columns={'wall_ms':'aligned_timestamp_ms'}, inplace=True)
    aligned['relative_user_click_ms'] = aligned['aligned_timestamp_ms'] - sync['user_click_ts']

    final_cols_order = [
        'aligned_timestamp_ms', 'relative_user_click_ms',
        'frame_index_acc_left','frame_index_gyr_left',
        'frame_index_acc_right','frame_index_gyr_right',
        'acc_x_left','acc_y_left','acc_z_left',
        'gyr_x_left','gyr_y_left','gyr_z_left',
        'acc_x_right','acc_y_right','acc_z_right',
        'gyr_x_right','gyr_y_right','gyr_z_right',
    ]
    final_cols = [c for c in final_cols_order if c in aligned.columns]
    aligned = aligned[final_cols]

    safe_action = re.sub(r'[\\/:*?"<>|]', '_', action)
    out_csv = session_dir / f"aligned_{cid}_{safe_action}_tol{TOLERANCE_MS}ms.csv"
    aligned.to_csv(out_csv, index=False, float_format='%.6f', encoding='utf-8-sig')

    report = (
        f"date: {date}\n"
        f"session_dir: {session_dir.name}\n"
        f"id: {cid}\naction: {action}\n"
        f"tolerance_ms: {TOLERANCE_MS}\n"
        f"user_click_ts: {sync['user_click_ts']}\n"
        f"master_start: {master_start}\nmaster_end: {master_end}\n"
        f"rows_left_after_cut: {len(df_left)}\nrows_right_after_cut: {len(df_right)}\n"
        f"rows_aligned: {len(aligned)}\n"
    )
    (session_dir / f"aligned_{cid}_{safe_action}_summary.txt").write_text(report, encoding='utf-8-sig')

    return {
        "session": session_dir.name,
        "id": cid,
        "action": action,
        "rows_left": len(df_left),
        "rows_right": len(df_right),
        "rows_aligned": len(aligned),
        "output_csv": str(out_csv)
    }

def main():
    date = resolve_date(sys.argv)
    day_dir = SESSIONS_ROOT / date
    if not day_dir.exists():
        print(f"[ERROR] 目录不存在：{day_dir}")
        sys.exit(1)

    # 会话文件夹：名称里含日期即可
    session_dirs = sorted([p for p in day_dir.iterdir() if p.is_dir() and date in p.name])
    if not session_dirs:
        print(f"[WARN] {day_dir} 下未发现会话目录。")
        sys.exit(0)

    print(f"[INFO] 共发现 {len(session_dirs)} 个会话，开始处理……")

    rows = []
    fail_log = []

    for i, sess in enumerate(session_dirs, 1):
        print(f"\n[{i}/{len(session_dirs)}] 处理：{sess.name}")
        try:
            info = process_one_session(sess, date)
            rows.append(info)
            print(f"    ✓ 完成，对齐行数：{info['rows_aligned']}")
        except Exception as e:
            print(f"    ✗ 失败：{e}")
            fail_log.append({"session": sess.name, "error": str(e)})
            # 写入会话目录 error.log 便于排查
            (sess / "align_error.log").write_text(
                f"{e}\n\n{traceback.format_exc()}", encoding='utf-8'
            )

    # 汇总
    if rows:
        df_sum = pd.DataFrame(rows)
        sum_csv = day_dir / f"aligned_summary_{date}.csv"
        df_sum.to_csv(sum_csv, index=False, encoding='utf-8-sig')
        print(f"\n[SUMMARY] 已写入汇总：{sum_csv}")
    else:
        print("\n[SUMMARY] 没有成功对齐的会话。")

    if fail_log:
        print("\n[FAILED] 有会话处理失败：")
        for r in fail_log:
            print(f"  - {r['session']}: {r['error']}")

    print("\n[ALL DONE]")

if __name__ == '__main__':
    main()
