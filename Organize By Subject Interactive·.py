"""
交互式：按“采集号区间（3位 base36：0-9,a-z）”把会话目录整理到“受试者”文件夹
==============================================================================
内置默认路径（可直接运行于 PyCharm）：
  源(root)：D:\Data\Watch_Data_Sessions
  目标(out)：D:\Data\Watch_Data_Subjects
也可通过命令行覆盖：--root / --out

功能要点：
- **先选日期**：列出根目录下的日期文件夹，支持 1,3-5,20251026 这样选择；回车=全部。
- **全交互**：逐个输入受试者与其采集号区间（可多段）；可选 overrides（单个采集号强制归属）。
- **三位 base36**：按 000..009..00a..00z..010.. 规则；自动补零与大小写规范。
- **dry-run**：先生成映射 CSV，确认后输入 yes 才移动；移动时保留日期层级。
- **去重安全**：若目标已存在，会自动在文件夹名后追加 _dup1/_dup2...

依赖：
  pip install pandas

运行示例：
  直接运行（用默认路径）：
    python organize_by_subject_interactive_defaults.py
  或指定路径：
    python organize_by_subject_interactive_defaults.py --root "D:/Data/Watch_Data_Sessions" --out "D:/Data/Watch_Data_Subjects"
"""
from __future__ import annotations
import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import pandas as pd

# ================== 基础：采集号解析（3位 base36） ==================

def normalize_code(code: str) -> str:
    code = code.strip().lower()
    if not 1 <= len(code) <= 3:
        raise ValueError("采集号必须是 1~3 位 base36 (0-9,a-z)")
    if not re.fullmatch(r"[0-9a-z]{1,3}", code):
        raise ValueError("采集号仅允许 0-9,a-z")
    return code.zfill(3)

def extract_collect_id(session_dir_name: str) -> Optional[str]:
    m = re.match(r"^\s*([0-9A-Za-z]{1,3})", session_dir_name)
    if not m:
        return None
    try:
        return normalize_code(m.group(1))
    except Exception:
        return None

def base36_value(code: str) -> int:
    return int(code, 36)

def in_range(code: str, start: str, end: str) -> bool:
    ci = base36_value(code)
    return base36_value(start) <= ci <= base36_value(end)

# ================== 交互：录入受试者区间与 overrides ==================

def input_subject_ranges() -> Dict[str, List[Tuple[str, str]]]:
    print("[交互] 逐个输入受试者与其采集号区间。示例：S01 的区间 00b 到 00v。")
    print("      区间可以有多段；在该受试者下，空行结束；整体回车可结束所有输入。\n")
    ranges: Dict[str, List[Tuple[str, str]]] = {}
    while True:
        subj = input("输入受试者标签（如 S01），直接回车结束所有输入：").strip()
        if subj == "":
            break
        segs: List[Tuple[str, str]] = []
        while True:
            line = input(f"[{subj}] 输入采集号区间（start end），或回车结束该受试者：").strip()
            if line == "":
                break
            parts = line.split()
            if len(parts) != 2:
                print("  ✗ 格式应为：<start> <end>，例如：00b 00v")
                continue
            try:
                st = normalize_code(parts[0])
                ed = normalize_code(parts[1])
                if base36_value(st) > base36_value(ed):
                    print("  ✱ 注意：start > end，已自动对调。")
                    st, ed = ed, st
                segs.append((st, ed))
                print(f"  ✓ 已添加区间：{st} ~ {ed}")
            except Exception as e:
                print(f"  ✗ {e}")
        if segs:
            ranges.setdefault(subj, []).extend(segs)
            print(f"[{subj}] 共 {len(segs)} 个区间。\n")
        else:
            print(f"[{subj}] 未添加任何区间，已跳过。\n")
    return ranges


def input_overrides() -> Dict[str, str]:
    print("[可选] 录入 overrides（特定采集号直接指定受试者）。")
    print("      示例：022 S03；回车结束。\n")
    overrides: Dict[str, str] = {}
    while True:
        line = input("override <code> <Subject> > ").strip()
        if line == "":
            break
        parts = line.split()
        if len(parts) != 2:
            print("  ✗ 格式应为：<code> <Subject>")
            continue
        try:
            code = normalize_code(parts[0])
            subj = parts[1].strip()
            overrides[code] = subj
            print(f"  ✓ 已添加 override：{code} -> {subj}")
        except Exception as e:
            print(f"  ✗ {e}")
    return overrides

# ================== 判定：采集号 → 受试者 ==================

def decide_subject(code: str, ranges: Dict[str, List[Tuple[str, str]]], overrides: Dict[str, str]) -> List[str]:
    if code in overrides:
        return [overrides[code]]
    matches = []
    for subj, segs in ranges.items():
        for st, ed in segs:
            if in_range(code, st, ed):
                matches.append(subj)
                break
    return matches

# ================== 文件移动（保留日期层级） ==================

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        # 若目标已存在，避免覆盖：附加后缀
        k = 1
        new_dst = dst
        while new_dst.exists():
            new_dst = dst.with_name(dst.name + f"_dup{k}")
            k += 1
        dst = new_dst
    shutil.move(str(src), str(dst))
    return dst

# ================== 新增：日期选择 ==================

def pick_dates(src_root: Path) -> List[str]:
    dates = [p.name for p in sorted(src_root.iterdir()) if p.is_dir()]
    if not dates:
        print("[ERR] 根目录下未发现日期文件夹。")
        sys.exit(1)
    print("\n=== 请选择要处理的日期 ===")
    for i, d in enumerate(dates, 1):
        print(f"  {i:2d}. {d}")
    print("输入序号/区间或直接输入日期名，逗号分隔；如：1,3-5,20251026。回车=全部。")
    s = input("选择 > ").strip()
    if not s:
        return dates
    chosen: List[str] = []
    tokens = [t.strip() for t in s.split(',') if t.strip()]
    for tok in tokens:
        if re.fullmatch(r"\d+\-\d+", tok):
            a, b = tok.split('-')
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            for i in range(a, b+1):
                if 1 <= i <= len(dates):
                    chosen.append(dates[i-1])
        elif tok.isdigit():
            i = int(tok)
            if 1 <= i <= len(dates):
                chosen.append(dates[i-1])
        else:
            if tok in dates:
                chosen.append(tok)
    # 去重并保持原顺序
    seen = set()
    picked = []
    for d in dates:
        if d in chosen and d not in seen:
            picked.append(d)
            seen.add(d)
    if not picked:
        print("未选择到有效日期，将处理全部。")
        return dates
    print("已选择：", ", ".join(picked))
    return picked

# ================== 主程序（内置默认路径 + 日期选择） ==================
DEFAULT_SRC = Path(r"D:\Data\Watch_Data_Sessions")
DEFAULT_OUT = Path(r"D:\Data\Watch_Data_Subjects")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_SRC), help="源根目录（原 Watch_Data_Sessions）")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="目标根目录（按受试者分组后）")
    args = ap.parse_args()

    src_root = Path(args.root)
    dst_root = Path(args.out)
    dst_root.mkdir(parents=True, exist_ok=True)

    # 选择需要处理的日期
    selected_dates = pick_dates(src_root)

    print("\n=== 第一步：输入受试者与区间 ===")
    ranges = input_subject_ranges()
    if not ranges:
        print("未输入任何区间，退出。")
        sys.exit(0)

    print("\n=== 第二步（可选）：输入 overrides ===")
    overrides = input_overrides()

    # 汇总展示
    print("\n=== 区间与 overrides 总览 ===")
    for subj, segs in ranges.items():
        seg_str = ", ".join([f"{st}~{ed}" for st, ed in segs])
        print(f"  {subj}: {seg_str}")
    if overrides:
        print("  overrides:")
        for c, s in overrides.items():
            print(f"    {c} -> {s}")

    # 生成计划
    plan_rows: List[Tuple[str, str, str, str, str]] = []
    for date_dir in sorted(src_root.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_dir.name not in selected_dates:
            continue
        date_name = date_dir.name
        for session_dir in sorted(date_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            sid = session_dir.name
            code = extract_collect_id(sid)
            if not code:
                plan_rows.append((sid, date_name, "", "_UNASSIGNED", "NO_CODE"))
                continue
            cand = decide_subject(code, ranges, overrides)
            if len(cand) == 1:
                plan_rows.append((sid, date_name, code, cand[0], "OK"))
            elif len(cand) == 0:
                plan_rows.append((sid, date_name, code, "_UNASSIGNED", "NO_MATCH"))
            else:
                # 多个 subject 命中：进入交互选择
                print(f"[冲突] 采集号 {code} 同时命中 {cand}，请从中选择：", end="")
                choice = input().strip()
                if choice in cand:
                    plan_rows.append((sid, date_name, code, choice, "RESOLVED_INTERACTIVE"))
                else:
                    plan_rows.append((sid, date_name, code, "_CONFLICT", "AMBIGUOUS"))

    df = pd.DataFrame(plan_rows, columns=["session_dir", "date", "collect_id", "subject", "reason"])
    map_csv = dst_root / "subject_mapping_plan.csv"
    df.to_csv(map_csv, index=False, encoding="utf-8-sig")
    print(f"\n[INFO] 计划映射表已生成: {map_csv}  (rows={len(df)})")

    # 统计摘要
    print("\n=== 计划摘要 ===")
    print(df["reason"].value_counts(dropna=False))

    # 是否执行移动
    ans = input("\n是否执行移动？输入 'yes' 确认，其他键取消：").strip().lower()
    if ans != "yes":
        print("已取消移动。")
        sys.exit(0)

    # 执行移动
    moved = 0
    for sid, date_name, code, subj, reason in plan_rows:
        src = src_root / date_name / sid
        if not src.exists():
            print(f"[SKIP] 源不存在: {src}")
            continue
        dst = dst_root / subj / date_name / sid
        try:
            final_dst = safe_move(src, dst)
            moved += 1
            print(f"[MOVE] {src} -> {final_dst}  ({reason})")
        except Exception as e:
            print(f"[ERR ] 移动失败: {src} -> {dst} : {e}")

    print(f"\n[DONE] 已移动 {moved} 个会话目录。映射表: {map_csv}")

if __name__ == "__main__":
    main()
