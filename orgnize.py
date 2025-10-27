# organize_from_exports.py
# 作用：从 export_all.py 产出的 tar 包解压并“按会话”归档：
# - 会话键：采集号(三位字母/数字) + 动作 + 日期（忽略 Left/Right）
# - 代表时间：该会话内最早的 HHMMSS，用于会话目录末尾
# - 日志识别：log_YYYYMMDD_HHMM_<CID>_<ACTION>_(FINAL|TEMP).txt
# - 期望：每会话 7 个文件（含 1 个 log_*_FINAL/TEMP）
# - 完成后导出：session_summary_YYYYMMDD.csv、session_anomalies_YYYYMMDD.txt

import datetime
import pathlib
import re
import tarfile
import tempfile
import shutil
import sys
from collections import defaultdict
import csv

# ====== 可改配置 ======
IMPORT_DIR   = pathlib.Path(r"D:\Data\Watch_Data_original")    # tar 所在目录
OUTPUT_ROOT  = pathlib.Path(r"D:\Data\Watch_Data_sessions")    # 输出根目录
DATE         = None                                            # 如 "20251026"，None=交互选择/或传参
ALLOW_EXTS   = {".csv", ".jsonl", ".json", ".log", ".txt"}     # 参与归档的文件类型
MAX_ASSIGN_DELTA_SEC = 600                                     # 兜底“就近合并”的最大时间差（秒）
EXPECTED_FILES_PER_SESSION = 7                                 # 每会话应有的文件数
# =====================

# ---------- 日期解析与交互 ----------
DATE_PAT_ANY = re.compile(r"(\d{8})")

def _today():
    return datetime.datetime.now().strftime("%Y%m%d")

def _yesterday():
    return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")

def _list_available_dates(import_dir: pathlib.Path):
    if not import_dir.exists():
        return []
    dates = set()
    for tarf in import_dir.glob("*.tar"):
        m = DATE_PAT_ANY.search(tarf.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates, reverse=True)

def _resolve_date(import_dir: pathlib.Path, preset: str | None) -> str:
    # 命令行参数优先
    if len(sys.argv) > 1:
        cand = sys.argv[1].strip()
        if re.fullmatch(r"\d{8}", cand):
            return cand
        print(f"[ERROR] 非法日期参数: {cand}（应为 yyyyMMdd）")
        sys.exit(1)

    if preset and re.fullmatch(r"\d{8}", preset):
        return preset

    options = _list_available_dates(import_dir)
    today = _today()
    yest  = _yesterday()
    for d in (today, yest):
        if d not in options:
            options.append(d)
    options = sorted(set(options), reverse=True)

    if options:
        print("\n可用日期：")
        for i, d in enumerate(options, 1):
            print(f"  {i:2d}. {d}")
        print("  0 . 手动输入（或直接回车=今天）")
        choice = input("请选择日期编号：").strip()
        if choice == "":
            return today
        if choice == "0":
            while True:
                raw = input("请输入日期 (yyyyMMdd)，回车=今天，y=昨天，q=退出：").strip()
                if raw == "":
                    return today
                if raw.lower() in ("y", "yesterday"):
                    return yest
                if raw.lower() in ("q", "quit"):
                    print("已取消。"); sys.exit(0)
                if re.fullmatch(r"\d{8}", raw):
                    return raw
                print("格式不对，请输入 8 位日期。")
        try:
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except Exception:
            pass
        print("[WARN ] 输入无效，使用今天。")
        return today
    else:
        while True:
            raw = input("未发现可用 tar。请输入日期 (yyyyMMdd)，回车=今天，y=昨天，q=退出：").strip()
            if raw == "":
                return today
            if raw.lower() in ("y", "yesterday"):
                return yest
            if raw.lower() in ("q", "quit"):
                print("已取消。"); sys.exit(0)
            if re.fullmatch(r"\d{8}", raw):
                return raw
            print("格式不对，请输入 8 位日期。")

# ---------- 文件名解析（严格：CID=三位字母/数字） ----------
# 手表/数据文件：<CID>_<ACTION>_[Left|Right]_YYYYMMDD_(HHMMSS)
PAT_STRICT = re.compile(
    r"^(?P<cid>[0-9A-Za-z]{3})_(?P<action>.+?)_"
    r"(?:(?P<side>Left|Right)_)?"
    r"(?P<date>\d{8})"
    r"(?:_(?P<hms>\d{6}))?$",
    re.IGNORECASE
)

# 宽松搜索：文件名任意处包含 <CID>_<ACTION>_[Left|Right]_YYYYMMDD_(HHMMSS)
PAT_RELAX_ANY = re.compile(
    r"(?P<cid>[0-9A-Za-z]{3})_(?P<action>.+?)_"
    r"(?:(?P<side>Left|Right)_)?"
    r"(?P<date>\d{8})"
    r"(?:_(?P<hms>\d{6}))?",
    re.IGNORECASE
)

# 手机完整日志：log_YYYYMMDD_HHMM_<CID>_<ACTION>_(FINAL|TEMP).txt
PAT_PHONE_TXT = re.compile(
    r"^log_(?P<date>\d{8})_(?P<hm>\d{4})_(?P<cid>[0-9A-Za-z]{3})_(?P<action>.+?)_(FINAL|TEMP)\.txt$",
    re.IGNORECASE
)

# 仅日期+时间（用于兜底）：YYYYMMDD_HHMMSS 或 YYYYMMDD_HHMM
PAT_DATE_TIME_ONLY = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6}|\d{4})")

def parse_strict(name: str):
    m = PAT_STRICT.match(name)
    if not m:
        return None, None
    key = f"{m.group('cid')}_{m.group('action')}_{m.group('date')}"
    hms = m.group('hms')
    return key, hms

def parse_relax_any(name: str):
    m = PAT_RELAX_ANY.search(name)
    if not m:
        return None, None
    key = f"{m.group('cid')}_{m.group('action')}_{m.group('date')}"
    hms = m.group('hms')
    return key, hms

def parse_phone_txt(name: str):
    m = PAT_PHONE_TXT.match(name)
    if not m:
        return None, None
    date = m.group('date')
    hm   = m.group('hm')   # 4位
    cid  = m.group('cid')
    act  = m.group('action')
    key  = f"{cid}_{act}_{date}"
    hms  = hm + "00"       # HHMM -> HHMMSS（补 00 秒）
    return key, hms

def parse_date_time_only(name: str):
    m = PAT_DATE_TIME_ONLY.search(name)
    if not m:
        return None, None
    date = m.group('date')
    hms  = m.group('hms')
    if len(hms) == 4:
        hms += "00"
    return date, hms

# ---------- 工具 ----------
def hhmmss_to_sec(hms: str) -> int:
    return int(hms[0:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])

def safe_move(src: pathlib.Path, dst: pathlib.Path):
    if not dst.exists():
        shutil.move(str(src), str(dst))
        return dst
    stem, suf = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem}_dup{i}{suf}")
        if not cand.exists():
            shutil.move(str(src), str(cand))
            return cand
        i += 1

def extract_tar(tar_path: pathlib.Path, to_dir: pathlib.Path):
    subdir = to_dir / tar_path.stem
    subdir.mkdir(parents=True, exist_ok=True)
    print(f"[WORK ] Extracting {tar_path.name} -> {subdir}")
    with tarfile.open(tar_path, "r:*") as tf:
        tf.extractall(path=subdir)
    return subdir

def collect_files(root: pathlib.Path, date_re: re.Pattern):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOW_EXTS and date_re.search(p.name):
            files.append(p)
    return files

# ---------- 汇总导出 ----------
def _iter_session_dirs(base_dir: pathlib.Path, date_str: str):
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and f"_{date_str}_" in d.name:
            yield d

def summarize_sessions(output_root: pathlib.Path, date_str: str, csv_path: pathlib.Path,
                       anomaly_path: pathlib.Path, expected_files_per_session: int):
    rows, anomalies = [], []
    for sess_dir in _iter_session_dirs(output_root, date_str):
        files = sorted([p for p in sess_dir.iterdir() if p.is_file()])
        count = len(files)
        name = sess_dir.name
        parts = name.split("_")
        collect_id = parts[0] if parts else ""
        action = "_".join(parts[1:-2]) if len(parts) >= 4 else ""
        rep_date = parts[-2] if len(parts) >= 2 else ""
        rep_time = parts[-1] if len(parts) >= 1 else ""
        rows.append({
            "session_dir": name,
            "collect_id": collect_id,
            "action": action,
            "date": rep_date,
            "time": rep_time,
            "file_count": count,
            "files": "; ".join([p.name for p in files]),
        })
        if count != expected_files_per_session:
            anomalies.append({
                "session_dir": name,
                "file_count": count,
                "files": [p.name for p in files],
            })

    print("\n===== Session Summary =====")
    for r in rows:
        mark = "" if r["file_count"] == expected_files_per_session else "  <-- 🚩异常"
        print(f"[SESSION] {r['session_dir']}  | 数量 = {r['file_count']}{mark}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["session_dir","collect_id","action","date","time","file_count","files"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[WRITE] 汇总已保存：{csv_path}")

    if anomalies:
        with open(anomaly_path, "w", encoding="utf-8") as f:
            for a in anomalies:
                f.write(f"[ANOM] {a['session_dir']} | count={a['file_count']}\n")
                for fn in a["files"]:
                    f.write(f"    - {fn}\n")
        print(f"[WRITE] 异常会话清单：{anomaly_path}")
    else:
        print("✅ 没有发现异常会话。")

# ---------- 主流程 ----------
def main():
    selected_date = _resolve_date(IMPORT_DIR, DATE)
    print(f"[INFO ] Organizing exports for DATE={selected_date}")
    date_re = re.compile(rf"{selected_date}")

    if not IMPORT_DIR.exists():
        print(f"[ERROR] IMPORT_DIR 不存在：{IMPORT_DIR}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    day_root = OUTPUT_ROOT / selected_date
    day_root.mkdir(parents=True, exist_ok=True)

    # 1) 找到该日期的 tar
    tars = sorted(IMPORT_DIR.glob(f"{selected_date}_*.tar"))
    if not tars:
        print(f"[WARN ] 未在 {IMPORT_DIR} 找到 {selected_date}_*.tar")
        # 没 tar 也允许继续（比如你手动放了目录），这里只是提示
    else:
        print("[INFO ] TAR files:")
        for t in tars:
            print("  -", t.name)

    total_moved = 0
    sessions_count = 0

    with tempfile.TemporaryDirectory(prefix=f"org_{selected_date}_") as tmpdir:
        tmp_root = pathlib.Path(tmpdir)

        # 2) 解压
        extracted_dirs = []
        for t in tars:
            try:
                extracted_dirs.append(extract_tar(t, tmp_root))
            except Exception as e:
                print(f"[ERR  ] 解压失败 {t.name}: {e}")

        # 如果没有 tar，也支持直接在 IMPORT_DIR/selected_date 子目录里抓
        if not extracted_dirs:
            if (IMPORT_DIR / selected_date).exists():
                extracted_dirs.append(IMPORT_DIR / selected_date)
            else:
                # 最后兜底：整个 IMPORT_DIR
                extracted_dirs.append(IMPORT_DIR)

        # 3) 聚合：严格 -> 手机txt -> 宽松
        groups = defaultdict(lambda: {"files": [], "times": []})
        residuals = []

        for d in extracted_dirs:
            for f in collect_files(d, date_re):
                name = f.name

                key, hms = parse_strict(name)
                if not key and name.lower().startswith("log_") and name.lower().endswith(".txt"):
                    key, hms = parse_phone_txt(name)
                if not key:
                    key, hms = parse_relax_any(name)

                if key:
                    groups[key]["files"].append(f)
                    if hms:
                        groups[key]["times"].append(hms)
                else:
                    residuals.append(f)

        # 4) 计算代表时间（min）
        session_rep_time_sec = {}
        for key, info in groups.items():
            rep_hms = min(info["times"]) if info["times"] else None
            session_rep_time_sec[key] = hhmmss_to_sec(rep_hms) if rep_hms else None

        # 5) 兜底（仅日期+时间时，按代表时间“就近合并”）
        misc_dir = day_root / f"misc_{selected_date}"
        for f in residuals:
            date_only, hms = parse_date_time_only(f.name)
            if date_only == selected_date and hms:
                hms_sec = hhmmss_to_sec(hms)
                cand_key, cand_dist = None, None
                for k, sec in session_rep_time_sec.items():
                    if sec is None:
                        continue
                    dist = abs(sec - hms_sec)
                    if (cand_dist is None) or (dist < cand_dist):
                        cand_dist = dist
                        cand_key = k
                if cand_key and (cand_dist is not None) and (cand_dist <= MAX_ASSIGN_DELTA_SEC):
                    groups[cand_key]["files"].append(f)
                    continue
            # 实在无法归属，放 misc
            misc_dir.mkdir(parents=True, exist_ok=True)
            target = misc_dir / f.name
            print(f"[MISC ] {f}  ->  {target}")
            safe_move(f, target)
            total_moved += 1

        # 6) 输出会话目录并移动
        for key in sorted(groups.keys()):
            times = groups[key]["times"]
            rep = min(times) if times else None
            folder_name = f"{key}_{rep}" if rep else key  # <CID>_<ACTION>_<DATE>_<HHMMSS>
            sess_dir = day_root / folder_name
            if not sess_dir.exists():
                sess_dir.mkdir(parents=True, exist_ok=True)
                sessions_count += 1

            print(f"\n[SESSION] {folder_name}  ->  {sess_dir}  (文件数 {len(groups[key]['files'])})")
            for f in groups[key]["files"]:
                target = sess_dir / f.name
                print(f"  [MOVE] {f}  ->  {target}")
                safe_move(f, target)
                total_moved += 1

    print(f"\n[DONE ] 日期 {selected_date}：会话 {sessions_count} 个，移动文件 {total_moved} 个。")
    print(f"[PATH ] 输出目录：{day_root}")
    print(f"[NOTE ] 源 tar 仍在：{IMPORT_DIR}（未改动）")

    # 7) 统计导出（当天）
    _csv  = day_root / f"session_summary_{selected_date}.csv"
    _anom = day_root / f"session_anomalies_{selected_date}.txt"
    summarize_sessions(day_root, selected_date, _csv, _anom, expected_files_per_session=EXPECTED_FILES_PER_SESSION)

if __name__ == "__main__":
    main()
