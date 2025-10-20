# organize_from_exports.py
# 从 export_all.py 生成的 tar 包自动解压，按“采集号+动作+日期”聚合。
# - 忽略 Left/Right；时间 HHMMSS 仅用于命名代表时间；
# - 直接在解压出的 Phone_CollectionLogs 中识别手机完整日志 txt：
#   log_YYYYMMDD_HHMM_XXX_动作名_(FINAL|TEMP).txt
# - 新增：支持交互式选择日期（扫描 IMPORT_DIR 下可用的 tar 的日期）

import datetime
import pathlib
import re
import tarfile
import tempfile
import shutil
import sys
from collections import defaultdict

# ====== 可改配置 ======
IMPORT_DIR   = pathlib.Path(r"D:\Data\Watch_Data_original")  # tar 所在目录
OUTPUT_ROOT  = pathlib.Path(r"D:\Data\Watch_Data_sessions")  # 输出根目录（会创建 <DATE>/…）
DATE         = None  # 例如 "20251019"；None=交互选择（或今天）
ALLOW_EXTS   = {".csv", ".jsonl", ".json", ".log", ".txt"}  # 收集的扩展名
# =====================

# ---------- 新增：日期解析与交互 ----------
DATE_PAT_ANY = re.compile(r"(\d{8})")

def _today():
    return datetime.datetime.now().strftime("%Y%m%d")

def _yesterday():
    return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")

def _list_available_dates(import_dir: pathlib.Path):
    """从 IMPORT_DIR 下的 *.tar 文件名中提取所有可用的 8 位日期集合。"""
    if not import_dir.exists():
        return []
    dates = set()
    for tarf in import_dir.glob("*.tar"):
        m = DATE_PAT_ANY.search(tarf.name)
        if m:
            dates.add(m.group(1))
    return sorted(dates, reverse=True)

def _resolve_date(import_dir: pathlib.Path, preset: str | None) -> str:
    """优先使用命令行参数/预设 DATE；否则扫描可选日期并提供交互菜单。"""
    # 1) 命令行参数 > 预设变量
    if len(sys.argv) > 1:
        cand = sys.argv[1].strip()
        if re.fullmatch(r"\d{8}", cand):
            return cand
        print(f"[ERROR] 非法日期参数: {cand}（应为 yyyyMMdd）")
        sys.exit(1)

    if preset and re.fullmatch(r"\d{8}", preset):
        return preset

    # 2) 扫描 IMPORT_DIR 下可用日期
    options = _list_available_dates(import_dir)
    today = _today()
    yest = _yesterday()
    # 把今天/昨天也加入可选（即使目录里暂时没有）
    for d in (today, yest):
        if d not in options:
            options.append(d)
    # 去重+排序
    options = sorted(set(options), reverse=True)

    # 3) 交互菜单
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
        # 没扫描到 tar，就走手动输入
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

# ---------- 原有解析正则 ----------
# —— 严格解析（手表 CSV / TimeSyncPairs）——
PAT_STRICT = re.compile(
    r"^(?P<prefix>\d{3}_.+?)_"
    r"(?:(?P<side>Left|Right)_)?"
    r"(?P<date>\d{8})"
    r"(?:_(?P<hms>\d{6}))?",
    re.IGNORECASE
)

# —— 宽松解析（任意位置搜索，兜底 CollectionLogs 里命名不标准的）——
PAT_RELAX_ANY = re.compile(
    r"(?P<prefix>\d{3}_.+?)_"
    r"(?:(?P<side>Left|Right)_)?"
    r"(?P<date>\d{8})"
    r"(?:_(?P<hms>\d{6}))?",
    re.IGNORECASE
)

# —— 仅日期+时间（HHMMSS），最后兜底“离代表时间最近的会话” ——
PAT_DATE_TIME_ONLY = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6})")

# —— 手机“完整日志”txt：log_YYYYMMDD_HHMM_XXX_动作名_(FINAL|TEMP).txt ——
PAT_PHONE_TXT = re.compile(
    r"^log_(?P<date>\d{8})_(?P<hm>\d{4})_(?P<id>\d{3})_(?P<action>.+?)_(FINAL|TEMP)\.txt$",
    re.IGNORECASE
)

def parse_strict(name: str):
    m = PAT_STRICT.match(name)
    if not m:
        return None, None
    key = f"{m.group('prefix')}_{m.group('date')}"  # 采集号+动作+日期
    return key, m.group('hms')

def parse_relax_any(name: str):
    m = PAT_RELAX_ANY.search(name)
    if not m:
        return None, None
    key = f"{m.group('prefix')}_{m.group('date')}"
    return key, m.group('hms')

def parse_date_time_only(name: str):
    m = PAT_DATE_TIME_ONLY.search(name)
    if not m:
        return None, None
    return m.group('date'), m.group('hms')

def parse_phone_txt(name: str):
    """
    手机完整日志 txt，返回 (group_key, hms)；
    hms 用 HHMM + '00' 补秒。
    """
    m = PAT_PHONE_TXT.match(name)
    if not m:
        return None, None
    date = m.group('date')
    hm   = m.group('hm')      # 4位
    cid  = m.group('id')
    act  = m.group('action')
    key  = f"{cid}_{act}_{date}"
    hms  = hm + "00"          # HHMM -> HHMMSS（补 00 秒）
    return key, hms

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

def main():
    # 解析日期（命令行/预设/交互）
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
        # 允许继续（比如你想整理“混在别的 tar 名里的该日期文件”），但通常此时退出更直观：
        sys.exit(0)

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

        # 3) 聚合：严格 -> 手机txt专用 -> 宽松
        groups = defaultdict(lambda: {"files": [], "times": []})
        residuals = []  # 最后兜底

        for d in extracted_dirs:
            for f in collect_files(d, date_re):
                name = f.name
                key, hms = parse_strict(name)
                if not key:
                    # 手机完整日志 txt
                    if name.lower().startswith("log_") and name.lower().endswith(".txt"):
                        key, hms = parse_phone_txt(name)
                if not key:
                    key, hms = parse_relax_any(name)
                if key:
                    groups[key]["files"].append(f)
                    if hms:
                        groups[key]["times"].append(hms)
                else:
                    residuals.append(f)

        # 4) 计算每个会话代表时间（min）
        session_rep_time_sec = {}
        for key, info in groups.items():
            rep_hms = min(info["times"]) if info["times"] else None
            session_rep_time_sec[key] = hhmmss_to_sec(rep_hms) if rep_hms else None

        # 5) 兜底：仅日期+时间（支持 HHMMSS；若文件只有 HHMM，就补 00）
        misc_dir = day_root / f"misc_{selected_date}"
        for f in residuals:
            date_only, hms = parse_date_time_only(f.name)
            if not hms:
                m_hm = re.search(rf"{selected_date}_(\d{{4}})", f.name)
                if m_hm:
                    hms = m_hm.group(1) + "00"
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
                if cand_key:
                    groups[cand_key]["files"].append(f)
                    continue
            # 实在无法归属，放 misc
            misc_dir.mkdir(parents=True, exist_ok=True)
            target = misc_dir / f.name
            print(f"[MISC ] {f}  ->  {target}")
            safe_move(f, target)
            total_moved += 1

        # 6) 输出：<OUTPUT_ROOT>/<DATE>/<会话名>/
        for key in sorted(groups.keys()):
            times = groups[key]["times"]
            rep = min(times) if times else None
            folder_name = f"{key}_{rep}" if rep else key
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

if __name__ == "__main__":
    main()
