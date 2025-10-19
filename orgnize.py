# organize_from_exports.py
# 从 export_all.py 生成的 tar 包自动解压，按“采集号+动作+日期”聚合。
# - 忽略 Left/Right；时间 HHMMSS 仅用于命名代表时间；
# - 直接在解压出的 Phone_CollectionLogs 中识别手机完整日志 txt：
#   log_YYYYMMDD_HHMM_XXX_动作名_(FINAL|TEMP).txt

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
DATE         = None  # 例如 "20251019"；None=今天
ALLOW_EXTS   = {".csv", ".jsonl", ".json", ".log", ".txt"}  # 收集的扩展名
# =====================

def _today():
    return datetime.datetime.now().strftime("%Y%m%d")

DATE = DATE or _today()
DATE_RE = re.compile(rf"{DATE}")
print(f"[INFO ] Organizing exports for DATE={DATE}")

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
#    例：log_20251019_1539_001_腕旋前后_FINAL.txt
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

def collect_files(root: pathlib.Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in ALLOW_EXTS and DATE_RE.search(p.name):
            files.append(p)
    return files

def main():
    if not IMPORT_DIR.exists():
        print(f"[ERROR] IMPORT_DIR 不存在：{IMPORT_DIR}")
        sys.exit(1)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    day_root = OUTPUT_ROOT / DATE
    day_root.mkdir(parents=True, exist_ok=True)

    # 1) 找到该日期的 tar
    tars = sorted(IMPORT_DIR.glob(f"{DATE}_*.tar"))
    if not tars:
        print(f"[WARN ] 未在 {IMPORT_DIR} 找到 {DATE}_*.tar")
        sys.exit(0)

    print("[INFO ] TAR files:")
    for t in tars:
        print("  -", t.name)

    total_moved = 0
    sessions_count = 0

    with tempfile.TemporaryDirectory(prefix=f"org_{DATE}_") as tmpdir:
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
            for f in collect_files(d):
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
        misc_dir = day_root / f"misc_{DATE}"
        for f in residuals:
            date_only, hms = parse_date_time_only(f.name)
            if not hms:
                m_hm = re.search(rf"{DATE}_(\d{{4}})", f.name)
                if m_hm:
                    hms = m_hm.group(1) + "00"
            if date_only == DATE and hms:
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

    print(f"\n[DONE ] 日期 {DATE}：会话 {sessions_count} 个，移动文件 {total_moved} 个。")
    print(f"[PATH ] 输出目录：{day_root}")
    print(f"[NOTE ] 源 tar 仍在：{IMPORT_DIR}（未改动）")

if __name__ == "__main__":
    main()
