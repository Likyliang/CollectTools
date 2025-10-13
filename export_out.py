# export_all.py
# 用法：
#   python export_all.py            # 导出今天
#   python export_all.py 20251012   # 导出指定日期（yyyyMMdd）

import sys
import subprocess
import pathlib
import datetime
import re

# ====== 基本配置 ======
WATCH_PKG = "com.dlut.wearosbleserver"
PHONE_PKG = "com.dlut.androidbleclient"

WATCHES = [
    ("Watch_A", "adb-RFAX20LXZVD-xtiC5U._adb-tls-connect._tcp"),
    ("Watch_B", "adb-RFAX20M1FHH-7IKjNY._adb-tls-connect._tcp"),
]

EXPORT_DIR = pathlib.Path(r"D:\Data\Watch_Data_original")
LOG_DIR = pathlib.Path(__file__).resolve().parent / "_logs"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y%m%d")
print(f"[INFO ] date={DATE}  pattern=*_{DATE}_*")

# ====== ADB 工具 ======
def adb(*args, capture=False, text=True):
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        if text:
            kwargs["text"] = True
            kwargs["encoding"] = "utf-8"
            kwargs["errors"] = "replace"
    return subprocess.run(["adb", *args], **kwargs, check=False)

def exec_out_to_files(serial, inner_cmd, out_path, err_path):
    with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        proc = subprocess.Popen(["adb", "-s", serial, "exec-out", inner_cmd],
                                stdout=fout, stderr=ferr)
        return proc.wait()

def test_online(serial):
    return adb("-s", serial, "get-state").returncode == 0

# ====== 手表：外部目录导出 ======
WATCH_EXT_BASES = [
    lambda pkg: f"/sdcard/Android/data/{pkg}/files",
    lambda pkg: f"/storage/emulated/0/Android/data/{pkg}/files",
]

def choose_watch_ext_base(serial, pkg):
    """返回包含当天会话目录的外部 base；否则 None。"""
    for basef in WATCH_EXT_BASES:
        base = basef(pkg)
        cmd = f"sh -c 'ls -1d {base}/*_{DATE}_* 2>/dev/null || echo NO_MATCH'"
        p = adb("-s", serial, "shell", cmd, capture=True)
        out = (p.stdout or "").strip()
        if out and "NO_MATCH" not in out:
            print(out)
            return base
    return None

def export_watch_external(name, serial, pkg):
    print(f"[CHECK] {name} ({serial})")
    if not test_online(serial):
        print(f"[MISS ] {name} not connected.\n"); return
    print(f"[FOUND] {name} is connected.")

    base = choose_watch_ext_base(serial, pkg)
    if base is None:
        print(f"[WARN ] No external folders for {DATE} on {name}. 外部目录无当日会话。\n")
        return

    outfile = EXPORT_DIR / f"{DATE}_{name}.tar"
    stderrf = LOG_DIR / f"{name}_{DATE}.stderr"
    print(f"[WORK ] Exporting (WATCH EXTERNAL) from '{base}' -> '{outfile}' ...")

    inner = (
        f"sh -c '"
        f"cd {base} || exit 2; "
        f"MATCHES=$(ls -1d *_{DATE}_* 2>/dev/null | head -n1); "
        f"if [ -z \"$MATCHES\" ]; then echo NO_MATCH >&2; exit 3; fi; "
        f"tar -cf - *_{DATE}_*'"
    )

    exec_out_to_files(serial, inner, str(outfile), str(stderrf))

    if not outfile.exists() or outfile.stat().st_size == 0:
        reason = stderrf.read_text(errors="ignore") if stderrf.exists() else ""
        print(f"[WARN ] Output empty. See '{stderrf}'.\nDetails:\n{reason}")
        return

    print(f"[DONE ] {name} -> {outfile} ({outfile.stat().st_size} bytes)\n")

# ====== 手机：导出 TimeSyncPairs + Documents/CollectionLogs ======
def autodetect_phone_serial():
    p = adb("devices", capture=True)
    lines = (p.stdout or "").splitlines()
    devs = [ln.split()[0] for ln in lines if "\tdevice" in ln]
    # 你当前环境基本只有一台手机在连；直接取第一个即可
    return devs[0] if devs else None

def phone_paths(subdir):
    return f"/sdcard/Android/data/{PHONE_PKG}/files/{subdir}"

def list_remote(serial, path):
    p = adb("-s", serial, "shell", f"sh -c 'ls -1 {path} 2>/dev/null || true'", capture=True)
    return (p.stdout or "").strip().splitlines()

def export_phone_dirs(serial):
    print(f"[CHECK] Phone (auto)")
    if not serial or not test_online(serial):
        print(f"[MISS ] Phone not connected.\n"); return
    print(f"[FOUND] Phone autodetected as serial: {serial}")

    pairs_root = phone_paths("TimeSyncPairs")
    logs_root  = phone_paths("Documents/CollectionLogs")

    # TimeSyncPairs：文件名不含日期（如 00h_Left_xxx.jsonl），直接全量导出当天产生的目录——
    # 这里直接全量导出该目录（一般数量不大）
    pairs_list = list_remote(serial, pairs_root)
    need_pairs = len(pairs_list) > 0

    # CollectionLogs：按文件名里的 YYYYMMDD 过滤
    logs_list = list_remote(serial, logs_root)
    picked_logs = [f for f in logs_list if re.search(rf"{DATE}", f)]
    need_logs = len(picked_logs) > 0

    if not need_pairs and not need_logs:
        print(f"[WARN ] No external folders for {DATE} on Phone. 可能今天手机侧未产出文件。")
        return

    # 分别导出两个小 tar，更直观
    if need_pairs:
        out_pairs = EXPORT_DIR / f"{DATE}_Phone_TimeSyncPairs.tar"
        err_pairs = LOG_DIR / f"Phone_TimeSyncPairs_{DATE}.stderr"
        print(f"[WORK ] Exporting (PHONE) TimeSyncPairs -> '{out_pairs}' ...")
        cmd_pairs = f"sh -c 'cd {pairs_root} && tar -cf - . 2>/dev/null || true'"
        exec_out_to_files(serial, cmd_pairs, str(out_pairs), str(err_pairs))
        if out_pairs.exists() and out_pairs.stat().st_size > 0:
            print(f"[DONE ] Phone TimeSyncPairs -> {out_pairs} ({out_pairs.stat().st_size} bytes)")
        else:
            reason = err_pairs.read_text(errors="ignore") if err_pairs.exists() else ""
            print(f"[WARN ] Empty TimeSyncPairs export. See '{err_pairs}'. Details:\n{reason}")

    if need_logs:
        out_logs = EXPORT_DIR / f"{DATE}_Phone_CollectionLogs.tar"
        err_logs = LOG_DIR / f"Phone_CollectionLogs_{DATE}.stderr"
        print(f"[WORK ] Exporting (PHONE) CollectionLogs (filtered by {DATE}) -> '{out_logs}' ...")
        files_quoted = " ".join([f"'{x}'" for x in picked_logs])
        cmd_logs = f"sh -c 'cd {logs_root} && tar -cf - {files_quoted} 2>/dev/null || true'"
        exec_out_to_files(serial, cmd_logs, str(out_logs), str(err_logs))
        if out_logs.exists() and out_logs.stat().st_size > 0:
            print(f"[DONE ] Phone CollectionLogs -> {out_logs} ({out_logs.stat().st_size} bytes)")
        else:
            reason = err_logs.read_text(errors="ignore") if err_logs.exists() else ""
            print(f"[WARN ] Empty CollectionLogs export. See '{err_logs}'. Details:\n{reason}")

    print("[INFO ] Phone export finished.\n")

def main():
    # 1) 导出两块手表（外部 files 下当天会话目录）
    for name, serial in WATCHES:
        export_watch_external(name, serial, WATCH_PKG)

    # 2) 导出手机（TimeSyncPairs + CollectionLogs）
    phone_serial = autodetect_phone_serial()
    export_phone_dirs(phone_serial)

    print(f"Exports in: {EXPORT_DIR}")
    print(f"Logs in   : {LOG_DIR}")

if __name__ == "__main__":
    main()
