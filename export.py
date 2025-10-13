# export.py  —— 只导出外部目录 /sdcard/Android/data/<pkg>/files
# 用法：
#   python export.py            # 导出今天
#   python export.py 20251012   # 导出指定日期（yyyyMMdd）

import sys
import subprocess
import pathlib
import datetime

APP_PKG = "com.dlut.wearosbleserver"
# 两个等价外部路径，取第一个存在且有匹配的
EXT_BASES = [
    f"/sdcard/Android/data/{APP_PKG}/files",
    f"/storage/emulated/0/Android/data/{APP_PKG}/files",
]

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

def test_online(serial):
    return adb("-s", serial, "get-state").returncode == 0

def choose_ext_base(serial):
    """返回第一个包含当日会话目录的外部 base；若都没有返回 None。"""
    for base in EXT_BASES:
        cmd = f"sh -c 'ls -1d {base}/*_{DATE}_* 2>/dev/null || echo NO_MATCH'"
        p = adb("-s", serial, "shell", cmd, capture=True)
        out = (p.stdout or "").strip()
        if out and "NO_MATCH" not in out:
            # 打印匹配，确认确实命中外部目录
            print(out)
            return base
    return None

def exec_out_to_files(serial, inner_cmd, out_path, err_path):
    with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        proc = subprocess.Popen(["adb", "-s", serial, "exec-out", inner_cmd],
                                stdout=fout, stderr=ferr)
        return proc.wait()

def export_one(name, serial):
    print(f"[CHECK] {name} ({serial})")
    if not test_online(serial):
        print(f"[MISS ] {name} not connected.\n"); return
    print(f"[FOUND] {name} is connected.")

    base = choose_ext_base(serial)
    if base is None:
        print(f"[WARN ] No external folders for {DATE} on {name}. 外部目录没有匹配，会话可能尚未写入或日期不对。\n")
        return

    outfile = EXPORT_DIR / f"{DATE}_{name}.tar"
    stderrf = LOG_DIR / f"{name}_{DATE}.stderr"
    print(f"[WORK ] Exporting (EXTERNAL ONLY) from '{base}' -> '{outfile}' ...")

    inner = (
        f"sh -c '"
        f"cd {base} || exit 2; "
        f"MATCHES=$(ls -1d *_{DATE}_* 2>/dev/null | head -n1); "
        f"if [ -z \"$MATCHES\" ]; then echo NO_MATCH >&2; exit 3; fi; "
        f"tar -cf - *_{DATE}_*'"
    )

    exec_out_to_files(serial, inner, str(outfile), str(stderrf))

    if not outfile.exists() or outfile.stat().st_size == 0:
        reason = ""
        if stderrf.exists():
            try: reason = stderrf.read_text(errors="ignore")
            except: pass
        print(f"[WARN ] Output empty. See '{stderrf}'.\nDetails:\n{reason}")
        return

    print(f"[DONE ] {name} -> {outfile} ({outfile.stat().st_size} bytes)\n")

def main():
    for name, serial in WATCHES:
        export_one(name, serial)
    print(f"Exports in: {EXPORT_DIR}")
    print(f"Logs in   : {LOG_DIR}")

if __name__ == "__main__":
    main()
