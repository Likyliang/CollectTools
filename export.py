# export.py
import os, sys, subprocess, pathlib, datetime

# === 配置 ===
APP_PKG = "com.dlut.wearosbleserver"
APP_DIRS = [
    f"/data/user/0/{APP_PKG}/files",
    f"/data/data/{APP_PKG}/files",  # 回退路径
]
WATCHES = [
    ("Watch_A", "adb-RFAX20LXZVD-xtiC5U._adb-tls-connect._tcp"),
    ("Watch_B", "adb-RFAX20M1FHH-7IKjNY._adb-tls-connect._tcp"),
]

# === 目录 & 日期 ===
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# 将导出目录改到 D:\Data\Watch_Data_original
# 注意：如果在非 Windows 环境运行，这个路径会被当作普通字符串处理；在 Windows 上则是标准盘符路径
EXPORT_DIR = pathlib.Path(r"D:\Data\Watch_Data_original")

# 日志目录仍放在脚本同级（也可改为固定路径，按需）
LOG_DIR = SCRIPT_DIR / "_logs"

# 自动创建目录
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DATE = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now().strftime("%Y%m%d")
print(f"[INFO ] date={DATE}  pattern=*_{DATE}_*")

def adb(*args, capture=False, text=True):
    """小工具：调用 adb。"""
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        kwargs["text"] = text
    return subprocess.run(["adb", *args], **kwargs, check=False)

def test_online(serial):
    return adb("-s", serial, "get-state").returncode == 0

def test_runas(serial):
    p = adb("-s", serial, "shell", f"run-as {APP_PKG} id", capture=True)
    return p.returncode == 0 and "uid=" in (p.stdout or "")

def list_day_dirs(serial):
    """返回 (选用的APP目录, 匹配到的目录列表[str])；如果没有匹配，返回 (None, [])。"""
    for base in APP_DIRS:
        cmd = f"run-as {APP_PKG} sh -c 'ls -1d {base}/*_{DATE}_* 2>/dev/null || echo NO_MATCH'"
        p = adb("-s", serial, "shell", cmd, capture=True)
        out = (p.stdout or "").strip()
        # 打印给用户看（可见目录）
        if out:
            for line in out.splitlines():
                print(line)
        if "NO_MATCH" not in out:
            # 有匹配
            items = [ln.strip() for ln in out.splitlines() if ln.strip()]
            return base, items
    return None, []

def exec_out_to_files(serial, inner_cmd, out_path, err_path):
    """
    用 Popen 直接把 stdout/stderr 写文件；避免 cmd /c + 重定向的引号/编码问题。
    inner_cmd 例如： "run-as com.xxx sh -c 'cd ...; tar -cf - *_DATE_*'"
    """
    with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
        proc = subprocess.Popen(["adb", "-s", serial, "exec-out", inner_cmd],
                                stdout=fout, stderr=ferr)
        return proc.wait()

def export_one(name, serial):
    print(f"[CHECK] {name} ({serial})")
    if not test_online(serial):
        print(f"[MISS ] {name} not connected.\n"); return
    print(f"[FOUND] {name} is connected.")
    if not test_runas(serial):
        print(f"[ERROR] run-as failed on {name}. Make sure the app is a debuggable build.\n")
        return

    base, items = list_day_dirs(serial)
    if not items:
        print(f"[WARN ] No folders for {DATE} on {name} (pattern '*_{DATE}_*').\n"); return

    outfile = EXPORT_DIR / f"{DATE}_{name}.tar"
    stderrf = LOG_DIR / f"{name}_{DATE}.stderr"
    print(f"[WORK ] Exporting to '{outfile}' ...")

    # 设备端执行：有匹配才打包，避免空 tar
    inner = (
        f"run-as {APP_PKG} sh -c '"
        f"cd {base} || exit 2; "
        f"MATCHES=$(ls -1d *_{DATE}_* 2>/dev/null | head -n1); "
        f"if [ -z \"$MATCHES\" ]; then echo NO_MATCH >&2; exit 3; fi; "
        f"tar -cf - *_{DATE}_*'"
    )
    rc = exec_out_to_files(serial, inner, str(outfile), str(stderrf))

    # 结果检查
    if not outfile.exists() or outfile.stat().st_size == 0:
        # 打印 stderr 里的原因，便于诊断
        reason = ""
        if stderrf.exists():
            try:
                reason = stderrf.read_text(errors="ignore")
            except:
                pass
        if "NO_MATCH" in reason:
            print(f"[WARN ] No folders at pack time (race?). See '{stderrf}'.\n")
        else:
            print(f"[WARN ] Output empty. See '{stderrf}'.\n")
        return

    print(f"[DONE ] {name} -> {outfile} ({outfile.stat().st_size} bytes)\n")

def main():
    for name, serial in WATCHES:
        export_one(name, serial)
    print(f"Exports in: {EXPORT_DIR}")
    print(f"Logs in   : {LOG_DIR}")

if __name__ == "__main__":
    main()
