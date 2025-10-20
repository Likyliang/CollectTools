# export_all.py
# 用法：
#   交互式：python export_all.py  ← 回车后按提示输入日期
#   非交互：python export_all.py 20251012   ← 仍然支持传参（yyyyMMdd）

import sys
import subprocess
import pathlib
import datetime
import re

# ====== 基本配置 ======
WATCH_PKG = "com.dlut.wearosbleserver"
PHONE_PKG = "com.dlut.androidbleclient"

EXPORT_DIR = pathlib.Path(r"D:\Data\Watch_Data_original")
LOG_DIR = pathlib.Path(__file__).resolve().parent / "_logs"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ====== 交互式日期解析 ======
def _today():
    return datetime.datetime.now().strftime("%Y%m%d")

def _yesterday():
    return (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y%m%d")

def resolve_date():
    """
    优先使用命令行参数；否则交互式输入。
    支持：
      - 直接输入 8 位日期：20251012
      - 回车：使用今天
      - t/T/today：今天
      - y/Y/yesterday：昨天
      - q/quit：退出
    """
    if len(sys.argv) > 1:
        cand = sys.argv[1].strip()
        if not re.fullmatch(r"\d{8}", cand):
            print(f"[ERROR] 非法日期参数: {cand}，应为 yyyyMMdd。")
            sys.exit(1)
        return cand

    while True:
        raw = input("请输入导出日期 (yyyyMMdd)，回车=今天，y=昨天，q=退出：").strip()
        if raw == "":
            return _today()
        if raw.lower() in ("t", "today"):
            return _today()
        if raw.lower() in ("y", "yesterday"):
            return _yesterday()
        if raw.lower() in ("q", "quit"):
            print("已取消。")
            sys.exit(0)
        if re.fullmatch(r"\d{8}", raw):
            return raw
        print("格式不对，请输入 8 位日期（例如 20251012），或直接回车/输入 y。")

DATE = resolve_date()
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

def _shell(serial, cmd):
    p = adb("-s", serial, "shell", cmd, capture=True)
    return (p.stdout or "").strip(), (p.stderr or "").strip(), p.returncode

# ====== 设备自动识别 ======
def list_attached_serials():
    p = adb("devices", capture=True)
    lines = (p.stdout or "").splitlines()
    devs = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 2 and parts[1] == "device":
            devs.append(parts[0])
    return devs

def has_pkg(serial, pkg):
    # pm path 优先；部分设备需使用 cmd package path
    out, err, _ = _shell(serial, f"pm path {pkg} || cmd package path {pkg} 2>/dev/null || true")
    return bool(out and "package:" in out)

def get_characteristics(serial):
    out, _, _ = _shell(serial, "getprop ro.build.characteristics")
    return out.lower()

def get_android_id(serial):
    # settings get secure android_id 更稳定（与用户/设备绑定），失败则退回 ro.serialno
    out, _, _ = _shell(serial, "settings get secure android_id 2>/dev/null || true")
    val = out.strip()
    if not val or val == "null":
        out2, _, _ = _shell(serial, "getprop ro.serialno")
        val = out2.strip() or serial
    return val

def classify_device(serial):
    # 先看是否安装了对应包，再看 build 特征
    is_phone_pkg = has_pkg(serial, PHONE_PKG)
    is_watch_pkg = has_pkg(serial, WATCH_PKG)
    ch = get_characteristics(serial)

    is_watch_char = "watch" in ch
    is_phone_char = "phone" in ch or "default" in ch  # 有些手机给 default/handheld

    if is_phone_pkg or (is_phone_char and not is_watch_pkg):
        return "phone"
    if is_watch_pkg or is_watch_char:
        return "watch"
    return "unknown"

def autodetect_devices():
    """返回 {'Phone': <serial或None>, 'Watch_A': <serial或None>, 'Watch_B': <serial或None>}"""
    serials = list_attached_serials()
    if not serials:
        print("[ERROR] 未检测到任何处于 device 状态的设备。请先 adb connect / 打开调试。")
        return {"Phone": None, "Watch_A": None, "Watch_B": None}

    phones, watches = [], []
    for s in serials:
        if not test_online(s):
            continue
        role = classify_device(s)
        if role == "phone":
            phones.append(s)
        elif role == "watch":
            watches.append(s)
        else:
            print(f"[WARN ] 未能识别角色: {s}（既非明确 phone 也非 watch）")

    phone_serial = phones[0] if phones else None

    # 对手表按稳定 ID 排序并命名为 A / B
    watch_with_id = [(s, get_android_id(s)) for s in watches]
    watch_with_id.sort(key=lambda x: x[1])  # 稳定、可跨电脑保持一致
    wa = watch_with_id[0][0] if len(watch_with_id) >= 1 else None
    wb = watch_with_id[1][0] if len(watch_with_id) >= 2 else None

    # 打印识别结果
    print("[AUTO ] 设备识别结果：")
    if phone_serial:
        print(f"        Phone   -> {phone_serial}")
    else:
        print("        Phone   -> 未找到（将跳过手机侧导出）")

    if wa:
        print(f"        Watch_A -> {wa}")
    else:
        print("        Watch_A -> 未找到")

    if wb:
        print(f"        Watch_B -> {wb}")
    else:
        print("        Watch_B -> 未找到")

    return {"Phone": phone_serial, "Watch_A": wa, "Watch_B": wb}

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
    if not serial or not test_online(serial):
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

    # —— TimeSyncPairs：现在文件名已含日期，按 *_{DATE}_* 过滤 —— #
    pairs_list = list_remote(serial, pairs_root)
    picked_pairs = [f for f in pairs_list if re.search(rf"_{DATE}_", f)]
    need_pairs = len(picked_pairs) > 0

    # —— CollectionLogs：按日期过滤 —— #
    logs_list = list_remote(serial, logs_root)
    picked_logs = [f for f in logs_list if re.search(rf"{DATE}", f)]
    need_logs = len(picked_logs) > 0

    if not need_pairs and not need_logs:
        print(f"[WARN ] No phone files for {DATE}. 可能当天手机侧未产出匹配文件。")
        return

    if need_pairs:
        out_pairs = EXPORT_DIR / f"{DATE}_Phone_TimeSyncPairs.tar"
        err_pairs = LOG_DIR / f"Phone_TimeSyncPairs_{DATE}.stderr"
        print(f"[WORK ] Exporting (PHONE) TimeSyncPairs (filtered by {DATE}) -> '{out_pairs}' ...")
        cmd_pairs = (
            f"sh -c '"
            f"cd {pairs_root} || exit 2; "
            f"MATCHES=$(ls -1 *_{DATE}_* 2>/dev/null | head -n1); "
            f"if [ -z \"$MATCHES\" ]; then echo NO_MATCH >&2; exit 3; fi; "
            f"tar -cf - *_{DATE}_*'"
        )
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
        cmd_logs = (
            f"sh -c '"
            f"cd {logs_root} || exit 2; "
            f"MATCHES=$(ls -1 *{DATE}* 2>/dev/null | head -n1); "
            f"if [ -z \"$MATCHES\" ]; then echo NO_MATCH >&2; exit 3; "
            f"fi; tar -cf - *{DATE}*'"
        )
        exec_out_to_files(serial, cmd_logs, str(out_logs), str(err_logs))
        if out_logs.exists() and out_logs.stat().st_size > 0:
            print(f"[DONE ] Phone CollectionLogs -> {out_logs} ({out_logs.stat().st_size} bytes)")
        else:
            reason = err_logs.read_text(errors="ignore") if err_logs.exists() else ""
            print(f"[WARN ] Empty CollectionLogs export. See '{err_logs}'. Details:\n{reason}")

    print("[INFO ] Phone export finished.\n")

# ====== 主流程 ======
def main():
    roles = autodetect_devices()
    # 1) 导出两块手表（外部 files 下当天会话目录）
    if roles.get("Watch_A"):
        export_watch_external("Watch_A", roles["Watch_A"], WATCH_PKG)
    else:
        print("[INFO ] 跳过 Watch_A。")

    if roles.get("Watch_B"):
        export_watch_external("Watch_B", roles["Watch_B"], WATCH_PKG)
    else:
        print("[INFO ] 跳过 Watch_B。")

    # 2) 导出手机（TimeSyncPairs + CollectionLogs）
    export_phone_dirs(roles.get("Phone"))

    print(f"Exports in: {EXPORT_DIR}")
    print(f"Logs in   : {LOG_DIR}")

if __name__ == "__main__":
    main()
