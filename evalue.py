# dq_report.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import datetime

# ===== 路径配置 =====
ROOT = r"D:\NEWALL"                         # 数据根目录
OUT_DIR = Path(r"D:\Data\Watch_Reports")    # 报告目录（固定）
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATE = datetime.datetime.now().strftime("%Y%m%d")
OUT: Path = OUT_DIR / f"data_quality_report_{DATE}.xlsx"

# 传感器列（存在即统计缺失率）
REQ_COLS = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]

# --- 报警阈值（可按需调整） ---
FS_MIN_OK = 98.8          # 采样率下限(Hz)
EXACT_RATIO_MIN = 0.95    # 直接命中比例下限
DT_STD_MAX = 0.0012       # 抖动上限(s)

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _to_float(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")

def _pick_time_in_seconds(df: pd.DataFrame) -> np.ndarray:
    """
    时间列优先级（从高到低）：
    1) event_timestamp_ns  (IMU硬件时间戳)
    2) gyro_event_timestamp_ns（兜底）
    3) corrected_wall_clock_ms -> s
    4) wall_clock_ms -> s
    """
    ldf = _lower_cols(df)

    for col in ["event_timestamp_ns", "event_timestamp"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if t_ns.size >= 2:
                return (t_ns - t_ns[0]) / 1e9

    for col in ["gyro_event_timestamp_ns"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if t_ns.size >= 2:
                return (t_ns - t_ns[0]) / 1e9

    for col in ["corrected_wall_clock_ms", "corrected_wall_clock"]:
        if col in ldf:
            t_ms = _to_float(ldf[col]); t_ms = t_ms[np.isfinite(t_ms)]
            if t_ms.size >= 2:
                if np.any(np.diff(t_ms) == 0):
                    _, inv, counts = np.unique(t_ms, return_inverse=True, return_counts=True)
                    offsets = np.zeros_like(t_ms, dtype="float64")
                    idx = np.argsort(inv, kind="stable")
                    start = 0
                    for c in counts:
                        if c > 1:
                            offsets[idx[start:start+c]] = np.arange(c, dtype="float64")
                        start += c
                    t_ms = t_ms + offsets * 1e-6
                return (t_ms - t_ms[0]) / 1e3

    for col in ["wall_clock_ms", "wall_clock"]:
        if col in ldf:
            t_ms = _to_float(ldf[col]); t_ms = t_ms[np.isfinite(t_ms)]
            if t_ms.size >= 2:
                if np.any(np.diff(t_ms) == 0):
                    _, inv, counts = np.unique(t_ms, return_inverse=True, return_counts=True)
                    offsets = np.zeros_like(t_ms, dtype="float64")
                    idx = np.argsort(inv, kind="stable")
                    start = 0
                    for c in counts:
                        if c > 1:
                            offsets[idx[start:start+c]] = np.arange(c, dtype="float64")
                        start += c
                    t_ms = t_ms + offsets * 1e-6
                return (t_ms - t_ms[0]) / 1e3

    return np.arange(len(ldf), dtype="float64")

def _mode_stats(df: pd.DataFrame) -> dict:
    """统计 gyro_mode 数量/比例；对 EXACT 的 delta_ns 给出 P50/P95/mean（ms），并过滤异常/占位 0。"""
    ldf = _lower_cols(df)
    out = {}

    if "gyro_mode" in ldf.columns:
        modes = ldf["gyro_mode"].astype(str).str.strip().str.upper()
        n = len(modes)
        n_exact  = int((modes == "EXACT").sum())
        n_interp = int((modes == "INTERP").sum())
        n_none   = int((modes == "NONE").sum())

        out["mode_exact_cnt"]  = n_exact
        out["mode_interp_cnt"] = n_interp
        out["mode_none_cnt"]   = n_none
        out["mode_exact_ratio"]  = round(n_exact  / n, 6) if n else np.nan
        out["mode_interp_ratio"] = round(n_interp / n, 6) if n else np.nan
        out["mode_none_ratio"]   = round(n_none   / n, 6) if n else np.nan

        # EXACT 的 delta_ns：过滤 0 与>20ms（上限）并换算成 ms
        if "delta_ns" in ldf.columns and n_exact > 0:
            delta_ns = pd.to_numeric(ldf.loc[modes == "EXACT", "delta_ns"], errors="coerce")
            # 仅保留 (0, 20ms) 的有效值
            valid = delta_ns[(delta_ns > 0) & (delta_ns < 20_000_000)]
            d = valid.to_numpy(dtype="float64") / 1e6  # ms
            if d.size >= 2:
                out["exact_delta_ms_p50"]  = round(float(np.percentile(d, 50)), 6)
                out["exact_delta_ms_p95"]  = round(float(np.percentile(d, 95)), 6)
                out["exact_delta_ms_mean"] = round(float(np.mean(d)), 6)
            elif d.size == 1:
                out["exact_delta_ms_p50"] = out["exact_delta_ms_p95"] = out["exact_delta_ms_mean"] = round(float(d[0]), 6)
            else:
                out["exact_delta_ms_p50"] = out["exact_delta_ms_p95"] = out["exact_delta_ms_mean"] = np.nan
        else:
            out["exact_delta_ms_p50"] = out["exact_delta_ms_p95"] = out["exact_delta_ms_mean"] = np.nan
    else:
        out["mode_exact_cnt"] = out["mode_interp_cnt"] = out["mode_none_cnt"] = np.nan
        out["mode_exact_ratio"] = out["mode_interp_ratio"] = out["mode_none_ratio"] = np.nan
        out["exact_delta_ms_p50"] = out["exact_delta_ms_p95"] = out["exact_delta_ms_mean"] = np.nan

    return out

def analyze_csv(path: Path) -> dict:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"file": str(path), "error": f"read_csv failed: {e}"}

    df = _lower_cols(df)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        return {"file": str(path), "error": f"missing cols: {missing}"}

    t = _pick_time_in_seconds(df)
    if t.size < 2:
        return {"file": str(path), "n_rows": int(len(df)), "error": "too few samples"}

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if dt.size == 0:
        return {"file": str(path), "n_rows": int(len(df)), "error": "all dt <= 0 or non-finite"}

    mean_dt = float(np.mean(dt))
    std_dt  = float(np.std(dt))
    fs      = (1.0 / mean_dt) if mean_dt > 0 else np.nan
    cv_dt   = (std_dt / mean_dt) if mean_dt > 0 else np.nan
    p50_dt  = float(np.percentile(dt, 50))
    p95_dt  = float(np.percentile(dt, 95))

    nan_ratios = {}
    for c in REQ_COLS:
        col = pd.to_numeric(df[c], errors="coerce")
        nan_ratios[f"nan_{c}"] = float(col.isna().mean())

    out = {
        "file": str(path),
        "n_rows": int(len(df)),
        "fs_est_Hz": round(fs, 3),
        "dt_mean_s": round(mean_dt, 6),
        "dt_std_s":  round(std_dt, 6),
        "dt_cv":     round(cv_dt, 6),
        "dt_p50_s":  round(p50_dt, 6),
        "dt_p95_s":  round(p95_dt, 6),
        "dt_min_s":  round(float(np.min(dt)), 6),
        "dt_max_s":  round(float(np.max(dt)), 6),
    }
    out.update(nan_ratios)
    out.update(_mode_stats(df))
    return out

def walk_all(root: str) -> pd.DataFrame:
    rows = []
    base = Path(root)
    csvs = sorted(base.rglob("*.csv"))
    if not csvs:
        return pd.DataFrame([{"file": str(base), "error": "no csv found"}])
    for f in csvs:
        rows.append(analyze_csv(f))
    return pd.DataFrame(rows)

def _reorder_full_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    front = [
        "file", "n_rows",
        "fs_est_Hz",
        "mode_exact_ratio", "mode_interp_ratio", "mode_none_ratio",
        "exact_delta_ms_p50", "exact_delta_ms_p95", "exact_delta_ms_mean",
        "dt_mean_s", "dt_std_s", "dt_cv", "dt_p50_s", "dt_p95_s", "dt_min_s", "dt_max_s",
    ]
    ordered = [c for c in front if c in cols]
    rest = [c for c in cols if c not in ordered]
    return df[ordered + rest]

def _make_brief(df: pd.DataFrame) -> pd.DataFrame:
    keep = ["file", "n_rows", "fs_est_Hz", "mode_exact_ratio"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    brief = df[keep].copy()
    brief["warn_fs"]     = (brief["fs_est_Hz"] < FS_MIN_OK)
    brief["warn_exact"]  = (brief["mode_exact_ratio"] < EXACT_RATIO_MIN)
    if "dt_std_s" in df.columns:
        brief["warn_jitter"] = (df["dt_std_s"] > DT_STD_MAX)
    else:
        brief["warn_jitter"] = False

    brief = brief.sort_values(
        by=["warn_fs","warn_exact","warn_jitter","fs_est_Hz"],
        ascending=[False, False, False, False],
        na_position="last"
    )
    return brief

def main():
    df = walk_all(ROOT)
    full = _reorder_full_columns(df)
    brief = _make_brief(full)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("=== BRIEF ===")
        print(brief.head(30))
        print("\n=== FULL ===")
        print(full.head(30))

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        brief.to_excel(xw, sheet_name="brief", index=False)
        full.sort_values(by=["fs_est_Hz"], ascending=False, na_position="last").to_excel(
            xw, sheet_name="full", index=False
        )
    print(f"✅ 质量报告已写入: {OUT.resolve()}")

if __name__ == "__main__":
    main()
