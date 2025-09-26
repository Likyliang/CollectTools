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

# 给输出文件加日期（不想带日期就改成 OUT = OUT_DIR / "data_quality_report.xlsx"）
DATE = datetime.datetime.now().strftime("%Y%m%d")
OUT: Path = OUT_DIR / f"data_quality_report_{DATE}.xlsx"

REQ_COLS = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _to_float(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")

def _pick_time_in_seconds(df: pd.DataFrame) -> np.ndarray:
    ldf = _lower_cols(df)
    for col in ["corrected_wall_clock", "corrected_wall_clock_ms"]:
        if col in ldf:
            t_ms = _to_float(ldf[col]); t_ms = t_ms[np.isfinite(t_ms)]
            if len(t_ms) >= 2: return (t_ms - t_ms[0]) / 1e3
    for col in ["wall_clock", "wall_clock_ms"]:
        if col in ldf:
            t_ms = _to_float(ldf[col]); t_ms = t_ms[np.isfinite(t_ms)]
            if len(t_ms) >= 2: return (t_ms - t_ms[0]) / 1e3
    for col in ["event_timestamp", "event_timestamp_ns"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if len(t_ns) >= 2: return (t_ns - t_ns[0]) / 1e9
    if "delta_ns" in ldf:
        d_ns = _to_float(ldf["delta_ns"])
        d_ns = np.where(np.isfinite(d_ns) & (d_ns > 0), d_ns, np.nan)
        d_ns = d_ns[~np.isnan(d_ns)]
        if len(d_ns) >= 2:
            t_ns = np.cumsum(d_ns)
            return (t_ns - t_ns[0]) / 1e9
    return np.arange(len(ldf), dtype="float64")

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
    if len(t) < 2:
        return {"file": str(path), "n_rows": int(len(df)), "error": "too few samples"}

    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return {"file": str(path), "n_rows": int(len(df)), "error": "all dt <= 0 or non-finite"}

    mean_dt = float(np.mean(dt))
    std_dt  = float(np.std(dt))
    fs      = (1.0 / mean_dt) if mean_dt > 0 else np.nan
    cv_dt   = (std_dt / mean_dt) if mean_dt > 0 else np.nan

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
        "dt_min_s":  round(float(np.min(dt)), 6),
        "dt_max_s":  round(float(np.max(dt)), 6),
    }
    out.update(nan_ratios)
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

def main():
    df = walk_all(ROOT)
    if "fs_est_Hz" in df.columns:
        df = df.sort_values(by=["fs_est_Hz"], ascending=False, na_position="last")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(30))
    df.to_excel(OUT, index=False)          # pandas 支持 Path
    print(f"✅ 质量报告已写入: {OUT.resolve()}")

if __name__ == "__main__":
    main()
