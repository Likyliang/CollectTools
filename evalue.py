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

REQ_COLS = ["accel_x","accel_y","accel_z","gyro_x","gyro_y","gyro_z"]

def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _to_float(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")

def _pick_time_in_seconds(df: pd.DataFrame) -> np.ndarray:
    """
    时间列优先级（从高到低）：
    1) event_timestamp_ns  (IMU硬件时间戳，单调且纳秒)
    2) gyro_event_timestamp_ns（兜底）
    3) corrected_wall_clock_ms -> s
    4) wall_clock_ms -> s
    """
    ldf = _lower_cols(df)

    # 1) accel 硬件时间戳
    for col in ["event_timestamp_ns", "event_timestamp"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if t_ns.size >= 2:
                return (t_ns - t_ns[0]) / 1e9

    # 2) gyro 硬件时间戳（少见但给兜底）
    for col in ["gyro_event_timestamp_ns"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if t_ns.size >= 2:
                return (t_ns - t_ns[0]) / 1e9

    # 3) 纠正后的墙钟（毫秒）——相同毫秒加极小扰动避免 dt=0
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
                    t_ms = t_ms + offsets * 1e-6  # 1e-6 ms 抖动
                return (t_ms - t_ms[0]) / 1e3

    # 4) 原始墙钟（毫秒）
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

    # 5) 实在没有：用行号伪时间（仅为避免崩溃）
    return np.arange(len(ldf), dtype="float64")

def _mode_stats(df: pd.DataFrame) -> dict:
    """统计 gyro_mode 的数量与占比；并对 EXACT 行的 delta_ns 做统计（ms）"""
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

        # EXACT 的 delta_ns 统计（ms）
        if "delta_ns" in ldf.columns:
            delta = pd.to_numeric(ldf.loc[modes == "EXACT", "delta_ns"], errors="coerce")
            delta = delta[np.isfinite(delta)]
            if delta.size >= 1:
                d_ms = delta.to_numpy(dtype="float64") / 1e6
                out["exact_delta_ms_mean"] = round(float(np.mean(d_ms)), 6)
                out["exact_delta_ms_p95"]  = round(float(np.percentile(d_ms, 95)), 6)
            else:
                out["exact_delta_ms_mean"] = np.nan
                out["exact_delta_ms_p95"]  = np.nan
    else:
        # 没有 gyro_mode 列时给出空占位，避免下游出错
        out["mode_exact_cnt"] = out["mode_interp_cnt"] = out["mode_none_cnt"] = np.nan
        out["mode_exact_ratio"] = out["mode_interp_ratio"] = out["mode_none_ratio"] = np.nan
        out["exact_delta_ms_mean"] = out["exact_delta_ms_p95"] = np.nan

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

    # dt 计算
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

    # 缺失率
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
    out.update(_mode_stats(df))       # ✅ 新增：gyro_mode 与 delta 质量指标
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
    df.to_excel(OUT, index=False)
    print(f"✅ 质量报告已写入: {OUT.resolve()}")

if __name__ == "__main__":
    main()
