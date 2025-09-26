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
                t = (t_ns - t_ns[0]) / 1e9
                return t

    # 2) gyro 硬件时间戳（少见但给兜底）
    for col in ["gyro_event_timestamp_ns"]:
        if col in ldf:
            t_ns = _to_float(ldf[col]); t_ns = t_ns[np.isfinite(t_ns)]
            if t_ns.size >= 2:
                t = (t_ns - t_ns[0]) / 1e9
                return t

    # 3) 纠正后的墙钟（毫秒）——注意可能有重复/台阶，做去重扰动
    for col in ["corrected_wall_clock_ms", "corrected_wall_clock"]:
        if col in ldf:
            t_ms = _to_float(ldf[col]); t_ms = t_ms[np.isfinite(t_ms)]
            if t_ms.size >= 2:
                # 去重/台阶小抖动：对相同毫秒内的样本加极小递增抖动，避免 dt=0
                # 抖动幅度 << 1ms，不影响统计（这里用 1e-6 ms = 1e-9 s）
                if np.any(np.diff(t_ms) == 0):
                    # 计算每个相同值的序号
                    _, inv, counts = np.unique(t_ms, return_inverse=True, return_counts=True)
                    offsets = np.zeros_like(t_ms, dtype="float64")
                    idx = np.argsort(inv, kind="stable")
                    # 为每个重复组生成 0,1,2,... 的序号
                    start = 0
                    for c in counts:
                        if c > 1:
                            offsets[idx[start:start+c]] = np.arange(c, dtype="float64")
                        start += c
                    t_ms = t_ms + offsets * 1e-6  # 每个重复样本+1e-6ms
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

def analyze_csv(path: Path) -> dict:
    try:
        # 保持默认 parser；后续对列手动 numeric 转换
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
