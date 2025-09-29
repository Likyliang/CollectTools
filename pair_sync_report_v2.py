# pair_sync_report_v2.py
# -*- coding: utf-8 -*-
from pathlib import Path
import re
import numpy as np
import pandas as pd

# ===== 配置 =====
ROOT = r"D:\NEWALL"
OUT  = Path(r"D:\Data\Watch_Reports\pair_sync_report_v2.xlsx")

# 阈值（与你前面讨论保持一致）
FS_MIN, FS_MAX          = 98.5, 100.5
JITTER_STD_MAX_S        = 0.0008    # 0.8 ms
OVERLAP_MIN             = 0.95
LAG_GOOD_MS             = 20.0
LAG_OK_MS               = 80.0
DRIFT_GOOD_PPM          = 100.0
DRIFT_OK_PPM            = 300.0

# ===== 基础工具 =====
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _to_float(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")

def pick_time_ms(df: pd.DataFrame) -> np.ndarray:
    """优先 corrected_wall_clock_ms；回退 event_timestamp_ns -> ms；再回退 wall_clock_ms。"""
    df = _lower_cols(df)
    for c in ("corrected_wall_clock_ms", "corrected_wall_clock"):
        if c in df:
            t = _to_float(df[c]); t = t[np.isfinite(t)]
            if t.size: return t
    for c in ("event_timestamp_ns", "event_timestamp"):
        if c in df:
            t = _to_float(df[c]); t = t[np.isfinite(t)]
            if t.size: return t / 1e6
    for c in ("wall_clock_ms", "wall_clock"):
        if c in df:
            t = _to_float(df[c]); t = t[np.isfinite(t)]
            if t.size: return t
    # 兜底：100Hz 伪时间（仅避免崩）
    return np.arange(len(df), dtype="float64") * 10.0

def accel_mag(df: pd.DataFrame) -> np.ndarray:
    df = _lower_cols(df)
    ax = _to_float(df.get("accel_x", pd.Series([np.nan]*len(df))))
    ay = _to_float(df.get("accel_y", pd.Series([np.nan]*len(df))))
    az = _to_float(df.get("accel_z", pd.Series([np.nan]*len(df))))
    m = np.sqrt(ax*ax + ay*ay + az*az)
    # 简单插补 NaN
    if np.isnan(m).any() and np.isfinite(m).any():
        idx = np.arange(m.size)
        m = np.interp(idx, idx[np.isfinite(m)], m[np.isfinite(m)])
    return m

def resample_linear(t_src_ms: np.ndarray, y: np.ndarray, t_dst_ms: np.ndarray) -> np.ndarray:
    """线性插值 + 常值外推"""
    return np.interp(t_dst_ms, t_src_ms, y, left=y[0], right=y[-1])

def fs_from_time(t_ms: np.ndarray) -> float:
    if len(t_ms) < 2: return np.nan
    dt = np.diff(t_ms) / 1000.0
    dt = dt[np.isfinite(dt)]
    if len(dt) == 0: return np.nan
    return 1.0 / np.mean(dt)

def jitter_std_s(t_ms: np.ndarray) -> float:
    if len(t_ms) < 2: return np.nan
    dt = np.diff(t_ms) / 1000.0
    dt = dt[np.isfinite(dt)]
    return float(np.std(dt)) if len(dt) else np.nan

def pair_key_from_path(path: Path) -> str:
    """
    把 004_draw2_Left_20250927_144250 → 004_draw2_20250927
    规则：去掉 _Left_/_Right_，再保留“采集号_动作_日期”三段。
    """
    stem = path.stem  # 文件名不含后缀
    s = stem.replace("_Left_", "_").replace("_Right_", "_")
    # 提取形如 001_drawX_YYYYMMDD 的前三段
    m = re.match(r"^([^_]+_[^_]+)_(\d{8})", s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return s  # 兜底

def find_pairs(csvs):
    """返回 {pair_key: {'L': Path, 'R': Path}}"""
    pairs = {}
    for f in csvs:
        p = Path(f)
        key = pair_key_from_path(p)
        is_left  = "_Left_"  in p.stem
        is_right = "_Right_" in p.stem
        if not (is_left or is_right):
            continue
        d = pairs.setdefault(key, {})
        d["L" if is_left else "R"] = p
    # 只保留成对的
    return {k:v for k,v in pairs.items() if "L" in v and "R" in v}

def overlap_ratio(tL, tR):
    a0, a1 = tL[0], tL[-1]
    b0, b1 = tR[0], tR[-1]
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0

def xcorr_lag_ms(tA_ms, sA, tB_ms, sB, max_lag_ms=300.0, grid_step_ms=10.0):
    """把两序列重采样到公共等间隔网格，做互相关，返回滞后(右相对左为正)、峰值。"""
    t0 = max(tA_ms[0], tB_ms[0])
    t1 = min(tA_ms[-1], tB_ms[-1])
    if t1 - t0 < 3*grid_step_ms:
        return np.nan, np.nan
    grid = np.arange(t0, t1, grid_step_ms)
    a = resample_linear(tA_ms, sA, grid)
    b = resample_linear(tB_ms, sB, grid)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    # numpy.correlate：full 模式下，正峰索引差对应 b 滞后 a 的样本数
    xc = np.correlate(b, a, mode="full")
    lags = np.arange(-len(a)+1, len(b)) * grid_step_ms
    # 限窗
    sel = (lags >= -max_lag_ms) & (lags <= max_lag_ms)
    lags = lags[sel]; xc = xc[sel]
    k = int(np.argmax(xc))
    return float(lags[k]), float(xc[k])

def drift_ppm_linear(tL_ms: np.ndarray, tR_ms: np.ndarray) -> float:
    """在重叠区用等间隔网格取样，拟合 tL ≈ a + b*tR；ppm=(1-b)*1e6。"""
    t0 = max(tL_ms[0], tR_ms[0])
    t1 = min(tL_ms[-1], tR_ms[-1])
    if t1 <= t0: return np.nan
    grid = np.linspace(t0, t1, 200)
    TL = resample_linear(tL_ms, tL_ms, grid)
    TR = resample_linear(tR_ms, tR_ms, grid)
    A = np.vstack([np.ones_like(TR), TR]).T
    a, b = np.linalg.lstsq(A, TL, rcond=None)[0]
    return float((1.0 - b) * 1e6)

def verdict_row(fsL, fsR, lag_ms, ppm, overlap):
    # 先看单机采样率/重叠
    if not (FS_MIN <= fsL <= FS_MAX and FS_MIN <= fsR <= FS_MAX):
        return "Review"
    if overlap < OVERLAP_MIN:
        return "Review"
    # 结合 lag & 漂移
    if abs(lag_ms) <= LAG_GOOD_MS and (np.isnan(ppm) or abs(ppm) <= DRIFT_GOOD_PPM):
        return "Direct"
    if abs(lag_ms) <= LAG_OK_MS and (np.isnan(ppm) or abs(ppm) <= DRIFT_OK_PPM):
        return "Shift"
    return "Review"

# ===== 主流程 =====
def main():
    base = Path(ROOT)
    csvs = sorted(str(p) for p in base.rglob("*.csv"))
    pairs = find_pairs(csvs)

    rows = []
    for key, lr in sorted(pairs.items()):
        left_p  = lr["L"]; right_p = lr["R"]
        dfL = _lower_cols(pd.read_csv(left_p))
        dfR = _lower_cols(pd.read_csv(right_p))

        tL = pick_time_ms(dfL)
        tR = pick_time_ms(dfR)

        # 起点差（绝对时钟）
        start_clock_delta_ms = float(tR[0] - tL[0])

        # 采样率 & 抖动
        fsL = fs_from_time(tL); fsR = fs_from_time(tR)
        jitL = jitter_std_s(tL); jitR = jitter_std_s(tR)

        # 重叠
        olap = overlap_ratio(tL, tR)

        # 互相关滞后（硬件/校正时钟域）
        mL = accel_mag(dfL); mR = accel_mag(dfR)
        lag_ms, peak = xcorr_lag_ms(tL, mL, tR, mR, max_lag_ms=300.0, grid_step_ms=10.0)

        # 漂移
        ppm = drift_ppm_linear(tL, tR)

        # 简单质量标记
        warn_fs   = not (FS_MIN <= fsL <= FS_MAX and FS_MIN <= fsR <= FS_MAX)
        warn_jit  = (jitL > JITTER_STD_MAX_S) or (jitR > JITTER_STD_MAX_S)
        warn_olap = (olap < OVERLAP_MIN)

        verdict = verdict_row(fsL, fsR, lag_ms if np.isfinite(lag_ms) else 1e9,
                              ppm, olap)

        rows.append({
            "pair_id": key,
            "left_file":  str(left_p),
            "right_file": str(right_p),
            "n_left":  len(dfL),
            "n_right": len(dfR),

            "fs_left_Hz":  round(fsL, 3),
            "fs_right_Hz": round(fsR, 3),
            "jitter_left_std_s":  round(jitL, 6),
            "jitter_right_std_s": round(jitR, 6),

            "start_clock_delta_ms": round(start_clock_delta_ms, 3),
            "dur_left_s":  round((tL[-1]-tL[0])/1000.0, 3),
            "dur_right_s": round((tR[-1]-tR[0])/1000.0, 3),
            "overlap_ratio": round(olap, 6),

            "lag_xcorr_hw_ms": round(lag_ms, 3) if np.isfinite(lag_ms) else np.nan,
            "xcorr_peak": round(peak, 6) if np.isfinite(peak) else np.nan,
            "drift_ppm":  round(ppm, 3) if np.isfinite(ppm) else np.nan,

            "verdict": verdict,
            "warn_fs":   bool(warn_fs),
            "warn_jitter": bool(warn_jit),
            "warn_overlap": bool(warn_olap),
        })

    df = pd.DataFrame(rows)
    # 便于目测：按 verdict / lag / drift 排序
    if not df.empty:
        df = df.sort_values(by=["verdict","lag_xcorr_hw_ms","drift_ppm"],
                            ascending=[True, True, True], na_position="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUT, index=False)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(30))
    print(f"✅ Pair sync report written: {OUT.resolve()}")

if __name__ == "__main__":
    main()
