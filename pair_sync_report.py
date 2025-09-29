# pair_sync_report.py
# -*- coding: utf-8 -*-
from pathlib import Path
import re, datetime
import numpy as np
import pandas as pd

# ===== 路径配置 =====
ROOT = r"D:\NEWALL"
OUT_DIR = Path(r"D:\Data\Watch_Reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATE = datetime.datetime.now().strftime("%Y%m%d")
OUT = OUT_DIR / f"pair_sync_report_{DATE}.xlsx"

# ===== 工具函数 =====
def _lower_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def _to_float(a: pd.Series) -> np.ndarray:
    return pd.to_numeric(a, errors="coerce").to_numpy(dtype="float64")

def _pick_time_ms(df: pd.DataFrame) -> np.ndarray:
    """优先硬件 ns → ms；其次 corrected_wall_clock_ms；最后 wall_clock_ms。"""
    ldf = _lower_cols(df)
    if "event_timestamp_ns" in ldf:
        t = _to_float(ldf["event_timestamp_ns"]); t = t[np.isfinite(t)]
        if t.size: return (t - t[0]) / 1e6
    if "corrected_wall_clock_ms" in ldf:
        t = _to_float(ldf["corrected_wall_clock_ms"]); t = t[np.isfinite(t)]
        if t.size: return t - t[0]
    if "wall_clock_ms" in ldf:
        t = _to_float(ldf["wall_clock_ms"]); t = t[np.isfinite(t)]
        if t.size: return t - t[0]
    # 兜底：按 100Hz 伪时间
    return np.arange(len(ldf), dtype="float64") * 10.0

def _fs_from_time_ms(t_ms: np.ndarray) -> float:
    if t_ms.size < 2: return np.nan
    dt = np.diff(t_ms) / 1000.0
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(1.0 / np.mean(dt)) if dt.size else np.nan

def _accel_mag(df: pd.DataFrame) -> np.ndarray:
    ldf = _lower_cols(df)
    ax = _to_float(ldf.get("accel_x", pd.Series([])))
    ay = _to_float(ldf.get("accel_y", pd.Series([])))
    az = _to_float(ldf.get("accel_z", pd.Series([])))
    n = min(ax.size, ay.size, az.size)
    if n == 0: return np.array([], dtype="float64")
    a = np.vstack([ax[:n], ay[:n], az[:n]]).T
    a[~np.isfinite(a)] = np.nan
    return np.nan_to_num(np.linalg.norm(a, axis=1))

def _resample_to_grid(t_ms: np.ndarray, y: np.ndarray, grid_ms: np.ndarray) -> np.ndarray:
    """线性插值到统一时间网格（ms）。"""
    if t_ms.size == 0 or y.size == 0: return np.zeros_like(grid_ms)
    # 去重
    if t_ms.size > 1:
        keep = np.r_[True, np.diff(t_ms) > 0]
        t_ms = t_ms[keep]; y = y[keep]
    return np.interp(grid_ms, t_ms, y, left=y[0], right=y[-1])

def _xcorr_lag_ms(t1_ms: np.ndarray, y1: np.ndarray,
                  t2_ms: np.ndarray, y2: np.ndarray,
                  max_lag_ms: int = 300, grid_step_ms: float = 10.0):
    """
    交叉相关估计相对延迟（ms）。>0 表示右表领先左表 lag_ms。
    """
    if t1_ms.size < 2 or t2_ms.size < 2: return np.nan, np.nan
    t_start = max(t1_ms[0], t2_ms[0])
    t_end   = min(t1_ms[-1], t2_ms[-1])
    if t_end - t_start < 5 * grid_step_ms:
        return np.nan, np.nan
    grid = np.arange(t_start, t_end + 1e-6, grid_step_ms, dtype="float64")
    y1g = _resample_to_grid(t1_ms, y1, grid)
    y2g = _resample_to_grid(t2_ms, y2, grid)
    # 标准化
    y1g = (y1g - np.mean(y1g)) / (np.std(y1g) + 1e-12)
    y2g = (y2g - np.mean(y2g)) / (np.std(y2g) + 1e-12)
    max_k = int(max_lag_ms / grid_step_ms)
    lags = range(-max_k, max_k + 1)
    corrs = []
    for k in lags:
        if k >= 0:
            c = np.dot(y1g[k:], y2g[:len(y2g)-k]) / max(1, len(y1g)-k)
        else:
            c = np.dot(y1g[:len(y1g)+k], y2g[-k:]) / max(1, len(y1g)+k)
        corrs.append(c)
    corrs = np.asarray(corrs, dtype="float64")
    k_best = int(np.argmax(corrs))
    lag_best_ms = (k_best - max_k) * grid_step_ms
    return float(lag_best_ms), float(np.max(corrs))

def _drift_ppm(tL_ms: np.ndarray, tR_ms: np.ndarray) -> float:
    """线性漂移：拟合 y=a+b*x，ppm=(b-1)*1e6。"""
    if len(tL_ms) < 10 or len(tR_ms) < 10: return np.nan
    # 下采样以稳健
    idxL = np.linspace(0, len(tL_ms)-1, num=min(2000, len(tL_ms))).astype(int)
    idxR = np.linspace(0, len(tR_ms)-1, num=min(2000, len(tR_ms))).astype(int)
    x = tL_ms[idxL]; y = tR_ms[idxR]
    m = min(len(x), len(y))
    if m < 10: return np.nan
    x = x[:m]; y = y[:m]
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        b, a = np.linalg.lstsq(A, y, rcond=None)[0]
        return float((b - 1.0) * 1e6)
    except Exception:
        return np.nan

# ===== 配对逻辑（按 prefix+日期）=====
# 形如：003_draw2_Left_20250927_144250.csv
PAT = re.compile(
    r"^(?P<prefix>.+?)_(?P<hand>Left|Right)_(?P<date>\d{8})_(?P<time>\d{6})\.csv$",
    re.IGNORECASE
)

def _parse_name(path: Path):
    m = PAT.match(path.name)
    if not m: return None
    g = m.groupdict()
    return {
        "prefix": g["prefix"],
        "hand": g["hand"].capitalize(),
        "date": g["date"],      # YYYYMMDD
        "time": g["time"],      # HHMMSS
        "path": path
    }

def discover_pairs_by_date(root: str):
    """
    按（prefix, date）分桶；同日若出现多于一侧多文件，则挑 time 最接近的一对。
    """
    files = sorted(Path(root).rglob("*.csv"))
    buckets = {}
    singles = []
    for p in files:
        info = _parse_name(p)
        if not info:
            singles.append(p); continue
        key = (info["prefix"], info["date"])
        d = buckets.setdefault(key, {"Left": [], "Right": []})
        d[info["hand"]].append(info)

    pairs = []
    for key, d in buckets.items():
        Ls, Rs = d["Left"], d["Right"]
        if not Ls and not Rs:
            continue
        if Ls and Rs:
            # 正常：各 1 个
            if len(Ls) == 1 and len(Rs) == 1:
                pairs.append((key, Ls[0]["path"], Rs[0]["path"]))
            else:
                # 罕见：多于一个，挑时间最接近的一对
                def _hms_to_sec(hms: str) -> int:
                    return int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])
                best = None
                for li in Ls:
                    tl = _hms_to_sec(li["time"])
                    for ri in Rs:
                        tr = _hms_to_sec(ri["time"])
                        diff = abs(tl - tr)
                        cand = (diff, li["path"], ri["path"])
                        if best is None or diff < best[0]:
                            best = cand
                _, lp, rp = best
                pairs.append((key, lp, rp))
                # 其余落单
                usedL = {lp}; usedR = {rp}
                for li in Ls:
                    if li["path"] not in usedL: singles.append(li["path"])
                for ri in Rs:
                    if ri["path"] not in usedR: singles.append(ri["path"])
        else:
            # 只有一侧
            for li in Ls: singles.append(li["path"])
            for ri in Rs: singles.append(ri["path"])
    return pairs, singles

# ===== 主分析 =====
def analyze_pair(key, left_path: Path, right_path: Path) -> dict:
    prefix, date = key
    d = {"pair_id": f"{prefix}_{date}", "left_file": str(left_path), "right_file": str(right_path)}
    try:
        dfL = _lower_cols(pd.read_csv(left_path))
        dfR = _lower_cols(pd.read_csv(right_path))
    except Exception as e:
        d["error"] = f"read_csv failed: {e}"
        return d

    # 时间轴与采样率
    tL = _pick_time_ms(dfL)
    tR = _pick_time_ms(dfR)
    d["n_left"]  = int(len(dfL))
    d["n_right"] = int(len(dfR))
    d["fs_left_Hz"]  = round(_fs_from_time_ms(tL), 3)
    d["fs_right_Hz"] = round(_fs_from_time_ms(tR), 3)

    # 起止与重叠
    startL, endL = float(tL[0] if len(tL) else np.nan), float(tL[-1] if len(tL) else np.nan)
    startR, endR = float(tR[0] if len(tR) else np.nan), float(tR[-1] if len(tR) else np.nan)
    d["start_delta_ms"] = round(abs(startL - startR), 3) if np.isfinite(startL) and np.isfinite(startR) else np.nan
    durL = endL - startL if np.isfinite(endL) and np.isfinite(startL) else np.nan
    durR = endR - startR if np.isfinite(endR) and np.isfinite(startR) else np.nan
    d["dur_left_s"]  = round(durL/1000.0, 3) if np.isfinite(durL) else np.nan
    d["dur_right_s"] = round(durR/1000.0, 3) if np.isfinite(durR) else np.nan
    if np.isfinite(startL) and np.isfinite(endL) and np.isfinite(startR) and np.isfinite(endR):
        inter = max(0.0, min(endL, endR) - max(startL, startR))
        union = max(endL, endR) - min(startL, startR)
        d["overlap_ratio"] = round(inter / union, 6) if union > 0 else np.nan
    else:
        d["overlap_ratio"] = np.nan

    # 基于 accel|a| 的相关滞后
    aL = _accel_mag(dfL); aR = _accel_mag(dfR)
    lag_ms, corr_max = _xcorr_lag_ms(tL, aL, tR, aR, max_lag_ms=300, grid_step_ms=10.0)
    d["lag_xcorr_ms"] = round(lag_ms, 3) if np.isfinite(lag_ms) else np.nan
    d["xcorr_peak"]   = round(corr_max, 6) if np.isfinite(corr_max) else np.nan

    # 漂移 ppm
    d["drift_ppm"] = round(_drift_ppm(tL, tR), 3)

    # 内部命中率（如有 gyro_mode）
    for side, df in (("left", dfL), ("right", dfR)):
        if "gyro_mode" in df.columns:
            m = df["gyro_mode"].astype(str).str.upper().str.strip()
            ratio = float((m == "EXACT").mean())
        else:
            ratio = np.nan
        d[f"exact_ratio_{side}"] = round(ratio, 6) if np.isfinite(ratio) else np.nan

    # 告警（可按需调）
    d["warn_fs"]   = (d["fs_left_Hz"]  < 98.8) or (d["fs_right_Hz"] < 98.8)
    d["warn_lag"]  = (abs(d["lag_xcorr_ms"]) if np.isfinite(d.get("lag_xcorr_ms", np.nan)) else 0) > 50
    d["warn_olap"] = (d["overlap_ratio"] if np.isfinite(d["overlap_ratio"]) else 0) < 0.8
    return d

def main():
    pairs, singles = discover_pairs_by_date(ROOT)
    rows = [analyze_pair(key, L, R) for key, L, R in pairs]
    df_pairs = pd.DataFrame(rows).sort_values(by=["pair_id"]).reset_index(drop=True)
    df_singles = pd.DataFrame({"unpaired_file": [str(p) for p in singles]}) if singles else pd.DataFrame()

    # 摘要
    if not df_pairs.empty:
        summary = pd.DataFrame([{
            "pairs": len(df_pairs),
            "avg_fs_left_Hz":  round(pd.to_numeric(df_pairs["fs_left_Hz"],  errors="coerce").mean(), 3),
            "avg_fs_right_Hz": round(pd.to_numeric(df_pairs["fs_right_Hz"], errors="coerce").mean(), 3),
            "avg_overlap":     round(pd.to_numeric(df_pairs["overlap_ratio"], errors="coerce").mean(), 6),
            "avg_abs_lag_ms":  round(pd.to_numeric(df_pairs["lag_xcorr_ms"],  errors="coerce").abs().mean(), 3),
            "avg_drift_ppm":   round(pd.to_numeric(df_pairs["drift_ppm"],    errors="coerce").mean(), 3),
            "warn_fs_cnt": int(df_pairs["warn_fs"].sum()),
            "warn_lag_cnt": int(df_pairs["warn_lag"].sum()),
            "warn_overlap_cnt": int(df_pairs["warn_olap"].sum()),
        }])
    else:
        summary = pd.DataFrame([{"pairs": 0}])

    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="summary", index=False)
        if not df_pairs.empty:
            df_pairs.to_excel(xw, sheet_name="pairs", index=False)
        if not df_singles.empty:
            df_singles.to_excel(xw, sheet_name="singles", index=False)

    print(f"✅ 同步报告已写入: {OUT.resolve()}")
    if not df_pairs.empty:
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df_pairs.head(20))

if __name__ == "__main__":
    main()
