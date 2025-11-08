# ml_classify_by_subject_windows.py
# ------------------------------------------------------------
# 快速增强版传统 ML：滑窗 + 标准化 + 组感知搜参（内层CV）+ 外层GroupKFold评估
# 目录结构/中文标签解析/受试者分组 与你的原脚本一致；
# 差异：每会话不再“整段取特征”，而是“滑窗提特征→每窗为一个样本”，标签继承会话标签。
# ------------------------------------------------------------
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================
# 配置与工具
# =====================
DEFAULT_ROOT = Path(r"D:\Data\Watch_Data_Subjects")
DEFAULT_OUT  = Path(r"D:\Data\Watch_Data_Subjects\ml_outputs_windows")

# 仅使用的 12 个数值列（若存在则优先按此顺序取；若缺失则从数值列中自动匹配 acc_*/gyr_*）
SENSOR_COLS = [
    "acc_x_left", "acc_y_left", "acc_z_left",
    "gyr_x_left", "gyr_y_left", "gyr_z_left",
    "acc_x_right", "acc_y_right", "acc_z_right",
    "gyr_x_right", "gyr_y_right", "gyr_z_right",
]

TIME_TAIL_RE = re.compile(r"_20\d{6}_\d{6}$")
PREFIX_ID_RE = re.compile(r"^[0-9a-z]{2,3}[_ ]")  # 采集号前缀，如 "00b_" 或 "022 "

# ---- 字体与中文显示 ----
def _ensure_chinese_font() -> str:
    preferred = [
        "Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "MS YaHei",
        "Sarasa UI SC", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei"
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            mpl.rcParams['font.family'] = name
            mpl.rcParams['axes.unicode_minus'] = False
            print(f"[INFO] Using Chinese font: {name}")
            break
    else:
        mpl.rcParams['axes.unicode_minus'] = False
        print("[WARN] 未找到常见中文字体，可能出现方块字。建议安装 'Microsoft YaHei' 或 'SimHei'.")

_ensure_chinese_font()

def parse_label_from_session_dir(session_dir_name: str) -> str:
    """从会话目录名提取中文动作标签（保留左右与括号等）。"""
    name = session_dir_name
    name = PREFIX_ID_RE.sub("", name)
    name = TIME_TAIL_RE.sub("", name)
    return name.strip()

def find_aligned_csv(session_path: Path) -> Path | None:
    for p in sorted(session_path.glob("aligned_*.csv")):
        return p
    return None

# =====================
# 特征工程（与原版风格一致，但用于滑窗）
# =====================
def zero_crossings(x: np.ndarray) -> int:
    x = np.asarray(x)
    return int(((x[:-1] * x[1:]) < 0).sum())

def signal_energy(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.nansum(x ** 2))

def iqr(x: np.ndarray) -> float:
    return float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))

def shannon_entropy(x: np.ndarray, bins: int = 30) -> float:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / np.sum(hist) if np.sum(hist) > 0 else np.zeros_like(hist)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) if p.size > 0 else 0.0

BASIC_FEATURES = {
    "mean":   lambda x: float(np.nanmean(x)),
    "std":    lambda x: float(np.nanstd(x)),
    "min":    lambda x: float(np.nanmin(x)),
    "max":    lambda x: float(np.nanmax(x)),
    "median": lambda x: float(np.nanmedian(x)),
    "iqr":    iqr,
    "rms":    lambda x: float(np.sqrt(np.nanmean(np.square(x)))),
    "energy": signal_energy,
    "skew":   lambda x: float(stats.skew(x, nan_policy='omit')),
    "kurt":   lambda x: float(stats.kurtosis(x, nan_policy='omit')),
    "ptp":    lambda x: float(np.nanmax(x) - np.nanmin(x)),
    "zcr":    zero_crossings,
    "entropy": shannon_entropy,
}

def select_sensor_columns(df: pd.DataFrame) -> List[str]:
    if all(c in df.columns for c in SENSOR_COLS):
        return SENSOR_COLS
    candidates = [c for c in df.columns if df[c].dtype.kind in "fi"]
    chosen = [c for c in candidates if re.match(r"^(acc|gyr)_[xyz]_(left|right)$", c)]
    if not chosen:
        raise ValueError("在CSV中未找到传感器列（acc_*/gyr_*）")
    return chosen

def extract_window_features(seg: np.ndarray, col_names: List[str]) -> Dict[str, float]:
    # seg: (win_len, n_cols)
    feats: Dict[str, float] = {}
    for j, col in enumerate(col_names):
        x = seg[:, j]
        for fname, ffunc in BASIC_FEATURES.items():
            try:
                feats[f"{col}__{fname}"] = ffunc(x)
            except Exception:
                feats[f"{col}__{fname}"] = np.nan
    feats["seq_len"] = float(seg.shape[0])
    return feats

# =====================
# 数据构建（滑窗）
# =====================
def read_csv_robust(csv_path: Path) -> pd.DataFrame:
    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"无法读取CSV: {csv_path}")
    return df

def build_window_dataset(
    root: Path,
    win_len: int = 256,
    step_len: int = 64,
    max_windows_per_session: int | None = None
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    返回：
      df_win: 每行=一个滑窗样本（含特征与元信息）
      labels: 文本标签（按窗口继承会话标签）
      groups: 分组ID（受试者），与labels同长度
      subjects: 受试者列表
    """
    samples = []
    total_windows = 0

    for subj_dir in sorted(root.iterdir()):
        if not subj_dir.is_dir():
            continue
        subject = subj_dir.name

        for session_dir in sorted(subj_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            label = parse_label_from_session_dir(session_dir.name)
            aligned = find_aligned_csv(session_dir)
            if aligned is None:
                continue

            try:
                df = read_csv_robust(aligned)
                cols = select_sensor_columns(df)
                num_df = df[cols].copy()
                N = len(num_df)
                if N < win_len:
                    continue  # 太短，跳过
                # 生成滑窗索引
                starts = list(range(0, N - win_len + 1, step_len))
                if max_windows_per_session is not None and len(starts) > max_windows_per_session:
                    starts = starts[:max_windows_per_session]

                for w_idx, s in enumerate(starts):
                    e = s + win_len
                    seg = num_df.iloc[s:e].to_numpy(dtype=float)
                    feats = extract_window_features(seg, cols)
                    row = {
                        "subject": subject,
                        "session_dir": session_dir.name,
                        "label": label,
                        "win_start": int(s),
                        "win_end": int(e),
                        "win_idx": int(w_idx),
                        **feats,
                    }
                    samples.append(row)
                total_windows += len(starts)

            except Exception as ex:
                print(f"[WARN] 会话处理失败: {aligned}: {ex}")
                continue

    if not samples:
        raise RuntimeError("未找到任何滑窗样本。请检查根目录/窗口长度等参数。")

    df_win = pd.DataFrame(samples)
    labels  = df_win["label"].tolist()
    groups  = df_win["subject"].tolist()
    subjects = sorted(df_win["subject"].unique().tolist())
    print(f"[INFO] 共生成窗口样本: {total_windows}（subjects={len(subjects)}）")
    return df_win, labels, groups, subjects

# =====================
# 可视化：混淆矩阵
# =====================
def plot_confusion(cm: np.ndarray, classes: List[str], title: str, save_path: Path):
    plt.figure(figsize=(7.2, 6.0))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches='tight')
    plt.close()

# =====================
# 训练与交叉验证（外层评估 + 内层搜参）
# =====================
def run_nested_cv_and_eval(
    df: pd.DataFrame,
    labels: List[str],
    groups: List[str],
    out_dir: Path,
    n_iter_search: int = 40,
    random_state: int = 42
):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = ["subject", "session_dir", "label", "win_start", "win_end", "win_idx"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].to_numpy(dtype=float)
    y_text = np.array(labels)
    g_all = np.array(groups)

    # 类别索引映射
    classes = sorted(np.unique(y_text).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[c] for c in y_text], dtype=int)

    # 基础预处理 + 模型（放在Pipeline里避免数据泄露）
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy='median')),
        ("scaler",  StandardScaler(with_mean=True, with_std=True)),
        ("svc",     SVC(class_weight="balanced"))
    ])

    # 搜参空间（对 RBF SVM）
    param_dist = {
        "svc__C":     np.logspace(-2, 3, 30),
        "svc__gamma": np.logspace(-4, 1, 30),
        "svc__kernel": ["rbf"],
        "svc__probability": [False],
    }

    # 外层 GroupKFold（通常等于受试者数；至少2折）
    n_subjects = len(np.unique(g_all))
    n_splits_outer = max(2, n_subjects)
    outer = GroupKFold(n_splits=n_splits_outer)

    overall_cm = np.zeros((len(classes), len(classes)), dtype=int)
    y_true_all, y_pred_all = [], []

    fold_reports = []
    best_params_per_fold = []

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups=g_all), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr, g_te = g_all[tr_idx], g_all[te_idx]

        # 内层：在训练受试者上做组感知搜参
        n_train_groups = len(np.unique(g_tr))
        inner_splits = max(2, min(5, n_train_groups))
        inner = GroupKFold(n_splits=inner_splits)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            cv=inner,
            n_jobs=-1,
            scoring="f1_macro",
            random_state=random_state,
            refit=True,
            verbose=0
        )
        search.fit(X_tr, y_tr, **{"groups": g_tr})
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv = search.best_score_
        best_params_per_fold.append((fold, best_params, best_cv))
        print(f"[INFO] Fold {fold}: best f1_macro (innerCV) = {best_cv:.4f}, params={best_params}")

        # 外层测试
        y_pred = best_model.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        f1m  = f1_score(y_te, y_pred, average='macro')
        report = classification_report(y_te, y_pred, target_names=classes, digits=4)

        cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))
        overall_cm += cm
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

        # 输出每折混淆矩阵、预测明细
        plot_confusion(cm, classes, f"Confusion Matrix - Fold {fold}", out_dir / f"confusion_matrix_fold{fold}.png")
        fold_df = pd.DataFrame({
            "subject": g_te,
            "true": [classes[i] for i in y_te],
            "pred": [classes[i] for i in y_pred],
            "session_dir": df.iloc[te_idx]["session_dir"].values,
            "win_start": df.iloc[te_idx]["win_start"].values,
            "win_end":   df.iloc[te_idx]["win_end"].values,
        })
        fold_df.to_csv(out_dir / f"predictions_fold{fold}.csv", index=False, encoding="utf-8-sig")

        fold_reports.append((fold, acc, f1m, report))

    # 汇总
    acc_all = accuracy_score(y_true_all, y_pred_all)
    f1m_all = f1_score(y_true_all, y_pred_all, average='macro')
    plot_confusion(overall_cm, classes, "Confusion Matrix - Overall (sum of folds)", out_dir / "confusion_matrix_overall.png")

    # 保存报告
    with open(out_dir / "cv_report.txt", "w", encoding="utf-8") as f:
        for (fold, acc, f1m, report) in fold_reports:
            f.write(f"===== Fold {fold} =====\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro-F1: {f1m:.4f}\n")
            f.write(report + "\n\n")

        f.write("===== Overall (concatenated predictions) =====\n")
        f.write(f"Accuracy: {acc_all:.4f}\n")
        f.write(f"Macro-F1: {f1m_all:.4f}\n\n")

        f.write("===== InnerCV Best per Fold (f1_macro) =====\n")
        for fold, params, bestcv in best_params_per_fold:
            f.write(f"Fold {fold}: best_f1_macro={bestcv:.4f}, params={params}\n")

    print(f"[OK] CV 完成。Overall Accuracy={acc_all:.4f}, Macro-F1={f1m_all:.4f}")
    print(f"[INFO] 报告与图像保存在: {out_dir}")

# =====================
# 主程序
# =====================
def main():
    parser = argparse.ArgumentParser(description="滑窗 + 标准化 + 组感知搜参（SVC） + GroupKFold 评估")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="按受试者整理后的根目录")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出目录")
    # 窗口参数（以“采样点”为单位；若需按秒，请自己换算为样本点）
    parser.add_argument("--win_len", type=int, default=256, help="滑窗长度（样本点数）")
    parser.add_argument("--step_len", type=int, default=64, help="滑窗步长（样本点数）")
    parser.add_argument("--max_windows_per_session", type=int, default=None, help="每会话最多取多少窗（None为不限制）")
    # 搜参强度
    parser.add_argument("--n_iter_search", type=int, default=40, help="随机搜参迭代次数")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] root={root}")
    df_win, labels, groups, subjects = build_window_dataset(
        root=root,
        win_len=args.win_len,
        step_len=args.step_len,
        max_windows_per_session=args.max_windows_per_session
    )

    # 保存窗口级特征表（可很大，注意体量）
    features_csv = out_dir / "windows_features_table.csv"
    df_win.to_csv(features_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 窗口特征表已保存: {features_csv} (shape={df_win.shape}, subjects={len(subjects)})")

    run_nested_cv_and_eval(
        df=df_win,
        labels=labels,
        groups=groups,
        out_dir=out_dir,
        n_iter_search=args.n_iter_search,
        random_state=42
    )

if __name__ == "__main__":
    main()
