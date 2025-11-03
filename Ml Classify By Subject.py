"""
按受试者分组的传统机器学习分类基线（GroupKFold）
===================================================
目录结构（已按受试者整理）：
D:/Data/Watch_Data_Subjects/
├── Tyj/
│   ├── 00b_腕旋前后 10s_20251026_170500/
│   │   └── aligned_*.csv
│   ├── 00z_手掌拍击(左10次)_20251026_171200/
│   │   └── aligned_*.csv
│   └── ...
├── Hj/
│   └── ...
└── ...

本脚本会：
1) 遍历受试者/会话，读取每个会话中的第一个 aligned_*.csv。
2) 从会话目录名提取中文动作标签（保留左右手等细粒度，如“手掌拍击(左10次)”）。
3) 仅使用 12 个数值传感器列（左右手 acc/gyr xyz）做统计特征。
4) 以“受试者”为 Group 做 GroupKFold 交叉验证（n_splits=受试者人数，若人数<3则设为min(人数,2)）。
5) 模型：RandomForestClassifier。输出每折与总体的 Accuracy、Macro-F1、分类报告、混淆矩阵图。
6) 生成 features_table.csv、feature_importance.csv。

依赖：
  pip install -U pandas numpy scipy scikit-learn matplotlib

运行：
  直接运行（使用默认 ROOT）：
    python ml_classify_by_subject.py
  或指定根目录/输出目录：
    python ml_classify_by_subject.py --root "D:/Data/Watch_Data_Subjects" --out "D:/Data/Watch_Data_Subjects/ml_outputs"
"""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================
# 配置与工具
# =====================
DEFAULT_ROOT = Path(r"D:\Data\Watch_Data_Subjects")
DEFAULT_OUT = Path(r"D:\Data\Watch_Data_Subjects\ml_outputs")

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
    """确保中文能显示：优先选择系统中常见中文字体；设置minus符号。"""
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
    """从会话目录名提取中文动作标签（保留左右与括号等）。
    规则：去掉开头采集号（2-3位字母数字）与分隔符、去掉末尾时间戳后缀。
    例："00z_手掌拍击(左10次)_20251026_171200" → "手掌拍击(左10次)"
    """
    name = session_dir_name
    name = PREFIX_ID_RE.sub("", name)
    name = TIME_TAIL_RE.sub("", name)
    return name.strip()


def find_aligned_csv(session_path: Path) -> Path | None:
    for p in sorted(session_path.glob("aligned_*.csv")):
        return p
    return None

# =====================
# 特征工程
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
    "mean": lambda x: float(np.nanmean(x)),
    "std": lambda x: float(np.nanstd(x)),
    "min": lambda x: float(np.nanmin(x)),
    "max": lambda x: float(np.nanmax(x)),
    "median": lambda x: float(np.nanmedian(x)),
    "iqr": iqr,
    "rms": lambda x: float(np.sqrt(np.nanmean(np.square(x)))),
    "energy": signal_energy,
    "skew": lambda x: float(stats.skew(x, nan_policy='omit')),
    "kurt": lambda x: float(stats.kurtosis(x, nan_policy='omit')),
    "ptp": lambda x: float(np.nanmax(x) - np.nanmin(x)),
    "zcr": zero_crossings,
    "entropy": shannon_entropy,
}


def select_sensor_columns(df: pd.DataFrame) -> List[str]:
    # 优先严格列集合；否则按正则匹配 acc_* / gyr_* 的数值列
    if all(c in df.columns for c in SENSOR_COLS):
        return SENSOR_COLS
    candidates = [c for c in df.columns if df[c].dtype.kind in "fi"]
    chosen = [c for c in candidates if re.match(r"^(acc|gyr)_[xyz]_(left|right)$", c)]
    if not chosen:
        raise ValueError("在CSV中未找到传感器列（acc_*/gyr_*）")
    return chosen


def extract_features_from_csv(csv_path: Path) -> Tuple[Dict[str, float], int]:
    # 兼容编码
    df = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except Exception:
            df = None
    if df is None:
        raise RuntimeError(f"无法读取CSV: {csv_path}")

    cols = select_sensor_columns(df)
    num_df = df[cols].copy()
    n_rows = int(num_df.shape[0])

    feats: Dict[str, float] = {}
    for col in num_df.columns:
        x = num_df[col].to_numpy()
        for fname, ffunc in BASIC_FEATURES.items():
            try:
                feats[f"{col}__{fname}"] = ffunc(x)
            except Exception:
                feats[f"{col}__{fname}"] = np.nan
    # 序列长度特征
    feats["seq_len"] = float(n_rows)
    return feats, n_rows

# =====================
# 数据集构建
# =====================

def build_dataset(root: Path) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    samples = []
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
                feats, n_rows = extract_features_from_csv(aligned)
            except Exception as e:
                print(f"[WARN] 特征提取失败: {aligned}: {e}")
                continue
            row = {
                "subject": subject,
                "session_dir": session_dir.name,
                "label": label,
                "n_rows": n_rows,
                **feats,
            }
            samples.append(row)
    if not samples:
        raise RuntimeError("未找到任何样本。请检查根目录与文件命名。")

    df = pd.DataFrame(samples)
    labels = df["label"].tolist()
    groups = df["subject"].tolist()  # 以受试者为组ID
    subjects = sorted(df["subject"].unique().tolist())
    return df, labels, groups, subjects

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
# 交叉验证与训练
# =====================

def run_cv(df: pd.DataFrame, labels: List[str], groups: List[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_cols = ["subject", "session_dir", "label", "n_rows"]
    feat_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feat_cols].to_numpy(dtype=float)
    y_text = np.array(labels)
    g = np.array(groups)

    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    classes = sorted(np.unique(y_text).tolist())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([cls_to_idx[c] for c in y_text], dtype=int)

    n_subjects = len(np.unique(g))
    n_splits = max(2, min(n_subjects, n_subjects))  # 至少2折；通常等于受试者数（LOSO）
    gkf = GroupKFold(n_splits=n_splits)

    # 模型
    clf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight='balanced_subsample',
        random_state=42,
    )

    fold_reports = []
    overall_cm = np.zeros((len(classes), len(classes)), dtype=int)
    y_true_all, y_pred_all = [], []

    # 为了让特征重要性有全局统计，后面在全量上再拟合一次
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=g), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1m = f1_score(y_te, y_pred, average='macro')
        report = classification_report(y_te, y_pred, target_names=classes, digits=4)
        fold_reports.append((fold, acc, f1m, report))

        cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))
        overall_cm += cm
        plot_confusion(cm, classes, f"Confusion Matrix - Fold {fold}", out_dir / f"confusion_matrix_fold{fold}.png")

        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

        # 保存该折真实/预测明细，便于排查
        fold_df = pd.DataFrame({
            "subject": g[test_idx],
            "true": [classes[i] for i in y_te],
            "pred": [classes[i] for i in y_pred],
            "session_dir": df.iloc[test_idx]["session_dir"].values,
        })
        fold_df.to_csv(out_dir / f"predictions_fold{fold}.csv", index=False, encoding="utf-8-sig")

    # 汇总
    acc_all = accuracy_score(y_true_all, y_pred_all)
    f1m_all = f1_score(y_true_all, y_pred_all, average='macro')
    plot_confusion(overall_cm, classes, "Confusion Matrix - Overall (sum)", out_dir / "confusion_matrix_overall.png")

    # 保存报告
    with open(out_dir / "cv_report.txt", "w", encoding="utf-8") as f:
        for (fold, acc, f1m, report) in fold_reports:
            f.write(f"===== Fold {fold} =====\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro-F1: {f1m:.4f}\n")
            f.write(report + "\n\n")
        f.write("===== Overall (concatenated predictions) =====\n")
        f.write(f"Accuracy: {acc_all:.4f}\n")
        f.write(f"Macro-F1: {f1m_all:.4f}\n")

    # 全量再拟合一次，导出特征重要性
    clf.fit(X, y)
    feat_imp = pd.DataFrame({
        "feature": feat_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    feat_imp.to_csv(out_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"[OK] CV 完成。Overall Accuracy={acc_all:.4f}, Macro-F1={f1m_all:.4f}")
    print(f"报告与图像保存在: {out_dir}")

# =====================
# 主程序
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="按受试者整理后的根目录")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出目录")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)

    print(f"[INFO] root={root}")
    df, labels, groups, subjects = build_dataset(root)

    out_dir.mkdir(parents=True, exist_ok=True)
    features_csv = out_dir / "features_table.csv"
    df.to_csv(features_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 特征表已保存: {features_csv} (shape={df.shape}, subjects={len(subjects)})")

    run_cv(df, labels, groups, out_dir=out_dir)


if __name__ == "__main__":
    main()
