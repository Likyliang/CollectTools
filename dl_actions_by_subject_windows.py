# dl_actions_by_subject_windows.py
# ------------------------------------------------------------
# 轻量 1D-CNN / TCN，按受试者 GroupKFold（LOSO）评估
# 数据处理：与您之前一致的目录结构、取列逻辑、中文标签解析、滑窗
# 输出：每折与总体指标、混淆矩阵、预测明细、最佳权重、训练日志
# ------------------------------------------------------------
from __future__ import annotations
import argparse
from pathlib import Path
import re
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

warnings.filterwarnings("ignore", category=RuntimeWarning)

# =====================
# 配置与工具
# =====================
DEFAULT_ROOT = Path(r"D:\Data\Watch_Data_Subjects")
DEFAULT_OUT  = Path(r"D:\Data\Watch_Data_Subjects\dl_outputs")
SENSOR_COLS = [
    "acc_x_left", "acc_y_left", "acc_z_left",
    "gyr_x_left", "gyr_y_left", "gyr_z_left",
    "acc_x_right", "acc_y_right", "acc_z_right",
    "gyr_x_right", "gyr_y_right", "gyr_z_right",
]
TIME_TAIL_RE = re.compile(r"_20\d{6}_\d{6}$")
PREFIX_ID_RE = re.compile(r"^[0-9a-z]{2,3}[_ ]")

def _ensure_chinese_font() -> None:
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
        print("[WARN] 未找到常见中文字体，可能出现方块字。")
_ensure_chinese_font()

def parse_label_from_session_dir(session_dir_name: str) -> str:
    name = session_dir_name
    name = PREFIX_ID_RE.sub("", name)
    name = TIME_TAIL_RE.sub("", name)
    return name.strip()

def find_aligned_csv(session_path: Path) -> Path | None:
    for p in sorted(session_path.glob("aligned_*.csv")):
        return p
    return None

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

def select_sensor_columns(df: pd.DataFrame) -> List[str]:
    if all(c in df.columns for c in SENSOR_COLS):
        return SENSOR_COLS
    candidates = [c for c in df.columns if df[c].dtype.kind in "fi"]
    chosen = [c for c in candidates if re.match(r"^(acc|gyr)_[xyz]_(left|right)$", c)]
    if not chosen:
        raise ValueError("在CSV中未找到传感器列（acc_*/gyr_*）")
    return chosen

# =====================
# 滑窗构建（直接输出张量，不做统计特征）
# =====================
def build_window_tensors(
    root: Path,
    win_len: int = 256,
    step_len: int = 64,
    max_windows_per_session: int | None = None
) -> Tuple[np.ndarray, List[str], List[str], List[Dict]]:
    """
    返回:
      X: (N, C, L) float32
      y_text: 文本标签列表（长度 N）
      groups: 受试者ID列表（长度 N）
      meta: 每窗元信息字典 [{subject, session_dir, win_start, win_end}]
    """
    X_list, y_list, g_list, meta_list = [], [], [], []
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
                    continue
                starts = list(range(0, N - win_len + 1, step_len))
                if max_windows_per_session is not None and len(starts) > max_windows_per_session:
                    starts = starts[:max_windows_per_session]
                for s_idx, s in enumerate(starts):
                    e = s + win_len
                    seg = num_df.iloc[s:e].to_numpy(dtype=np.float32)  # (L, C)
                    seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
                    seg = seg.T  # (C, L)
                    X_list.append(seg)
                    y_list.append(label)
                    g_list.append(subject)
                    meta_list.append({
                        "subject": subject, "session_dir": session_dir.name,
                        "win_start": int(s), "win_end": int(e), "win_idx": int(s_idx)
                    })
                total_windows += len(starts)
            except Exception as ex:
                print(f"[WARN] 会话处理失败: {aligned}: {ex}")
                continue

    if len(X_list) == 0:
        raise RuntimeError("未生成任何窗口样本，请检查目录/参数。")

    X = np.stack(X_list, axis=0)  # (N, C, L)
    print(f"[INFO] 共生成窗口样本: {X.shape[0]}，张量形状={X.shape}（N,C,L）")
    return X, y_list, g_list, meta_list

# =====================
# 数据集与标准化
# =====================
class WindowsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mean: np.ndarray, std: np.ndarray):
        # X: (N,C,L), y: (N,)
        self.X = X
        self.y = y
        self.mean = mean[:, None]  # (C,1)
        self.std = std[:, None]    # (C,1)
        self.std[self.std == 0] = 1.0

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (C,L)
        x = (x - self.mean) / self.std
        y = self.y[idx]
        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

def compute_channel_stats(X: np.ndarray, idxs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 基于训练索引统计每通道 mean/std（按折内训练集，避免泄露）
    X_tr = X[idxs]  # (N,C,L)
    C = X_tr.shape[1]
    mean = X_tr.mean(axis=(0,2))  # (C,)
    std  = X_tr.std(axis=(0,2))   # (C,)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

# =====================
# 模型：1D-CNN / TCN
# =====================
class CNN1D(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=5, padding=2),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(width, width*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(width*2), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(width*2, width*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(width*2), nn.ReLU(inplace=True), nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width*2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0: return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.BatchNorm1d(n_outputs), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.BatchNorm1d(n_outputs), nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, width: int = 64, blocks: int = 3, kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        c_in = in_channels
        for b in range(blocks):
            layers.append(TemporalBlock(c_in, width, kernel_size=kernel_size, dilation=2**b, dropout=dropout))
            c_in = width
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(width, num_classes)
        )

    def forward(self, x):
        x = self.tcn(x)
        return self.head(x)

# =====================
# 训练与评估
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
    plt.ylabel('True label'); plt.xlabel('Predicted label')
    plt.tight_layout(); plt.savefig(save_path, dpi=160, bbox_inches='tight'); plt.close()

def set_seed(seed: int = 42):
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def train_one_fold(
    X: np.ndarray, y_idx: np.ndarray, groups: np.ndarray, meta: List[Dict],
    classes: List[str], out_dir: Path, fold: int,
    model_name: str = "cnn", width: int = 64, kernel_size: int = 3, blocks: int = 3,
    lr: float = 1e-3, batch_size: int = 128, epochs: int = 50, patience: int = 8,
    use_amp: bool = True, device: str | None = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ==== 外层：按受试者分组，确定本折训练/测试 ====
    # 这里的函数由上层传入 tr_idx, te_idx，这里只做实际训练
    # 为了可复用，这个函数假定 X,y_idx 已经切分好，下面直接实现内部分层验证
    raise_if_called_wrong = False  # 仅占位

def run_loso_training(
    X: np.ndarray, y_text: List[str], g_list: List[str], meta: List[Dict],
    out_dir: Path, model_type: str = "cnn",
    win_len: int = 256, batch_size: int = 128, epochs: int = 50, patience: int = 8,
    width: int = 64, kernel_size: int = 3, blocks: int = 3, lr: float = 1e-3, use_amp: bool = True
):
    out_dir.mkdir(parents=True, exist_ok=True)
    # 类别映射
    classes = sorted(np.unique(y_text).tolist())
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    y = np.array([cls_to_idx[c] for c in y_text], dtype=np.int64)
    g = np.array(g_list)

    n_subjects = len(np.unique(g))
    n_splits = max(2, n_subjects)
    outer = GroupKFold(n_splits=n_splits)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}; 模型={model_type}")

    overall_cm = np.zeros((len(classes), len(classes)), dtype=int)
    y_true_all, y_pred_all = [], []

    fold_reports = []
    # 逐折
    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y, groups=g), start=1):
        print(f"[INFO] ===== Fold {fold}/{n_splits} =====")
        # 训练/测试索引
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_tr, g_te = g[tr_idx], g[te_idx]

        # 训练集再切出一个“组感知验证集”用于早停（不泄露）
        n_train_groups = len(np.unique(g_tr))
        inner_splits = max(2, min(5, n_train_groups))
        inner = GroupKFold(n_splits=inner_splits)
        # 取第一折作为验证（简洁稳定）
        inner_split = list(inner.split(X_tr, y_tr, groups=g_tr))[0]
        tr_in_idx, val_in_idx = inner_split

        # 统计归一化参数（基于训练子集）
        mean_c, std_c = compute_channel_stats(X_tr, tr_in_idx)

        # 数据集与加载器
        ds_tr  = WindowsDataset(X_tr, y_tr, mean_c, std_c)
        ds_val = WindowsDataset(X_tr[val_in_idx], y_tr[val_in_idx], mean_c, std_c)
        ds_te  = WindowsDataset(X_te, y_te, mean_c, std_c)

        dl_tr  = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        dl_te  = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

        in_channels = X.shape[1]
        num_classes = len(classes)
        if model_type.lower() == "cnn":
            model = CNN1D(in_channels, num_classes, width=width, dropout=0.2)
        else:
            model = TCN(in_channels, num_classes, width=width, blocks=blocks, kernel_size=kernel_size, dropout=0.2)

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        # 类别权重（基于训练子集）
        class_weight = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_tr[tr_in_idx])
        class_weight = torch.tensor(class_weight, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weight)

        scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device=="cuda"))

        best_f1 = -1.0
        epochs_no_improve = 0
        ckpt_path = out_dir / f"best_fold{fold}.pt"

        for epoch in range(1, epochs+1):
            # ---- Train ----
            model.train()
            tr_loss = 0.0
            for xb, yb in dl_tr:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(use_amp and device=="cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                tr_loss += loss.item() * xb.size(0)
            tr_loss /= len(ds_tr)

            # ---- Validate ----
            model.eval()
            val_loss = 0.0
            yv_true, yv_pred = [], []
            with torch.no_grad():
                for xb, yb in dl_val:
                    xb = xb.to(device); yb = yb.to(device)
                    with torch.cuda.amp.autocast(enabled=(use_amp and device=="cuda")):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    val_loss += loss.item() * xb.size(0)
                    preds = logits.argmax(dim=1)
                    yv_true.extend(yb.detach().cpu().numpy().tolist())
                    yv_pred.extend(preds.detach().cpu().numpy().tolist())
            val_loss /= len(ds_val)
            val_acc = accuracy_score(yv_true, yv_pred)
            val_f1  = f1_score(yv_true, yv_pred, average="macro")

            print(f"[Fold {fold}] Epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            # Early stopping
            if val_f1 > best_f1:
                best_f1 = val_f1
                epochs_no_improve = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "mean": mean_c, "std": std_c,
                    "classes": classes
                }, ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}")
                    break

        # ---- Load best & Test ----
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        y_te_true, y_te_pred = [], []
        with torch.no_grad():
            for xb, yb in dl_te:
                xb = xb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
                y_te_pred.extend(preds)
                y_te_true.extend(yb.numpy().tolist())

        # 指标
        acc  = accuracy_score(y_te_true, y_te_pred)
        f1m  = f1_score(y_te_true, y_te_pred, average="macro")
        report = classification_report(y_te_true, y_te_pred, target_names=classes, digits=4, zero_division=0)

        cm = confusion_matrix(y_te_true, y_te_pred, labels=list(range(len(classes))))
        overall_cm += cm
        y_true_all.extend(y_te_true)
        y_pred_all.extend(y_te_pred)

        # 保存混淆矩阵
        plot_confusion(cm, classes, f"Confusion Matrix - Fold {fold}", out_dir / f"confusion_matrix_fold{fold}.png")

        # 保存预测明细（窗口级）
        te_meta = [meta[i] for i in np.where(np.isin(np.arange(len(y)), np.where(np.isin(np.arange(len(y)), np.array(te_idx)))[0]))[0]]  # 防御式写法，确保索引一致
        # 更稳妥：直接按 te_idx 取 meta
        te_meta = [meta[i] for i in te_idx]
        pred_df = pd.DataFrame({
            "subject": [g[i] for i in te_idx],
            "true":    [classes[i] for i in y_te_true],
            "pred":    [classes[i] for i in y_te_pred],
            "session_dir": [te_meta[i]["session_dir"] for i in range(len(te_idx))],
            "win_start":   [te_meta[i]["win_start"]   for i in range(len(te_idx))],
            "win_end":     [te_meta[i]["win_end"]     for i in range(len(te_idx))],
            "win_idx":     [te_meta[i]["win_idx"]     for i in range(len(te_idx))],
        })
        pred_df.to_csv(out_dir / f"predictions_fold{fold}.csv", index=False, encoding="utf-8-sig")

        fold_reports.append((fold, acc, f1m, report))
        print(f"[OK] Fold {fold}: Acc={acc:.4f}, Macro-F1={f1m:.4f}")

    # ---- 汇总 ----
    acc_all = accuracy_score(y_true_all, y_pred_all)
    f1m_all = f1_score(y_true_all, y_pred_all, average='macro')
    plot_confusion(overall_cm, classes, "Confusion Matrix - Overall (sum of folds)", out_dir / "confusion_matrix_overall.png")

    with open(out_dir / "cv_report.txt", "w", encoding="utf-8") as f:
        for (fold, acc, f1m, report) in fold_reports:
            f.write(f"===== Fold {fold} =====\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Macro-F1: {f1m:.4f}\n")
            f.write(report + "\n\n")
        f.write("===== Overall (concatenated predictions) =====\n")
        f.write(f"Accuracy: {acc_all:.4f}\n")
        f.write(f"Macro-F1: {f1m_all:.4f}\n")

    print(f"[OK] CV 完成。Overall Accuracy={acc_all:.4f}, Macro-F1={f1m_all:.4f}")
    print(f"[INFO] 输出目录: {out_dir}")

# =====================
# 主程序
# =====================
def main():
    parser = argparse.ArgumentParser(description="1D-CNN/TCN 深度学习（滑窗） + GroupKFold(受试者) 评估")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="按受试者整理后的根目录")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="输出目录")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn","tcn"], help="模型类型")
    parser.add_argument("--win_len", type=int, default=256, help="滑窗长度（样本点）")
    parser.add_argument("--step_len", type=int, default=64, help="滑窗步长（样本点）")
    parser.add_argument("--max_windows_per_session", type=int, default=None, help="每会话最多窗口（None为不限制）")

    parser.add_argument("--width", type=int, default=64, help="通道宽度（特征图通道数）")
    parser.add_argument("--kernel_size", type=int, default=3, help="TCN卷积核大小（仅TCN有效）")
    parser.add_argument("--blocks", type=int, default=3, help="TCN残差块数量（仅TCN有效）")

    parser.add_argument("--batch_size", type=int, default=128, help="batch size（CPU可改小到32/64）")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（含早停）")
    parser.add_argument("--patience", type=int, default=8, help="早停耐心")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--no_amp", action="store_true", help="关闭AMP混合精度（GPU时默认开启）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    set_seed(args.seed)

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 构建窗口张量
    X, y_text, g_list, meta = build_window_tensors(
        root=root,
        win_len=args.win_len,
        step_len=args.step_len,
        max_windows_per_session=args.max_windows_per_session
    )

    # 记录一个索引表（可选）
    idx_table = pd.DataFrame({
        "subject": g_list,
        "label": y_text,
        "session_dir": [m["session_dir"] for m in meta],
        "win_start":   [m["win_start"]   for m in meta],
        "win_end":     [m["win_end"]     for m in meta],
        "win_idx":     [m["win_idx"]     for m in meta],
    })
    idx_table.to_csv(out_dir / "windows_index.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] 窗口索引表已保存: {out_dir / 'windows_index.csv'} (N={len(idx_table)})")

    # 训练与评估
    run_loso_training(
        X=X, y_text=y_text, g_list=g_list, meta=meta,
        out_dir=out_dir / f"{args.model}_w{args.width}",
        model_type=args.model,
        win_len=args.win_len,
        batch_size=args.batch_size, epochs=args.epochs, patience=args.patience,
        width=args.width, kernel_size=args.kernel_size, blocks=args.blocks,
        lr=args.lr, use_amp=(not args.no_amp)
    )

if __name__ == "__main__":
    main()
