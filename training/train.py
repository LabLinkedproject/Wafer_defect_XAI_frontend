"""
training/train.py
실물 웨이퍼 기준으로 가공한 WM-811K + MixedWM38 로 8종 복수 라벨 분류기를 학습한다.

실행: taskset -c 4-7 ./venv-train/bin/python -m training.train

입력은 전면 맵과, 검사 다이 44개를 1×1 ~ 6×6 구역으로 나눈 판정을 섞는다.
잡음은 표본마다 라벨과 무관하게 입힌다. 시험 지표는 training.evaluate 가 계산한다.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.model import Geometry, WaferNet, augment, full_map, quadrat_map   # noqa: E402
import training_status                                                          # noqa: E402

DATA_DIR = "datasets/wafer270"
OUT_DIR = "models"
CKPT = os.path.join(OUT_DIR, "wafernet.pt")
LOG = os.path.join(OUT_DIR, "training_log.jsonl")
MAX_NOISE = 0.22              # 전면 맵 잡음 밀도 상한 (MixedWM38 결함 맵 19 %)
QUADRATS = (1, 2, 3, 6)       # 다이 한 변을 나누는 구역 수
TRAIN, VAL, TEST = 0, 1, 2
MIXED, WM811K = 0, 1


def load_meta():
    m = np.load(f"{DATA_DIR}/meta.npz")
    with open(f"{DATA_DIR}/info.json") as f:
        info = json.load(f)
    return m["labels"].astype(np.float32), m["split"], m["source"], m["wafer"], info["patterns"]


def load_images(ids):
    """지정한 표본만 메모리에 올린다. 전체 파일을 한 번에 읽지 않는다."""
    raw = np.load(f"{DATA_DIR}/raw.npy", mmap_mode="r")
    core = np.load(f"{DATA_DIR}/core.npy", mmap_mode="r")
    return torch.from_numpy(np.array(raw[ids])), torch.from_numpy(np.array(core[ids]))


def to_float(a: torch.Tensor) -> torch.Tensor:
    return a.float()[:, None] / 255


def train_input(raw, core, y, g, random_index):
    """표본별로 입력 형태를 고른다. Random 은 구조 분리로 사라지므로 원본을 쓴다."""
    n = len(raw)
    both = augment(torch.cat([to_float(raw), to_float(core)], 1))
    r, c = both[:, :1], both[:, 1:]
    is_random = (y[:, random_index] > 0)[:, None, None, None]
    c = torch.where(is_random, r, c)
    mode = torch.rand(n)
    out = torch.zeros(n, 3, r.shape[2], r.shape[3])
    pick = mode < 0.2
    if pick.any():                                   # 전면 맵, 실제 잡음
        out[pick] = full_map(r[pick], g)
    pick = (mode >= 0.2) & (mode < 0.4)
    if pick.any():                                   # 전면 맵, 잡음을 라벨과 무관하게 다시 입힘
        k = int(pick.sum())
        out[pick] = full_map(c[pick], g, torch.empty(k).uniform_(0, MAX_NOISE) * (torch.rand(k) < 0.8))
    for i, q in enumerate(QUADRATS):                 # 검사 다이 44개, 구역 판정
        pick = (mode >= 0.4 + 0.15 * i) & (mode < 0.55 + 0.15 * i)
        if not pick.any():
            continue
        k = int(pick.sum())
        keep = torch.where(y[pick, random_index] > 0, torch.empty(k).uniform_(0.3, 1.0), torch.ones(k))
        out[pick] = quadrat_map(c[pick], g, q, light=torch.empty(k).uniform_(0.04, 0.10), heavy=torch.empty(k).uniform_(0.25, 0.45),
                                stray=torch.empty(k).uniform_(0, 0.08 / q ** 2) * (torch.rand(k) < 0.7), keep=keep)
    return out


@torch.no_grad()
def logits_of(model, make, raw, core, batch=256):
    """입력을 배치마다 만들어 메모리를 아낀다. make(raw, core) → 입력 텐서"""
    model.eval()
    return torch.cat([model(make(to_float(raw[i:i + batch]), to_float(core[i:i + batch])))
                      for i in range(0, len(raw), batch)]).numpy()


def macro_f1(prob, y, threshold=0.5):
    pred, truth = prob >= threshold, y > 0
    tp = (pred & truth).sum(0)
    fp = (pred & ~truth).sum(0)
    fn = (~pred & truth).sum(0)
    has = truth.sum(0) > 0
    return float((2 * tp / np.maximum(1, 2 * tp + fp + fn))[has].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--time-only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    labels, split, source, _, patterns = load_meta()
    random_index = patterns.index("Random")
    g = Geometry()
    tr = np.where(split == TRAIN)[0]
    va = np.where(split == VAL)[0]
    raw_tr, core_tr = load_images(tr)
    y_tr = torch.from_numpy(labels[tr])
    raw_va, core_va = load_images(va)
    y_va = labels[va]
    pos = y_tr.sum(0)
    weight = ((len(tr) - pos) / pos.clamp(min=1)).clamp(1, 20)
    model = WaferNet(len(patterns))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    batches = len(tr) // a.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * batches, pct_start=0.1)
    print("파라미터", sum(p.numel() for p in model.parameters()), "학습", len(tr), "검증", len(va),
          "pos_weight", weight.numpy().round(1).tolist(), flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    if not a.time_only:
        open(LOG, "w").close()
    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for b in range(batches):
            ids = perm[b * a.batch:(b + 1) * a.batch]
            y = y_tr[ids]
            x = train_input(raw_tr[ids], core_tr[ids], y, g, random_index)
            loss = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            if b % 25 == 0:
                training_status.report("train", ep, a.epochs, f"에폭 {ep + 1} · 배치 {b}/{batches} · 손실 {total / (b + 1):.3f}", unit="에폭")
            if a.time_only and b == 19:
                print(f"배치당 {(time.time() - t0) / 20:.3f} s → 에폭당 {(time.time() - t0) / 20 * batches / 60:.1f} 분")
                return
        prob_full = 1 / (1 + np.exp(-logits_of(model, lambda r, c: full_map(r, g), raw_va, core_va)))
        prob_die = 1 / (1 + np.exp(-logits_of(model, lambda r, c: quadrat_map(c, g, 2), raw_va, core_va)))
        row = {"epoch": ep + 1, "loss": round(total / batches, 4), "val_full_f1": round(macro_f1(prob_full, y_va), 4),
               "val_die_f1": round(macro_f1(prob_die, y_va), 4), "minutes": round((time.time() - t0) / 60, 1)}
        with open(LOG, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(row, flush=True)
        score = (row["val_full_f1"] + row["val_die_f1"]) / 2
        if score > best:
            best = score
            torch.save({"state": model.state_dict(), "patterns": patterns, "epoch": ep + 1, "seed": a.seed}, CKPT)
    print("최고 검증 점수", best, flush=True)


if __name__ == "__main__":
    main()
