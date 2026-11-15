"""冠军配方 5 折 OOF 补全（fold1）：11m 检测器 + 折专属 refiner + 两段式评测，一 kernel 一折。

论文严谨性要求：refiner_v2 在 folds1-4 是训练集内数据，直接用会虚高两段式 OOF——
本 kernel 每折重训「检测器（det_v4 recipe）+ refiner（ref_v2 recipe）」，全部排除本折。
成功产出：fold1 两段式 PQ（对照 fold0 冠军 0.3661）+ 原始框存档。
"""
import json, os, random, shutil, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_util
from torch.utils.data import Dataset, DataLoader

H = W = 2048
FOLD = 1
SEED = 42
DEVICE = "cuda"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")

# ================= 公共数据 =================
train_images_dir = next(INPUT_ROOT.rglob("train_images"))
ann_json = next(INPUT_ROOT.rglob("*Annotations*.json"))
coco = json.load(open(ann_json))
ann_by_img = defaultdict(list)
for a in coco["annotations"]:
    ann_by_img[a["image_id"]].append(a)
records = defaultdict(dict)
for im in coco["images"]:
    stem = os.path.splitext(im["file_name"])[0]
    annotator, _ = im["id"].split("-", 1)
    records[stem].setdefault(annotator, []).extend(ann_by_img.get(im["id"], []))

def semantic_mask(ann_list):
    rles = []
    for a in ann_list:
        rles.extend(mask_util.frPyObjects(a["segmentation"], H, W))
    return mask_util.decode(mask_util.merge(rles))

medoid_ann = {}
for stem, by_ann in records.items():
    if len(by_ann) == 1:
        medoid_ann[stem] = next(iter(by_ann)); continue
    masks = {k: semantic_mask(v) for k, v in by_ann.items()}
    keys = list(masks)
    def _iou(m1, m2):
        u = int(m1.sum()) + int(m2.sum())
        return 1.0 if u == 0 else float((m1 & m2).sum()) / u
    medoid_ann[stem] = max(keys, key=lambda k: np.mean([_iou(masks[k], masks[j]) for j in keys if j != k]))

stems_all = sorted(records)
days = sorted({s[:8] for s in stems_all})
rng = random.Random(SEED); rng.shuffle(days)
day_fold = {d: i % 5 for i, d in enumerate(days)}
fold_of = {s: day_fold[s[:8]] for s in stems_all}
train_stems = [s for s in stems_all if fold_of[s] != FOLD]
val_stems = [s for s in stems_all if fold_of[s] == FOLD]
print(f"[fold{FOLD}] train={len(train_stems)} val={len(val_stems)}", flush=True)

# ================= Stage 1：检测器（det_v4 recipe） =================
YOLO_DIR = Path("/tmp/yolo")
for split in ("train", "val"):
    (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
for split, split_stems in (("train", train_stems), ("val", val_stems)):
    for s in split_stems:
        src_img = Path(train_images_dir) / f"{s}.jpeg"
        dst_img = YOLO_DIR / "images" / split / f"{s}.jpeg"
        if not dst_img.exists():
            try:
                os.symlink(src_img, dst_img)
            except OSError:
                shutil.copy(src_img, dst_img)
        lines = []
        for a in records[s][medoid_ann[s]]:
            for poly in a["segmentation"]:
                if len(poly) < 6:
                    continue
                norm = " ".join(f"{(v / W) if i % 2 == 0 else (v / H):.4f}" for i, v in enumerate(poly))
                lines.append(f"0 {norm}")
        (YOLO_DIR / "labels" / split / f"{s}.txt").write_text("\n".join(lines))
data_yaml = YOLO_DIR / "data.yaml"
data_yaml.write_text(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['filament']\n")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
from ultralytics import YOLO

det = YOLO("yolo11m-seg.pt")
det.train(data=str(data_yaml), epochs=50, imgsz=1536, batch=2, device=0, patience=15,
          workers=2, cache=False, seed=42 + FOLD, project=str(WORK / "runs"),
          name=f"det4oof_f{FOLD}", exist_ok=True,
          hsv_v=0.25, degrees=0.0, translate=0.05, scale=0.3, fliplr=0.5,
          flipud=0.5, mosaic=0.25, close_mosaic=5)
det = YOLO(str(WORK / "runs" / f"det4oof_f{FOLD}" / "weights" / "best.pt"))
shutil.copy(WORK / "runs" / f"det4oof_f{FOLD}" / "weights" / "best.pt", WORK / f"det4oof_f{FOLD}_best.pt")
print("[det] done", flush=True)

# ================= Stage 2：折专属 refiner（ref_v2 recipe） =================
CROP, CONTEXT, MIN_SIDE, MIN_AREA = 256, 1.5, 96, 96
NEG_FRAC, EPOCHS = 0.25, 10
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
CLAHE = cv2.createCLAHE(2.0, (8, 8))

GRAY_CACHE = {}
def gray_of(stem):
    if stem not in GRAY_CACHE:
        GRAY_CACHE[stem] = cv2.imread(str(Path(train_images_dir) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if len(GRAY_CACHE) > 300:
            GRAY_CACHE.pop(next(iter(GRAY_CACHE)))
    return GRAY_CACHE[stem]

def make_crop(stem, cx, cy, side, box=None):
    side = max(int(side), MIN_SIDE)
    x0, y0 = int(cx - side // 2), int(cy - side // 2)
    g = gray_of(stem)
    patch = np.zeros((side, side), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
    patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = g[ys0:ys1, xs0:xs1]
    if box is None:
        box = (x0, y0, x0 + side, y0 + side)
    seed = np.zeros((side, side), np.float32)
    bx0 = max(xs0, int(box[0])); by0 = max(ys0, int(box[1]))
    bx1 = min(xs1, int(box[2])); by1 = min(ys1, int(box[3]))
    if bx1 > bx0 and by1 > by0:
        seed[by0 - y0:by1 - y0, bx0 - x0:bx1 - x0] = 1.0
    x = np.stack([patch.astype(np.float32) / 255.0, CLAHE.apply(patch).astype(np.float32) / 255.0, seed], 0)
    return x, (x0, y0, side)

def crop_target(mask_full, x0, y0, side):
    t = np.zeros((side, side), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
    t[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = mask_full[ys0:ys1, xs0:xs1]
    return t

def jitter(cx, cy, side):
    side = side * random.uniform(0.80, 1.30)
    cx += random.uniform(-0.20, 0.20) * side
    cy += random.uniform(-0.20, 0.20) * side
    return cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2

def build_items():
    items = []
    t0 = time.time()
    for stem in train_stems:
        allann = records[stem][medoid_ann[stem]]
        gt_union = semantic_mask(allann)
        g = gray_of(stem)
        for a in allann:
            m = semantic_mask([a])
            if m.sum() < 20:
                continue
            ys, xs = np.where(m)
            bcx, bcy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
            bside = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
            for _ in range(2):
                bx0, by0, bx1, by1 = jitter(bcx, bcy, bside)
                cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
                cside = max(bside * random.uniform(1.0, 1.8), MIN_SIDE)
                _, (x0, y0, side) = make_crop(stem, cx, cy, cside, (bx0, by0, bx1, by1))
                tgt = crop_target(m, x0, y0, side)
                items.append((stem, int(cx), int(cy), side, (int(bx0), int(by0), int(bx1), int(by1)), tgt, True))
        n_neg = max(1, int(0.67 * max(1, len(allann))))
        gt_far = gt_union.copy()
        if gt_union.any():
            gt_far = cv2.dilate(gt_union, np.ones((41, 41), np.uint8))
        dark_cand = (g < np.percentile(g, 30)) & (gt_far == 0)
        dys, dxs = np.where(dark_cand)
        bk = g > np.percentile(g, 60)
        bys, bxs = np.where(bk)
        for i in range(n_neg):
            side = random.randint(MIN_SIDE, 384)
            if (i % 5 < 3) and len(dxs):
                cxn, cyn = int(dxs[random.randrange(len(dxs))]), int(dys[random.randrange(len(dxs))])
            elif len(bxs):
                cxn, cyn = int(bxs[random.randrange(len(bxs))]), int(bys[random.randrange(len(bxs))])
            else:
                cxn, cyn = random.randint(0, W - 1), random.randint(0, H - 1)
            x0, y0 = cxn - side // 2, cyn - side // 2
            xs0, ys0 = max(0, x0), max(0, y0)
            xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
            if gt_union[ys0:ys1, xs0:xs1].any():
                continue
            u = random.uniform(0.5, 0.95)
            inb = side * u
            bx0 = x0 + (side - inb) / 2; by0 = y0 + (side - inb) / 2
            items.append((stem, cxn, cyn, side,
                          (int(bx0), int(by0), int(bx0 + inb), int(by0 + inb)),
                          np.zeros((side, side), np.uint8), False))
    random.shuffle(items)
    npos = sum(1 for i in items if i[6])
    print(f"[ref-items] crops={len(items)} pos={npos} neg%={(len(items)-npos)/max(1,len(items)):.1%} [{time.time()-t0:.0f}s]", flush=True)
    return items

from torchvision.models import resnet18

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        r = resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.d3 = self._up(512 + 256, 256)
        self.d2 = self._up(256 + 128, 128)
        self.d1 = self._up(128 + 64, 64)
        self.d0 = self._up(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
    @staticmethod
    def _up(in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0)); s2 = self.l2(s1); s3 = self.l3(s2); s4 = self.l4(s3)
        d = self.d3(torch.cat([F.interpolate(s4, scale_factor=2, mode="bilinear", align_corners=False), s3], 1))
        d = self.d2(torch.cat([d, s2], 1)); d = self.d1(torch.cat([d, s1], 1)); d = self.d0(torch.cat([d, s0], 1))
        return self.head(d)

class DS(Dataset):
    def __init__(self, its):
        self.its = its
    def __len__(self):
        return len(self.its)
    def __getitem__(self, i):
        stem, cx, cy, side, box, tgt, _ = self.its[i]
        x, _ = make_crop(stem, cx, cy, side, box)
        xr = cv2.resize(np.ascontiguousarray(x.transpose(1, 2, 0)), (CROP, CROP)).transpose(2, 0, 1)
        yr = cv2.resize(tgt, (CROP, CROP), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(xr.copy()) - MEAN) / STD, torch.from_numpy(yr).float()

def train_refiner():
    items = build_items()
    n_val = max(64, int(0.04 * len(items)))
    val_items, train_items = items[:n_val], items[n_val:]
    dl = DataLoader(DS(train_items), batch_size=32, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    model = Refiner().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    bce = nn.BCEWithLogitsLoss()
    def soft_dice(logits, y):
        p = torch.sigmoid(logits)
        num = 2 * (p * y).sum((1, 2)) + 1
        den = p.sum((1, 2)) + y.sum((1, 2)) + 1
        return 1 - (num / den).mean()
    for ep in range(EPOCHS):
        model.train(); tl = 0.0; nb = 0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda"):
                out = model(xb).squeeze(1)
                loss = 0.45 * bce(out, yb) + 0.55 * soft_dice(out.float(), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += float(loss); nb += 1
        sch.step()
        print(f"[ref ep{ep}] loss={tl/max(1,nb):.4f}", flush=True)
    torch.save({"model": model.state_dict(), "fold_excluded": FOLD}, WORK / f"ref_fold{FOLD}.pt")
    return model

refiner = train_refiner()
refiner.eval()
print("[ref] done", flush=True)

# ================= 两段式评测（fold1） =================
def disk_mask(g):
    thr = np.percentile(g, 60)
    m = (g > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return np.ones_like(m)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    d = (lab == big).astype(np.uint8)
    d = cv2.morphologyEx(d, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    k = int(0.01 * min(H, W)) // 2 * 2 + 1
    return cv2.erode(d, np.ones((k, k), np.uint8))

@torch.no_grad()
def refine_prob(g, bx0, by0, bx1, by1):
    bside = max(bx1 - bx0, by1 - by0)
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    side = max(int(max(bside * CONTEXT, MIN_SIDE)), MIN_SIDE)
    x0, y0 = int(cx - side // 2), int(cy - side // 2)
    patch = np.zeros((side, side), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
    patch[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = g[ys0:ys1, xs0:xs1]
    seed = np.zeros((side, side), np.float32)
    bx0c, by0c = max(xs0, int(bx0)), max(ys0, int(by0))
    bx1c, by1c = min(xs1, int(bx1)), min(ys1, int(by1))
    if bx1c > bx0c and by1c > by0c:
        seed[by0c - y0:by1c - y0, bx0c - x0:bx1c - x0] = 1.0
    x3 = np.stack([patch / 255.0, CLAHE.apply(patch) / 255.0, seed], 0).astype(np.float32)
    xr = cv2.resize(np.ascontiguousarray(x3.transpose(1, 2, 0)), (CROP, CROP)).transpose(2, 0, 1)
    t = ((torch.from_numpy(xr) - MEAN) / STD).unsqueeze(0).to(DEVICE)
    views = torch.cat([t, torch.flip(t, [3]), torch.flip(t, [2]), torch.flip(t, [2, 3])], 0)
    with torch.autocast("cuda"):
        p = torch.sigmoid(refiner(views)).squeeze(1).float()
    p = p[0:1] + torch.flip(p[1:2], [2]) + torch.flip(p[2:3], [1]) + torch.flip(p[3:4], [1, 2])
    prob = cv2.resize((p[0] / 4.0).cpu().numpy(), (side, side))
    full = np.zeros((H, W), np.float16)
    full[ys0:ys1, xs0:xs1] = prob[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return full

def mask_to_rle(m):
    c = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"]
    return c.decode("utf-8") if isinstance(c, bytes) else c

gt_rows = []
for im in coco["images"]:
    stem = os.path.splitext(im["file_name"])[0]
    if fold_of[stem] != FOLD:
        continue
    for k, a in enumerate(ann_by_img.get(im["id"], [])):
        rle = mask_util.merge(mask_util.frPyObjects(a["segmentation"], H, W))["counts"]
        gt_rows.append({"filament_id": f"{im['id']}_{k}",
                        "segmentation_rle": rle.decode() if isinstance(rle, bytes) else rle})
GT = pd.DataFrame(gt_rows, columns=["filament_id", "segmentation_rle"])

def pq_score(gt_df, pred_df):
    gt_group = gt_df["filament_id"].str.split("_").str[0]
    pred_group = pred_df["filament_id"].str.split("_").str[0].str.split("-").str[-1]
    tp_ious, fp, fn = [], 0, 0
    for g_ in gt_group.unique():
        gt_r = gt_df.loc[gt_group == g_, "segmentation_rle"].tolist()
        st = g_.split("-", 1)[1]
        pr = pred_df.loc[pred_group == st, "segmentation_rle"].tolist()
        if gt_r and pr:
            dec = lambda rs: [{"size": [H, W], "counts": r.encode()} for r in rs]
            iou = np.asarray(mask_util.iou(dec(pr), dec(gt_r), [0] * len(gt_r))).T
            hit = iou > 0.5
            tp_ious.extend(iou[hit].tolist())
            fp += int((hit.sum(0) == 0).sum())
            fn += int((hit.sum(1) == 0).sum())
        else:
            fp += len(pr); fn += len(gt_r)
    denom = len(tp_ious) + 0.5 * fp + 0.5 * fn
    return (sum(tp_ious) / denom if denom else 0.0), len(tp_ious), fp, fn, (float(np.mean(tp_ious)) if tp_ious else 0.0)

THR, GATE = 0.55, 0.10
CONFS = (0.20, 0.25, 0.30, 0.35)
rles = {cf: [] for cf in CONFS}
raw10 = {}
t0 = time.time()
for n, stem in enumerate(val_stems):
    g = gray_of(stem)
    disk = disk_mask(g)
    r = det.predict(str(Path(train_images_dir) / f"{stem}.jpeg"), conf=0.10, iou=0.60,
                    agnostic_nms=True, retina_masks=True, imgsz=1536, device=0, verbose=False)[0]
    dets = []
    if r.boxes is not None and len(r.boxes):
        dets = [[float(a), float(b), float(c), float(d), float(cf)]
                for (a, b, c, d), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    raw10[stem] = dets
    probs = [(d[4], max(d[2] - d[0], d[3] - d[1]), refine_prob(g, d[0], d[1], d[2], d[3])) for d in dets]
    for cf in CONFS:
        kept = []
        for conf, bside, prob in probs:
            if conf < cf:
                continue
            m = (prob > THR).astype(np.uint8) & disk
            a = int(m.sum())
            if a >= MIN_AREA and a >= GATE * max(96.0, bside) ** 2 * 0.05:
                kept.append((conf, m))
        occ = np.zeros((H, W), bool)
        for conf, m in sorted(kept, key=lambda z: -z[0]):
            m2 = m.astype(bool) & ~occ
            if m2.sum() >= MIN_AREA:
                occ |= m2
                rles[cf].append((stem, mask_to_rle(m2)))
    del probs
    if (n + 1) % 40 == 0:
        print(f"[eval] {n+1}/{len(val_stems)} [{time.time()-t0:.0f}s]", flush=True)

json.dump(raw10, open(WORK / f"val_fold{FOLD}_raw_dets.json", "w"))
summary = []
for cf in CONFS:
    counts = defaultdict(int)
    rows = []
    for stem, rle in rles[cf]:
        rows.append({"filament_id": f"{stem}_{counts[stem]}", "segmentation_rle": rle})
        counts[stem] += 1
    pdf = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    pq, tp, fp, fn, miou = pq_score(GT, pdf)
    summary.append({"fold": FOLD, "conf": cf, "thr": THR, "gate": GATE, "PQ": round(pq, 4),
                    "TP": tp, "FP": fp, "FN": fn, "meanIoU": round(miou, 4)})
    print(f"fold{FOLD} conf{cf}: PQ={pq:.4f} TP={tp} FP={fp} FN={fn} meanIoU={miou:.4f}", flush=True)
json.dump({"sweep": summary, "fold0_champion": 0.3661},
          open(WORK / f"pq_report_fold{FOLD}.json", "w"), indent=1)
best = max(summary, key=lambda z: z["PQ"])
print(f"[BEST] {best}   对照 fold0 冠军 0.3661", flush=True)
print("[done]", sorted(os.listdir(WORK)), flush=True)
