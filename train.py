#!/usr/bin/env python3
# train.py — Multi-mode: binary | food | nofood
# DEBUG VERSION (robusta y clara)

"""
train.py — Entrenamiento de modelos sobre Food-101

Soporta modos: 'binary' (food vs no-food) y 'food' (clasificación multi-clase).
Usa timm, albumentations y entrenamiento con AMP.

Ejemplo:
  python train.py --mode binary --data_dir /content/Food-101 --no_food_dir /content/drive/MyDrive/no_food --model_dir ./models --epochs 12
"""


import os, argparse, time
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import WeightedRandomSampler


from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

import timm
from torch.optim import AdamW

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from collections import Counter

# ======================================================
# DATASETS
# ======================================================
class AlbumentationsImageFolder(Dataset):
    def __init__(self, root_dir, transform, img_size):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.samples = []

        classes = sorted(d for d in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, d)))

        if len(classes) == 0:
            raise RuntimeError(f"No classes found in {root_dir}")

        self.class_to_idx = {c:i for i,c in enumerate(classes)}
        self.classes = classes

        for c in classes:
            cdir = os.path.join(root_dir, c)
            for f in os.listdir(cdir):
                self.samples.append((os.path.join(cdir, f), self.class_to_idx[c]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p,y = self.samples[idx]
        try:
            img = np.array(Image.open(p).convert("RGB"))
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = self.transform(image=img)["image"]
        return img, y


class BinaryFoodNoFoodDataset(Dataset):
    def __init__(self, food_root, transform, img_size,
                 no_food_dir=None, split="train",
                 val_ratio=0.2, seed=42):

        self.transform = transform
        self.img_size = img_size
        self.samples = []

        # FOOD
        for cls in os.listdir(food_root):
            cls_dir = os.path.join(food_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                self.samples.append((os.path.join(cls_dir, f), 0))

        # NO FOOD
        if no_food_dir:
            nf_files = []
            for root, _, files in os.walk(no_food_dir):
                for f in files:
                    if f.lower().endswith((".jpg",".png",".jpeg")):
                        nf_files.append(os.path.join(root, f))

            if len(nf_files) > 0:
                tr_nf, va_nf = train_test_split(
                    nf_files,
                    test_size=val_ratio,
                    random_state=seed,
                    shuffle=True
                )
                use_files = tr_nf if split == "train" else va_nf
                for p in use_files:
                    self.samples.append((p, 1))

        self.classes = ["food", "no_food"]

        if len(self.samples) == 0:
            raise RuntimeError("Binary dataset is empty")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        try:
            img = np.array(Image.open(p).convert("RGB"))
        except Exception:
            img = np.zeros((self.img_size, self.img_size, 3), np.uint8)
        img = self.transform(image=img)["image"]
        return img, y

# ======================================================
# TRANSFORMS
# ======================================================
def get_transforms(sz):
    size = (sz, sz)

    train_t = A.Compose([
        A.RandomResizedCrop(size=size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.02, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_t = A.Compose([
        A.Resize(height=sz, width=sz),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_t, val_t

# ======================================================
# TRAIN / EVAL
# ======================================================
def train_epoch(model, loader, opt, scaler, dev, crit, epoch, log_every):
    model.train()
    total_loss = 0
    seen = 0
    start = time.time()

    print(f"🟡 Epoch {epoch} | {len(loader)} batches")

    for i,(x,y) in enumerate(loader):
        x,y = x.to(dev), y.to(dev)
        opt.zero_grad(set_to_none=True)

        with autocast():
            out = model(x)
            loss = crit(out,y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        seen += x.size(0)

        if i % log_every == 0:
            print(f"   batch {i}/{len(loader)} | loss {loss.item():.4f}")

    avg = total_loss / seen
    print(f"🟢 Epoch {epoch} done | avg train loss {avg:.4f} | {time.time()-start:.1f}s")
    return avg

def eval_model(model, loader, dev):
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(dev), y.to(dev)
            p = model(x).argmax(1)
            ok += (p==y).sum().item()
            tot += y.size(0)
    return ok / tot

# ======================================================
# MAIN
# ======================================================
# ------------------ PEGAR/REEMPLAZAR main() POR ESTE BLOQUE ------------------
import shutil
import sys

def parse_args_robust():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--data_dir")
    ap.add_argument("--no_food_dir")
    ap.add_argument("--model_dir", default="./models")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=192)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--copy_to_tmp", action="store_true",
                    help="Copiar dataset a /tmp para evitar lentitud de Google Drive (recomendado).")
    ap.add_argument("--save_every_epoch", type=int, default=1,
                    help="Guardar checkpoint cada N épocas.")
    # Si ejecutas desde bash con args funcionará como antes.
    # Si ejecutas dentro de IPython/Colab y no pasas args, evitamos SystemExit y devolvemos None.
    try:
        args = ap.parse_args()
    except SystemExit:
        # Estamos en entorno interactivo sin args: devolver None para que el entorno (shell) decida.
        print(" argparse: falta --mode o estás en entorno interactivo. Usa la ejecución por línea de comandos (%%bash) o pasa args explícitos.", flush=True)
        raise
    return args

def main():
    args = parse_args_robust()

    # Si pides copiar a tmp (recomendado en Colab cuando tu dataset está en Drive)
    if args.copy_to_tmp and args.data_dir:
        tmp_dest = f"/tmp/food_dataset_{int(time.time())}"
        if os.path.exists(tmp_dest):
            print("🗑 Limpiando tmp anterior...", flush=True)
            shutil.rmtree(tmp_dest)
        print(f"Copiando datos desde {args.data_dir} -> {tmp_dest} (esto puede tardar unos minutos)...", flush=True)
        shutil.copytree(args.data_dir, tmp_dest)
        args.data_dir = tmp_dest
        print("Copia completada.", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}", flush=True)

    tr_t, va_t = get_transforms(args.img_size)

    if args.mode == "food":
        tr_ds = AlbumentationsImageFolder(os.path.join(args.data_dir,"train"), tr_t, args.img_size)
        va_ds = AlbumentationsImageFolder(os.path.join(args.data_dir,"val"), va_t, args.img_size)
        classes = tr_ds.classes
        with open(os.path.join(args.model_dir, f"classes_{args.mode}.txt"), "w") as f:
            f.write("\n".join(classes))
    elif args.mode == "binary":
        tr_ds = BinaryFoodNoFoodDataset(os.path.join(args.data_dir,"train"), tr_t, args.img_size, args.no_food_dir,"train")
        va_ds = BinaryFoodNoFoodDataset(os.path.join(args.data_dir,"val"), va_t, args.img_size, args.no_food_dir,"val")
        classes = ["food","no_food"]
    else:
        raise ValueError("Modo no soportado")

    print(f"Train samples: {len(tr_ds)}", flush=True)
    print(f"Val samples:   {len(va_ds)}", flush=True)
    print(f"Classes: {len(classes)}", flush=True)

    counts = Counter(y for _,y in tr_ds.samples)
    print("Class distribution:", counts, flush=True)

    sample_weights = [1.0 / counts[y] for _, y in tr_ds.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # Si estás leyendo desde Drive, para debug pon workers=0 (evita deadlocks)
    workers = args.workers
    if 'google.colab' in sys.modules:
        print("Detectado Colab: si tienes problemas con DataLoader prueba --workers 0", flush=True)

    tr_dl = DataLoader(
        tr_ds,
        batch_size=args.bs,
        sampler=sampler,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False
    )
    va_dl = DataLoader(va_ds, args.bs, shuffle=False,
                       num_workers=workers, pin_memory=(device.type == "cuda"))

    model = timm.create_model("efficientnet_b0", pretrained=True,
                              num_classes=len(classes)).to(device)

    opt = AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler()
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0
    os.makedirs(args.model_dir, exist_ok=True)

    try:
        for e in range(1, args.epochs+1):
            # log inicial por epoch
            print(f"\n=== 🔸 Época {e}/{args.epochs} ===", flush=True)
            print(f"🟡 Epoch {e} | {len(tr_dl)} batches (bs={args.bs})", flush=True)
            train_epoch(model, tr_dl, opt, scaler, device, crit, e, args.log_every)
            acc = eval_model(model, va_dl, device)
            print(f"📈 VAL ACC = {acc:.4f}", flush=True)

            # guardado por epochs según parámetro
            if e % args.save_every_epoch == 0:
                ckpt = {
                    "epoch": e,
                    "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "acc": acc
                }
                path = os.path.join(args.model_dir, f"checkpoint_epoch{e}_{args.mode}.pth")
                torch.save(ckpt, path)
                print(f"💾 Checkpoint guardado: {path}", flush=True)

            if acc > best:
                best = acc
                torch.save(model.state_dict(),
                           os.path.join(args.model_dir, f"best_{args.mode}.pth"))
                print("NEW BEST MODEL SAVED", flush=True)

    except Exception as exc:
        # guardar checkpoint parcial y reportar excepción
        print("Excepción durante el entrenamiento, guardando checkpoint parcial...", flush=True)
        try:
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "exception": str(exc)
            }, os.path.join(args.model_dir, f"crash_partial_{args.mode}.pth"))
            print("Checkpoint parcial guardado.", flush=True)
        except Exception as e2:
            print("No se pudo guardar checkpoint parcial:", e2, flush=True)
        raise

    print("-> Finished | Best acc:", best, flush=True)

# ---------------------------------------------------------------------------


