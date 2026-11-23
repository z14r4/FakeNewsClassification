"""
Multimodal fake-news classifier: BERT (text) + DINOv2 (image), late fusion.

Reads:
  manifests/train.csv, manifests/test.csv
  images/{train,test}/{id}.{ext}

Logs everything to Weights & Biases.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image, ImageFile
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parent


class FakedditDataset(Dataset):
    def __init__(self, csv_path, tokenizer, image_root, max_len=64, image_size=128, train=True):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_root = Path(image_root)
        self.max_len = max_len
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if train:
            self.tx = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.tx = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                norm,
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["clean_title"]) if pd.notna(row["clean_title"]) else ""
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        path = self.image_root / row["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (64, 64), (0, 0, 0))
        img = self.tx(img)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "label": torch.tensor(int(row["label_2way"]), dtype=torch.long),
        }


class DinoEncoder(nn.Module):
    """DINOv2 ViT trunk + projection head. CLS-token pooled."""

    def __init__(self, dino_name, out_dim=256, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(dino_name)
        in_feats = self.backbone.config.hidden_size
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, out_dim))

    def forward(self, x):
        cls = self.backbone(pixel_values=x).last_hidden_state[:, 0, :]
        return self.proj(cls)


class MultimodalClassifier(nn.Module):
    def __init__(self, bert_name, dino_name, num_classes=2, fusion_dim=256, dropout=0.3):
        super().__init__()
        self.text = AutoModel.from_pretrained(bert_name)
        text_hidden = self.text.config.hidden_size
        self.image = DinoEncoder(dino_name, out_dim=fusion_dim, dropout=dropout)
        self.text_proj = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(text_hidden, fusion_dim), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, image):
        cls = self.text(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_emb = self.text_proj(cls)
        img_emb = self.image(image)
        return self.classifier(torch.cat([text_emb, img_emb], dim=-1))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, total_n = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        img = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(ids, mask, img)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += loss.item()
        total_n += y.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(-1).numpy()
    metrics = {
        "loss": total_loss / total_n,
        "acc": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }
    if probs.shape[1] == 2:
        try:
            metrics["auc"] = roc_auc_score(labels, probs[:, 1])
        except ValueError:
            metrics["auc"] = float("nan")
    return metrics, preds, labels, probs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default=str(ROOT / "manifests/train.csv"))
    ap.add_argument("--test_csv", default=str(ROOT / "manifests/test.csv"))
    ap.add_argument("--image_root", default=str(ROOT))
    ap.add_argument("--bert", default="bert-base-uncased")
    ap.add_argument("--dino", default="facebook/dinov2-base",
                    help="HuggingFace ID of the DINO/DINOv2 image encoder")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224,
                    help="DINOv2 uses 14x14 patches, so this must be a multiple of 14 (e.g. 224, 252)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5,
                    help="early-stop after this many epochs without test/acc improvement; <=0 disables")
    ap.add_argument("--class_weight", default="balanced", choices=["none", "balanced"],
                    help="reweight cross-entropy to counter train-set class imbalance")
    ap.add_argument("--lr_bert", type=float, default=2e-5)
    ap.add_argument("--lr_dino", type=float, default=2e-5)
    ap.add_argument("--lr_other", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--fusion_dim", type=int, default=256)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--wandb_project", default="fake-news-multimodal")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--out_dir", default=str(ROOT / "runs"))
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        dir=str(out_dir),
        mode=args.wandb_mode,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.bert)

    train_ds = FakedditDataset(args.train_csv, tokenizer, args.image_root, args.max_len, args.image_size, train=True)
    test_ds = FakedditDataset(args.test_csv, tokenizer, args.image_root, args.max_len, args.image_size, train=False)
    print(f"[data] train={len(train_ds)}  test={len(test_ds)}")
    print(f"[data] train label_2way: {train_ds.df['label_2way'].value_counts().to_dict()}")
    print(f"[data] test  label_2way: {test_ds.df['label_2way'].value_counts().to_dict()}")
    train_label_counts = {str(k): int(v) for k, v in train_ds.df["label_2way"].value_counts().to_dict().items()}
    test_label_counts = {str(k): int(v) for k, v in test_ds.df["label_2way"].value_counts().to_dict().items()}
    wandb.config.update({
        "train_samples": len(train_ds),
        "test_samples": len(test_ds),
        "train_label_counts": train_label_counts,
        "test_label_counts": test_label_counts,
    })
    wandb.summary["train_samples"] = len(train_ds)
    wandb.summary["test_samples"] = len(test_ds)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = MultimodalClassifier(
        args.bert, args.dino, num_classes=2, fusion_dim=args.fusion_dim, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,}  trainable={n_trainable:,}")
    wandb.config.update({"params": n_params, "trainable_params": n_trainable})
    wandb.watch(model, log="gradients", log_freq=200)

    bert_params = list(model.text.parameters())
    dino_params = list(model.image.backbone.parameters())
    other_params = [
        p for n, p in model.named_parameters()
        if not n.startswith("text.") and not n.startswith("image.backbone.")
    ]
    optim = torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.lr_bert},
            {"params": dino_params, "lr": args.lr_dino},
            {"params": other_params, "lr": args.lr_other},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    # Cap warmup at one epoch — with large --epochs (e.g. 100) the ratio-based
    # warmup would otherwise stretch over many epochs and starve early training
    # of LR, causing early-stopping to fire before the schedule has even peaked.
    warmup_steps = min(int(total_steps * args.warmup_ratio), len(train_loader))
    print(f"[sched] total_steps={total_steps}  warmup_steps={warmup_steps}")
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    if args.class_weight == "balanced":
        counts = train_ds.df["label_2way"].value_counts().sort_index().to_numpy()
        w = counts.sum() / (len(counts) * counts)
        class_weights = torch.tensor(w, dtype=torch.float, device=device)
        print(f"[loss] balanced class weights: {class_weights.tolist()}")
    else:
        class_weights = None

    best_acc = 0.0
    epochs_since_best = 0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            img = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            logits = model(ids, mask, img)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            sched.step()

            preds = logits.argmax(-1)
            running_loss += loss.item() * y.size(0)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            global_step += 1
            if global_step % args.log_every == 0:
                wandb.log({
                    "train/loss_step": loss.item(),
                    "train/acc_step": (preds == y).float().mean().item(),
                    "train/lr_bert": optim.param_groups[0]["lr"],
                    "train/lr_dino": optim.param_groups[1]["lr"],
                    "train/lr_other": optim.param_groups[2]["lr"],
                    "epoch": epoch,
                    "step": global_step,
                })
            pbar.set_postfix(loss=f"{running_loss/running_total:.4f}",
                             acc=f"{running_correct/running_total:.4f}")

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        test_metrics, preds, labels, _ = evaluate(model, test_loader, device)

        log = {
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_acc": train_acc,
            **{f"test/{k}": v for k, v in test_metrics.items()},
        }
        wandb.log(log)

        cm = confusion_matrix(labels, preds)
        wandb.log({
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels.tolist(),
                preds=preds.tolist(),
                class_names=["fake", "real"],
            ),
            "epoch": epoch,
        })

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            + " ".join([f"test_{k}={v:.4f}" for k, v in test_metrics.items()])
            + f" | cm={cm.tolist()}"
        )

        if test_metrics["acc"] > best_acc:
            best_acc = test_metrics["acc"]
            epochs_since_best = 0
            ckpt = out_dir / "best.pt"
            torch.save(
                {"model": model.state_dict(), "args": vars(args),
                 "epoch": epoch, "metrics": test_metrics},
                ckpt,
            )
            wandb.summary["best/acc"] = test_metrics["acc"]
            wandb.summary["best/f1_macro"] = test_metrics["f1_macro"]
            wandb.summary["best/auc"] = test_metrics.get("auc", float("nan"))
            wandb.summary["best/epoch"] = epoch
            print(f"[ckpt] new best -> {ckpt}")
        else:
            epochs_since_best += 1
            if args.patience > 0 and epochs_since_best >= args.patience:
                print(f"[early-stop] no improvement for {epochs_since_best} epochs (best_acc={best_acc:.4f})")
                break

    final_ckpt = out_dir / "last.pt"
    torch.save({"model": model.state_dict(), "args": vars(args)}, final_ckpt)
    print(f"[ckpt] last -> {final_ckpt}")
    wandb.finish()


if __name__ == "__main__":
    main()
