###############################################################################
#
#   리뷰 데이터 BERT — ③ EDA 4배수 증강 학습
#
#   Google Drive > thesis 폴더에 넣을 파일:
#     - Twitter_EDA_1Real (1).csv
#     - Instagram_EDA_1Real (1).csv
#     - Facebook_EDA_1Real (1).csv
#
#   결과: review_result_eda.json (Google Drive에 저장)
#   예상 소요: A100 기준 약 3~4시간
#
###############################################################################

import subprocess
subprocess.run(["pip", "install", "-q", "transformers", "scikit-learn", "scipy", "tqdm"])

from google.colab import drive
drive.mount("/content/drive")

import os, json, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from tqdm import tqdm

DRIVE_FOLDER = "/content/drive/MyDrive/thesis"

CONFIG = {
    "test_size": 0.15,
    "n_folds": 10,
    "random_seed": 42,
    "model_name": "bert-base-uncased",
    "max_length": 256,
    "num_epochs": 10,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "dropout": 0.1,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(CONFIG["random_seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ===================== 데이터 로드 =====================
print("\n" + "=" * 60)
print("  데이터 로드")
print("=" * 60)

eda_files = [
    f"{DRIVE_FOLDER}/Twitter_EDA_1Real (1).csv",
    f"{DRIVE_FOLDER}/Instagram_EDA_1Real (1).csv",
    f"{DRIVE_FOLDER}/Facebook_EDA_1Real (1).csv",
]
dfs = []
for path in eda_files:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["source"] = os.path.basename(path).split("_")[0]
    dfs.append(df)

raw_data = pd.concat(dfs, ignore_index=True)
all_texts = raw_data["content"].tolist()
all_labels = raw_data["label"].tolist()
print(f"  원본: {len(all_texts)}건")

# ===================== Held-out Test 분리 =====================
indices = list(range(len(all_texts)))
train_val_idx, test_idx = train_test_split(
    indices, test_size=CONFIG["test_size"],
    random_state=CONFIG["random_seed"], stratify=all_labels
)
train_val_texts = [all_texts[i] for i in train_val_idx]
train_val_labels = [all_labels[i] for i in train_val_idx]
test_texts = [all_texts[i] for i in test_idx]
test_labels = [all_labels[i] for i in test_idx]
trainval_to_original = {local: original for local, original in enumerate(train_val_idx)}
print(f"  Train+Val: {len(train_val_texts)}건 / Test: {len(test_texts)}건")

# 토크나이저
print("  토크나이저 로드 중...")
tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
print("  토크나이저 로드 완료!")

# ===================== EDA 증강 함수 =====================
def get_eda_augmented(train_local_indices):
    aug_texts, aug_labels = [], []
    for local_idx in train_local_indices:
        orig_idx = trainval_to_original[local_idx]
        row = raw_data.iloc[orig_idx]
        label = row["label"]
        for col in ["SR", "RI", "RS", "RD"]:
            eda_text = row[col]
            if isinstance(eda_text, str) and len(eda_text.strip()) > 5:
                aug_texts.append(eda_text)
                aug_labels.append(label)
    return aug_texts, aug_labels

# ===================== Dataset =====================
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ===================== 학습/평가 =====================
def train_one_epoch(model, dataloader, optimizer, scheduler, epoch_num=0):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"    Train Epoch {epoch_num}", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels_list = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_list.extend(batch["labels"].cpu().numpy())
    acc = accuracy_score(all_labels_list, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels_list, all_preds, average="binary", zero_division=0)
    return {"val_loss": total_loss/len(dataloader), "accuracy": float(acc),
            "precision": float(prec), "recall": float(rec), "f1": float(f1)}

# ===================== CV 실험 =====================
print(f"\n{'='*60}")
print(f"  EDA 4배수 증강 학습 시작")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=CONFIG["n_folds"], shuffle=True,
                      random_state=CONFIG["random_seed"])
fold_results = []
all_epoch_results = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_val_texts, train_val_labels)):
    print(f"\n  --- Fold {fold_idx+1}/{CONFIG['n_folds']} ---")

    tr_texts = [train_val_texts[i] for i in train_idx]
    tr_labels = [train_val_labels[i] for i in train_idx]
    va_texts = [train_val_texts[i] for i in val_idx]
    va_labels = [train_val_labels[i] for i in val_idx]

    # ★ train에만 EDA 증강 ★
    aug_texts, aug_labels = get_eda_augmented(train_idx.tolist())
    tr_texts = tr_texts + aug_texts
    tr_labels = tr_labels + aug_labels

    if fold_idx == 0:
        print(f"    원본 train: {len(train_idx)}건 + EDA: {len(aug_texts)}건 = 총 {len(tr_texts)}건")

    train_dataset = ReviewDataset(tr_texts, tr_labels, tokenizer, CONFIG["max_length"])
    val_dataset = ReviewDataset(va_texts, va_labels, tokenizer, CONFIG["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"],
                            num_workers=2, pin_memory=True)

    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=2,
        hidden_dropout_prob=CONFIG["dropout"],
        attention_probs_dropout_prob=CONFIG["dropout"]).to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                     weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * CONFIG["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * CONFIG["warmup_ratio"]), total_steps)

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch+1)
        metrics = evaluate(model, val_loader)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch + 1
        metrics["fold"] = fold_idx + 1
        all_epoch_results.append(metrics)
        print(f"    Epoch {epoch+1}: train_loss={train_loss:.4f} "
              f"val_loss={metrics['val_loss']:.4f} F1={metrics['f1']:.4f} "
              f"Acc={metrics['accuracy']:.4f}")

    fold_results.append(all_epoch_results[-1])
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    # 중간저장
    interim = {"fold_results_so_far": fold_results}
    with open(f"{DRIVE_FOLDER}/review_result_eda_interim.json", "w") as f:
        json.dump(interim, f, indent=2, default=str)

# ===================== Held-out Test =====================
print(f"\n{'='*60}")
print(f"  Held-out Test")
print(f"{'='*60}")

tr_texts = list(train_val_texts)
tr_labels = list(train_val_labels)
aug_texts, aug_labels = get_eda_augmented(list(range(len(train_val_texts))))
tr_texts = tr_texts + aug_texts
tr_labels = tr_labels + aug_labels

train_dataset = ReviewDataset(tr_texts, tr_labels, tokenizer, CONFIG["max_length"])
test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, CONFIG["max_length"])
train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                          shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                         num_workers=2, pin_memory=True)

model = BertForSequenceClassification.from_pretrained(
    CONFIG["model_name"], num_labels=2).to(device)
optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                 weight_decay=CONFIG["weight_decay"])
total_steps = len(train_loader) * CONFIG["num_epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer, int(total_steps * CONFIG["warmup_ratio"]), total_steps)

for epoch in range(CONFIG["num_epochs"]):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, epoch+1)
    print(f"    Test Train Epoch {epoch+1}: loss={train_loss:.4f}")

test_metrics = evaluate(model, test_loader)
print(f"    ✅ Test F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f}")
del model; torch.cuda.empty_cache()

# ===================== 저장 =====================
f1s = [r["f1"] for r in fold_results]
output = {
    "config": CONFIG, "condition": "eda",
    "timestamp": datetime.now().isoformat(),
    "cv_results": {"folds": fold_results,
                   "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s))},
    "epoch_data": all_epoch_results,
    "test_results": test_metrics,
}
with open(f"{DRIVE_FOLDER}/review_result_eda.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  CV F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"  Test F1: {test_metrics['f1']:.4f}")
print(f"  저장: {DRIVE_FOLDER}/review_result_eda.json")
