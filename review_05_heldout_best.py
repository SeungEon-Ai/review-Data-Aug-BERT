###############################################################################
#
#   리뷰 데이터 — Held-out Test (Best Epoch 기준)
#
#   CV에서 찾은 평균 best epoch 수만큼 학습 후 test 평가
#   4가지 조건 순차 실행
#
#   Google Drive > thesis 폴더에 넣을 파일:
#     - Twitter_EDA_1Real (1).csv
#     - Instagram_EDA_1Real (1).csv
#     - Facebook_EDA_1Real (1).csv
#     - review_gpt4omini_augmented.csv
#     - review_result_base.json (CV 결과)
#     - review_result_eda.json
#     - review_result_gpt.json
#
#   결과: review_result_heldout_best.json
#   예상 소요: A100 약 30분 / L4 약 1시간
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datetime import datetime
from tqdm import tqdm

DRIVE_FOLDER = "/content/drive/MyDrive/thesis"

CONFIG = {
    "test_size": 0.15,
    "random_seed": 42,
    "model_name": "bert-base-uncased",
    "max_length": 256,
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

# ===================== CV 결과에서 Best Epoch 계산 =====================
print("\n" + "=" * 60)
print("  STEP 1: CV 결과에서 Best Epoch 계산")
print("=" * 60)

with open(f"{DRIVE_FOLDER}/review_result_base.json") as f:
    base_data = json.load(f)
with open(f"{DRIVE_FOLDER}/review_result_eda.json") as f:
    eda_data = json.load(f)
with open(f"{DRIVE_FOLDER}/review_result_gpt.json") as f:
    gpt_data = json.load(f)

def get_best_epochs(epoch_data):
    """fold별 best F1 epoch 번호 추출"""
    folds = {}
    for record in epoch_data:
        fold = record["fold"]
        if fold not in folds or record["f1"] > folds[fold]["f1"]:
            folds[fold] = record
    return [folds[f]["epoch"] for f in sorted(folds.keys())]

epoch_data_map = {
    "base": base_data["epoch_data"]["base"],
    "class_weight": base_data["epoch_data"]["class_weight"],
    "eda": eda_data["epoch_data"],
    "gpt": gpt_data["epoch_data"],
}

# 각 조건의 평균 best epoch (반올림)
best_epochs = {}
for cond, epochs in epoch_data_map.items():
    ep_list = get_best_epochs(epochs)
    avg = int(round(np.mean(ep_list)))
    best_epochs[cond] = avg
    print(f"  {cond:<15} fold별 best epochs: {ep_list} → 평균: {avg}")

# ===================== 데이터 로드 =====================
print("\n" + "=" * 60)
print("  STEP 2: 데이터 로드")
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

gpt_data_csv = pd.read_csv(f"{DRIVE_FOLDER}/review_gpt4omini_augmented.csv", encoding="utf-8-sig")

# ★ 동일한 분할 ★
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

tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
print("  토크나이저 로드 완료!")

# ===================== 증강 함수 =====================
def get_eda_augmented():
    aug_texts, aug_labels = [], []
    for local_idx in range(len(train_val_texts)):
        orig_idx = trainval_to_original[local_idx]
        row = raw_data.iloc[orig_idx]
        label = row["label"]
        for col in ["SR", "RI", "RS", "RD"]:
            eda_text = row[col]
            if isinstance(eda_text, str) and len(eda_text.strip()) > 5:
                aug_texts.append(eda_text)
                aug_labels.append(label)
    return aug_texts, aug_labels

def get_gpt_augmented():
    aug_texts, aug_labels = [], []
    for local_idx in range(len(train_val_texts)):
        orig_idx = trainval_to_original[local_idx]
        row = gpt_data_csv.iloc[orig_idx]
        label = row["label"]
        for col in ["Aug1", "Aug2", "Aug3", "Aug4"]:
            aug_text = row[col]
            if isinstance(aug_text, str) and len(aug_text.strip()) > 5:
                aug_texts.append(aug_text)
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
def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fct=None, epoch_num=0):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"    Train Epoch {epoch_num}", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = loss_fct(outputs.logits, batch["labels"]) if loss_fct else outputs.loss
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

# ===================== Held-out Test 실행 =====================
print("\n" + "=" * 60)
print("  STEP 3: Held-out Test (Best Epoch 기준)")
print("=" * 60)

test_results = {}

for cond in ["base", "class_weight", "eda", "gpt"]:
    num_epochs = best_epochs[cond]
    print(f"\n  --- {cond.upper()} (best epoch = {num_epochs}) ---")

    # 학습 데이터 준비
    tr_texts = list(train_val_texts)
    tr_labels = list(train_val_labels)

    loss_fct = None

    if cond == "eda":
        aug_texts, aug_labels = get_eda_augmented()
        tr_texts = tr_texts + aug_texts
        tr_labels = tr_labels + aug_labels
        print(f"    Train: {len(tr_texts)}건 (원본+EDA)")

    elif cond == "gpt":
        aug_texts, aug_labels = get_gpt_augmented()
        tr_texts = tr_texts + aug_texts
        tr_labels = tr_labels + aug_labels
        print(f"    Train: {len(tr_texts)}건 (원본+GPT)")

    elif cond == "class_weight":
        n_neg = tr_labels.count(0)
        n_pos = tr_labels.count(1)
        weight_pos = n_neg / n_pos if n_pos > 0 else 1.0
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, weight_pos]).to(device))
        print(f"    Train: {len(tr_texts)}건 (원본+가중치 [1.0, {weight_pos:.3f}])")

    else:
        print(f"    Train: {len(tr_texts)}건 (원본만)")

    # Dataset
    train_dataset = ReviewDataset(tr_texts, tr_labels, tokenizer, CONFIG["max_length"])
    test_dataset = ReviewDataset(test_texts, test_labels, tokenizer, CONFIG["max_length"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"],
                             num_workers=2, pin_memory=True)

    # 모델
    model = BertForSequenceClassification.from_pretrained(
        CONFIG["model_name"], num_labels=2,
        hidden_dropout_prob=CONFIG["dropout"],
        attention_probs_dropout_prob=CONFIG["dropout"]).to(device)

    optimizer = AdamW(model.parameters(), lr=CONFIG["learning_rate"],
                     weight_decay=CONFIG["weight_decay"])
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * CONFIG["warmup_ratio"]), total_steps)

    # Best epoch까지만 학습
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     loss_fct, epoch+1)
        print(f"    Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.4f}")

    # Test 평가
    metrics = evaluate(model, test_loader)
    metrics["best_epoch"] = num_epochs
    test_results[cond] = metrics
    print(f"    ✅ Test: F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} "
          f"Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")

    del model, optimizer, scheduler
    torch.cuda.empty_cache()

# ===================== 결과 저장 =====================
print("\n" + "=" * 60)
print("  STEP 4: 결과 저장")
print("=" * 60)

output = {
    "config": CONFIG,
    "timestamp": datetime.now().isoformat(),
    "best_epochs_used": best_epochs,
    "test_results": test_results,
}

output_path = f"{DRIVE_FOLDER}/review_result_heldout_best.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

# 요약
print(f"\n{'='*60}")
print(f"  SUMMARY — Held-out Test (Best Epoch 기준)")
print(f"{'='*60}")
print(f"  {'Condition':<15} {'Epochs':>6} {'F1':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
print(f"  {'─'*63}")
for cond in ["base", "class_weight", "eda", "gpt"]:
    r = test_results[cond]
    print(f"  {cond:<15} {r['best_epoch']:>6} {r['f1']:.4f}     {r['accuracy']:.4f}     "
          f"{r['precision']:.4f}     {r['recall']:.4f}")

print(f"\n  저장: {output_path}")
print(f"  완료!")
