###############################################################################
#
#   리뷰 데이터 BERT — ① Base(원본만) + ② Class Weight
#
#   Google Drive > thesis 폴더에 넣을 파일:
#     - Twitter_EDA_1Real (1).csv
#     - Instagram_EDA_1Real (1).csv
#     - Facebook_EDA_1Real (1).csv
#
#   결과: review_result_base.json (Google Drive에 저장)
#   예상 소요: A100 기준 약 2~3시간
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
from scipy import stats
from datetime import datetime
from tqdm import tqdm

DRIVE_FOLDER = "/content/drive/MyDrive/thesis"

# ===================== 하이퍼파라미터 =====================
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
print(f"  라벨: {pd.Series(all_labels).value_counts().to_dict()}")

# ===================== Held-out Test 분리 (15%) =====================
# ★ 이 seed와 test_size는 3개 코드 모두 동일해야 함 ★
indices = list(range(len(all_texts)))
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_seed"],
    stratify=all_labels
)

train_val_texts = [all_texts[i] for i in train_val_idx]
train_val_labels = [all_labels[i] for i in train_val_idx]
test_texts = [all_texts[i] for i in test_idx]
test_labels = [all_labels[i] for i in test_idx]

print(f"  Train+Val: {len(train_val_texts)}건 / Test: {len(test_texts)}건")

# ===================== 토크나이저 미리 로드 =====================
print("  토크나이저 로드 중...")
tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
print("  토크나이저 로드 완료!")

# ===================== Dataset (개별 토크나이징) =====================
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
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ===================== 학습/평가 함수 (진행바 포함) =====================
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

# ===================== CV 실험 =====================
def run_cv(condition_name):
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name.upper()}")
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

        print(f"    Train: {len(tr_texts)}건 / Val: {len(va_texts)}건")

        # Class weight
        loss_fct = None
        if condition_name == "class_weight":
            n_neg = tr_labels.count(0)
            n_pos = tr_labels.count(1)
            weight_pos = n_neg / n_pos if n_pos > 0 else 1.0
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor([1.0, weight_pos]).to(device))
            if fold_idx == 0:
                print(f"    Class weight: [1.0, {weight_pos:.3f}]")

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
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                         loss_fct, epoch+1)
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

    return fold_results, all_epoch_results

# ===================== Held-out Test =====================
def run_test(condition_name):
    print(f"\n  Test: {condition_name}")

    tr_texts = list(train_val_texts)
    tr_labels = list(train_val_labels)

    loss_fct = None
    if condition_name == "class_weight":
        n_neg = tr_labels.count(0)
        n_pos = tr_labels.count(1)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, n_neg/n_pos]).to(device))

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
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                                     loss_fct, epoch+1)
        print(f"    Test Train Epoch {epoch+1}: loss={train_loss:.4f}")

    metrics = evaluate(model, test_loader)
    print(f"    ✅ Test Result: F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f}")
    del model; torch.cuda.empty_cache()
    return metrics

# ===================== 실행 =====================
conditions = ["base", "class_weight"]
cv_results = {}
epoch_data = {}

for cond in conditions:
    fold_res, epoch_res = run_cv(cond)
    cv_results[cond] = fold_res
    epoch_data[cond] = epoch_res

    # 조건별 중간저장
    interim = {cond: {"folds": fold_res,
               "f1_mean": float(np.mean([r["f1"] for r in fold_res]))}}
    with open(f"{DRIVE_FOLDER}/review_result_base_interim.json", "w") as f:
        json.dump(interim, f, indent=2, default=str)
    print(f"\n  💾 {cond} 중간저장 완료")

# 통계 검정 (Base vs Class Weight)
base_f1 = [r["f1"] for r in cv_results["base"]]
cw_f1 = [r["f1"] for r in cv_results["class_weight"]]
t_stat, t_pval = stats.ttest_rel(cw_f1, base_f1)
try:
    w_stat, w_pval = stats.wilcoxon(cw_f1, base_f1)
except: w_pval = None
diff = np.array(cw_f1) - np.array(base_f1)
cohens_d = float(np.mean(diff)/np.std(diff)) if np.std(diff)>0 else 0

# Held-out Test
test_results = {}
for cond in conditions:
    test_results[cond] = run_test(cond)

# 저장
output = {
    "config": CONFIG, "timestamp": datetime.now().isoformat(),
    "cv_results": {c: {"folds": cv_results[c],
                       "f1_mean": float(np.mean([r["f1"] for r in cv_results[c]])),
                       "f1_std": float(np.std([r["f1"] for r in cv_results[c]]))}
                  for c in conditions},
    "epoch_data": epoch_data,
    "test_results": test_results,
    "stat_test_base_vs_cw": {"t_pval": float(t_pval), "cohens_d": cohens_d,
                              "w_pval": float(w_pval) if w_pval else None},
}
with open(f"{DRIVE_FOLDER}/review_result_base.json", "w") as f:
    json.dump(output, f, indent=2, default=str)

# 요약
print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
for c in conditions:
    f1s = [r["f1"] for r in cv_results[c]]
    print(f"  {c:<15} CV F1: {np.mean(f1s):.4f}±{np.std(f1s):.4f}  "
          f"Test F1: {test_results[c]['f1']:.4f}")
print(f"\n  Base vs CW: p={t_pval:.6f}, d={cohens_d:.3f}")
print(f"  저장: {DRIVE_FOLDER}/review_result_base.json")
