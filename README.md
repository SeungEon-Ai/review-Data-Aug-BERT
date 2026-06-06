# App Review Sentiment Classification with Data Augmentation

앱 리뷰 감성 분류에서 텍스트 데이터 증강 기법(EDA, LLM 기반 생성)이 BERT 분류 성능에 미치는 영향을 비교한 실험 코드와 결과입니다.

Experiment code and results comparing the effect of text data augmentation methods (EDA, LLM-based generation) on BERT classification performance for app review sentiment analysis.

---

## 개요 / Overview

Twitter, Instagram, Facebook 앱 리뷰(9,419건)를 대상으로 네 가지 조건을 동일한 설정에서 비교합니다: 원본만 사용, 클래스 가중치, EDA 증강, GPT-4o-mini 증강. 모든 실험은 10-fold 교차검증과 별도의 held-out test set으로 평가하며, fold별 표준편차와 통계 검정(paired t-test, Wilcoxon)을 함께 보고합니다.

Four conditions are compared under identical settings using app reviews (9,419 samples): original only, class weighting, EDA augmentation, and GPT-4o-mini augmentation. All experiments use 10-fold cross-validation with a separate held-out test set, reporting per-fold standard deviation and statistical tests.

증강 데이터는 학습 fold에만 적용하고 검증 fold는 원본만 사용하여 데이터 누수를 방지했습니다. 보고 성능은 각 fold의 best validation F1 epoch을 기준으로 합니다.

Augmented data is applied only to training folds to prevent data leakage. Reported performance is based on the best validation F1 epoch per fold.

---

## 결과 / Results

### Cross-Validation (10-fold, F1, mean ± SD)

| Condition | F1 | Accuracy |
|-----------|-----|----------|
| Base | 0.8641 ± 0.0119 | 0.8753 |
| Class Weight | 0.8622 ± 0.0085 | 0.8727 |
| EDA | 0.8580 ± 0.0115 | 0.8692 |
| GPT-4o-mini | 0.8603 ± 0.0100 | 0.8721 |

### Held-out Test (F1)

| Condition | F1 | Accuracy |
|-----------|-----|----------|
| Base | 0.8816 | 0.8924 |
| Class Weight | 0.8738 | 0.8832 |
| GPT-4o-mini | 0.8592 | 0.8726 |
| EDA | 0.8541 | 0.8662 |

이 데이터셋(클래스 비율 54:46)에서는 데이터 증강이 baseline 대비 유의한 성능 향상을 보이지 않았습니다. 자세한 수치는 results의 JSON 파일을 참고하세요.

For this dataset (class ratio 54:46), data augmentation did not yield significant improvement over the baseline. See the JSON files in `results/` for detailed metrics.

---

## 파일 구성 / Files

| 파일 | 설명 |
|------|------|
| `review_01_base.py` | 원본 학습 + 클래스 가중치 (Base + Class Weight) |
| `review_02_eda.py` | EDA 4배 증강 학습 (SR/RI/RS/RD) |
| `review_03_gpt.py` | GPT-4o-mini 증강 학습 |
| `review_04_compare.py` | 결과 통합 및 통계 검정 |
| `review_05_heldout_best.py` | Best epoch 기준 held-out test 평가 |
| `results/` | 실험 결과 JSON |

---

## 실험 설정 / Settings

- Model: `bert-base-uncased`
- Max sequence length: 256
- Epochs: 10 (best validation F1 epoch 선택)
- Batch size: 32
- Learning rate: 2e-5
- Optimizer: AdamW, weight decay 0.01
- Cross-validation: Stratified 10-fold
- Held-out test: 15% (stratified)
- Random seed: 42

GPT-4o-mini 증강은 temperature 0.9, top-p 0.95, seed 42로 원본 1건당 4개의 paraphrase를 생성했습니다.

---

## 실행 / Usage

```bash
pip install torch transformers scikit-learn scipy tqdm pandas

python review_01_base.py
python review_02_eda.py
python review_03_gpt.py
python review_04_compare.py
python review_05_heldout_best.py
```

코드 상단의 데이터 폴더 경로를 환경에 맞게 수정한 뒤 실행합니다.

---

## 데이터 / Data

데이터 파일은 용량 및 라이선스 문제로 포함하지 않았습니다. 코드는 다음 형식의 CSV를 입력으로 사용합니다.

- `content`: 리뷰 본문
- `label`: 0 (부정) / 1 (긍정)
- `SR`, `RI`, `RS`, `RD`: EDA 증강 텍스트 (각 연산별)

---
