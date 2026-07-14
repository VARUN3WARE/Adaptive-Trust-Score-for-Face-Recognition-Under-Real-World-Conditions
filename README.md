# Adaptive Trust Score for Face Recognition Under Real-World Conditions

Research-grade pipeline that answers: **Can we predict when a face recognition model is likely to be wrong before it makes a high-confidence mistake?**

Instead of identity alone, the system outputs a **Trust Score** — the estimated probability that the recognition decision is correct under the observed image conditions.

## Research Question

Face recognition systems (e.g. ArcFace) often remain overconfident on blurry, poorly lit, or pose-challenged images. This project trains a secondary **Quality Assessment / Trust Predictor** on handcrafted image-quality features so we can reject low-trust predictions before they become false accepts or false rejects.

## System Pipeline

1. **Input** — Face image (pristine or corrupted)
2. **Face detection** — RetinaFace via InsightFace
3. **Feature extraction** — ArcFace embedding → identity prediction
4. **Quality Assessment Module** — blur, brightness/contrast, head pose, face size, detection confidence, entropy
5. **Trust Score Predictor** — lightweight classifier → P(correct recognition), optionally fused with match similarity

## Project Layout

```
.
├── configs/default.yaml
├── data/{raw,corrupted,embeddings,features}/
├── models/
├── results/{figures,metrics}/
├── scripts/
│   ├── generate_corrupted_dataset.py
│   ├── build_trust_labels.py
│   ├── train_trust_module.py
│   ├── evaluate.py
│   ├── compare_baselines.py
│   └── report_by_corruption.py
├── src/
│   ├── data_corruption.py
│   ├── feature_extraction.py
│   ├── face_pipeline.py
│   ├── trust_predictor.py
│   ├── evaluation.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset layout

[Kaggle LFW](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) deepfunneled under `data/raw/`:

```
data/raw/lfw-deepfunneled/lfw-deepfunneled/<Person_Name>/*.jpg
```

`configs/default.yaml` already points `raw_dir` there.

## Experiment Workflow

```bash
# 1) Corrupt images (blur, JPEG, rain/fog, low-light, noise)
python scripts/generate_corrupted_dataset.py

# 2) ArcFace + quality features + correctness labels
python scripts/build_trust_labels.py --ctx-id -1   # CPU; use 0 for GPU

# 3) Train Trust Score (identity-disjoint split; balanced classes)
python scripts/train_trust_module.py --model-type xgboost --features quality_sim

# 4) FIQA-style evaluation: Risk-Coverage, ERC, FAR/FRR/EER, fixed reject rates
python scripts/evaluate.py --threshold 0.99

# 5) Baselines: similarity-only vs quality-trust vs quality+similarity
python scripts/compare_baselines.py

# 6) Per-corruption / pristine breakdown
python scripts/report_by_corruption.py
```

Outputs: `results/figures/` and `results/metrics/`.

## Evaluation Metrics (paper-style)

| Metric | Meaning |
|---|---|
| Baseline ArcFace accuracy | Ungated identification accuracy |
| Risk–Coverage | Accuracy vs fraction of images accepted ([SelectiveNet](http://proceedings.mlr.press/v97/geifman19a.html)-style) |
| **Error-vs-Reject (ERC)** | Error vs % rejected — [FaceQnet](https://ar5iv.labs.arxiv.org/html/2006.03298) / NIST FRVT-QA style |
| Fixed reject rates | Acc/error at 10% / 20% / 30% / 40% reject |
| FAR / FRR / EER | Trust accept/reject decision errors |
| Baseline compare | similarity-only vs quality vs quality+similarity |

## Results snapshot (LFW deepfunneled + corruptions)

From the full labeled table (~26.5k pristine+corrupted probes):

| Quantity | Value |
|---|---|
| Baseline ArcFace accuracy | ~92.2% |
| Risk–Coverage | ~**100% accuracy at ~78% coverage** (reject lowest-trust) |
| Practical gate | `--threshold 0.99` → ~60% coverage, ~**99.8%** acc on accepted |
| EER (trust decision) | ~8.8% |

Re-train after pulling feature updates:

```bash
python scripts/train_trust_module.py --features quality_sim --model-type xgboost
python scripts/evaluate.py --threshold 0.99
python scripts/compare_baselines.py
python scripts/report_by_corruption.py
```

## Related work

- Hernandez-Ortega et al., **FaceQnet** — quality as predicted recognition utility  
- Meng et al., **MagFace** (CVPR 2021) — embedding magnitude as quality  
- Terhörst et al., **SER-FIQ** — stochastic embedding robustness  
- Ou et al., **SDD-FIQA** (CVPR 2021) — similarity distribution distance  
- Geifman & El-Yaniv, **SelectiveNet** / risk–coverage selective classification  

## Status

- [x] Corruption suite + LFW run
- [x] Quality features + ArcFace labeling
- [x] Trust predictor (RF / XGBoost / MLP)
- [x] ERC, Risk–Coverage, FAR/FRR/EER
- [x] Similarity baselines + identity-disjoint / balanced training
- [x] Per-corruption reporting

## License

Research / academic use. Cite InsightFace and the LFW dataset papers when publishing results.
