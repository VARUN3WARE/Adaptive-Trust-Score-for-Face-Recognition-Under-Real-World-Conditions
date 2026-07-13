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
5. **Trust Score Predictor** — lightweight classifier → P(correct recognition)

## Project Layout

```
.
├── configs/default.yaml
├── data/{raw,corrupted,embeddings,features}/
├── models/
├── notebooks/
├── results/{figures,metrics}/
├── scripts/
│   ├── generate_corrupted_dataset.py
│   ├── build_trust_labels.py
│   ├── train_trust_module.py
│   └── evaluate.py
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

### Dataset layout (already configured)

This repo expects the [Kaggle LFW dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset) extracted under `data/raw/`:

```
data/raw/
  lfw-deepfunneled/lfw-deepfunneled/<Person_Name>/*.jpg
  lfw_allnames.csv
  pairs.csv
  people.csv
  ...
```

`configs/default.yaml` points `raw_dir` at `data/raw/lfw-deepfunneled/lfw-deepfunneled`.

## Experiment Workflow

```bash
# 1) Artificially degrade images (blur, JPEG, rain/fog, low-light, noise)
#    Full LFW (~13k images) can take a while; use --limit via a subset if needed.
python scripts/generate_corrupted_dataset.py

# 2) Run ArcFace, extract quality features, label correct/incorrect
python scripts/build_trust_labels.py --ctx-id -1   # CPU; use 0 for GPU

# 3) Train Trust Score classifier (xgboost | random_forest | mlp)
python scripts/train_trust_module.py --model-type xgboost

# 4) Biometric evaluation: Risk-Coverage, FAR/FRR/EER
python scripts/evaluate.py
```

Outputs land in `results/figures/` and `results/metrics/`.

## Evaluation Metrics

- ArcFace accuracy on the corrupted set (baseline)
- ArcFace + Trust gating accuracy on the accepted subset
- Risk-Coverage curve (accuracy vs fraction retained)
- FAR / FRR / EER of the trust accept/reject decision


## License

Research / academic use. Cite InsightFace and the LFW / CelebA dataset papers when publishing results.
