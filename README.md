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
├── configs/                 # Experiment hyperparameters
├── data/
│   ├── raw/                 # Pristine LFW / CelebA (gitignored)
│   ├── corrupted/           # Artificially degraded images (gitignored)
│   ├── embeddings/          # Cached ArcFace embeddings (gitignored)
│   └── features/            # Quality feature tables (gitignored)
├── models/                  # Trained trust predictors
├── notebooks/               # Exploratory analysis
├── results/                 # Metrics, Risk-Coverage curves
├── scripts/                 # CLI entrypoints for each experiment stage
├── src/
│   ├── data_corruption.py   # Gaussian blur, JPEG, rain/fog, low-light, noise
│   ├── feature_extraction.py
│   ├── face_pipeline.py
│   ├── trust_predictor.py
│   └── evaluation.py
├── requirements.txt
└── README.md
```

## Experimental Protocol

1. Start from LFW (or a CelebA subset).
2. Corrupt images (blur, JPEG, rain/fog, low light, Gaussian noise).
3. Run ArcFace on pristine + corrupted pairs; label `1` if prediction matches ground truth, else `0`.
4. Train a Random Forest / XGBoost / small MLP on quality features → Trust Score.
5. Evaluate like a biometrics paper: accuracy, Risk-Coverage, FAR / FRR / EER.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status

- [x] Project scaffold
- [x] Data corruption module
- [ ] Quality feature extraction + ArcFace integration
- [ ] Trust predictor training
- [ ] Biometric evaluation (Risk-Coverage, FAR/FRR/EER)

## License

Research / academic use. Cite InsightFace and the LFW / CelebA dataset papers when publishing results.
