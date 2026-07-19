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
├── docs/related_work.md
├── data/{raw,corrupted,embeddings,features}/
├── models/
├── results/{figures,metrics}/
├── scripts/
│   ├── generate_corrupted_dataset.py
│   ├── build_trust_labels.py
│   ├── train_trust_module.py
│   ├── evaluate.py
│   ├── compare_baselines.py
│   ├── run_ablation.py
│   ├── report_by_corruption.py
│   ├── plot_corruption_summary.py
│   └── demo_trust.py
├── src/
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

## Experiment Workflow

```bash
# 1) Corrupt images
python scripts/generate_corrupted_dataset.py

# 2) ArcFace labels + quality features
python scripts/build_trust_labels.py --ctx-id -1

# 3) Train Trust Score (identity-disjoint; quality + similarity)
python scripts/train_trust_module.py --features quality_sim --model-type xgboost

# 4) FIQA-style eval (Risk-Coverage, ERC, FAR/FRR, fixed reject rates)
python scripts/evaluate.py --threshold 0.99

# 5) One-command feature ablation (quality / sim_only / quality_sim)
python scripts/run_ablation.py

# 6) Per-corruption breakdown
python scripts/report_by_corruption.py

# 7) Per-corruption summary figure (baseline vs trust-gated accuracy)
python scripts/plot_corruption_summary.py --reject-rate-label 20%

# 8) Single-image demo (builds/loads gallery cache on first run)
python scripts/demo_trust.py data/raw/lfw-deepfunneled/lfw-deepfunneled/Aaron_Eckhart/Aaron_Eckhart_0001.jpg \
  --threshold 0.6 --ctx-id -1
```

## Results snapshot (LFW + corruptions, quality_sim model)

| Quantity | Value |
|---|---|
| Baseline ArcFace accuracy | **92.2%** |
| Trust ROC-AUC (identity-disjoint test) | **~0.95** |
| EER (trust accept/reject) | **~3.3%** (thr ≈ 0.60) |
| Gate `--threshold 0.99` | **~99.9%** acc on accepted set |
| ERC @ 10% reject | **~99.7%** retained accuracy |
| vs similarity-only @ 10% reject | ~99.0% (trust / fused better) |

Plots: `results/figures/trust_eval_*.png`, `baseline_compare_erc.png`, `ablation_erc.png`, `corruption_summary.png`.

### Limitations (rain)

Trust gating recovers near-ceiling accuracy under blur, JPEG, fog, low-light, and noise (e.g. Gaussian noise: **87.8% → 99.7%** retained at 20% reject). **Rain is the failure case**: baseline accuracy collapses to ~31.7% and only reaches ~39.7% even after rejecting 20%. The trust model *does* assign near-zero scores to almost all rain images (so they are rejected first in the global pool — the desired behavior), but the synthetic rain streaks destroy so much identity information that recognition cannot be salvaged within the rain subset. Softening the rain severity (fewer/thinner streaks) is left as future work.

## Related work

See [docs/related_work.md](docs/related_work.md) for FaceQnet, MagFace, SER-FIQ, SDD-FIQA, SelectiveNet, and how this project differs.

## Status

- [x] Corruption suite + LFW run
- [x] Quality features + ArcFace labeling
- [x] Trust predictor + identity-disjoint / balanced training
- [x] ERC, Risk–Coverage, FAR/FRR/EER, fixed reject rates
- [x] Feature ablation CLI
- [x] Per-corruption reporting
- [x] Single-image demo

## License

Research / academic use. Cite InsightFace and the LFW dataset papers when publishing results.
