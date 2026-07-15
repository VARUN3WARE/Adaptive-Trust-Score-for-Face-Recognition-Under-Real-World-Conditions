# Related Work

Pointers for FIQA (face image quality assessment) and selective prediction.
This project’s Trust Score is a lightweight, accuracy-supervised rejector for ArcFace under synthetic corruptions.

## Face image quality / utility prediction

| Paper | Idea | Relevance |
|---|---|---|
| [FaceQnet](https://ar5iv.labs.arxiv.org/html/2006.03298) (Hernandez-Ortega et al.) | Deep quality score trained to predict recognition utility | Closest framing: quality ≈ predicted accuracy |
| [MagFace](https://ar5iv.labs.arxiv.org/html/2103.06627) (Meng et al., CVPR 2021) | Embedding magnitude as quality | Strong FIQA baseline; ERC evaluation |
| [SER-FIQ](https://openaccess.thecvf.com/content_CVPR_2020/html/Terhorst_SER-FIQ_Unsupervised_Estimation_of_Face_Image_Quality_Based_on_Stochastic_CVPR_2020_paper.html) (Terhörst et al., CVPR 2020) | Stochastic embedding robustness | Unsupervised quality from recognition model |
| [SDD-FIQA](https://openaccess.thecvf.com/content/CVPR2021/html/Ou_SDD-FIQA_Unsupervised_Face_Image_Quality_Assessment_With_Similarity_Distribution_Distance_CVPR_2021_paper.html) (Ou et al., CVPR 2021) | Intra-/inter-class similarity distributions | Unsupervised pseudo-labels for quality |
| [PCNet](https://ar5iv.labs.arxiv.org/html/2009.00603) (Xie et al.) | Predictive confidence from mated pairs | Error-vs-reject for verification |
| [ACM FIQA survey](https://dl.acm.org/doi/10.1145/3507901) (Schlett et al.) | Taxonomy of FIQA methods | Survey context / utility definition |

## Selective prediction / reject option

| Paper | Idea | Relevance |
|---|---|---|
| [SelectiveNet](http://proceedings.mlr.press/v97/geifman19a.html) (Geifman & El-Yaniv, ICML 2019) | Joint classify + reject; risk–coverage | Metric framing used here |
| Geifman & El-Yaniv (2017) | Selective classification with coverage guarantees | Classical selective prediction |
| [ConfidNet](https://proceedings.neurips.cc/paper/2019/hash/757f843a169cc678064d9530d12a1881-Abstract.html) (Corbière et al., NeurIPS 2019) | Learn true-class probability for failure prediction | Failure prediction motivation |

## How this project differs

1. **Handcrafted quality cues** (blur, pose, lighting, entropy, …) plus optional ArcFace similarity — interpretable, fast secondary model.
2. **Corruption stress test** on LFW (blur / JPEG / rain / fog / low-light / noise) with labeled ArcFace success/failure.
3. **Paper-style reporting:** Error-vs-Reject (ERC), Risk–Coverage, fixed reject rates, and feature ablations (`quality` vs `sim_only` vs `quality_sim`).

We do **not** retrain MagFace/SDD-FIQA from scratch; those remain external SOTA baselines you can cite or plug in later.
