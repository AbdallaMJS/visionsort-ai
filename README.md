# ♻️ VisionSort AI

Explainable computer vision for recyclable-material classification using the **TrashNet** dataset.

The project is designed to demonstrate a full deep-learning workflow rather than a single accuracy number: dataset acquisition, deterministic class-aware splitting, image augmentation, a small CNN baseline, ImageNet transfer learning, validation-based checkpointing, macro-F1 evaluation, confusion-matrix analysis, and Grad-CAM visual explanations.

> **Status:** training code is ready; numerical results are intentionally withheld until an actual verified training run is completed.

## Research question
How much does transfer learning help on a relatively small waste-image dataset, and where does the resulting model still fail?

## Dataset
The repository downloads `dataset-resized.zip` from the public `garythung/trashnet` dataset repository on Hugging Face. Images are not copied into this repository. See `DATA_CARD.md` for provenance, class definitions, split policy, and limitations.

Target classes:
- cardboard
- glass
- metal
- paper
- plastic
- trash

## Models
### Small CNN baseline
A compact convolutional model trained from scratch establishes a lower-complexity reference point.

### MobileNetV3-Small transfer model
The main model starts from ImageNet-pretrained MobileNetV3-Small features and replaces the final classifier with a six-class output layer.

## Evaluation
The training pipeline records:
- accuracy,
- macro-F1,
- per-class precision/recall/F1,
- confusion matrix,
- train/validation loss and macro-F1 by epoch.

Macro-F1 is highlighted because TrashNet is not perfectly class balanced.

## Explainability
`visionsort.explain.GradCAM` generates a class-specific spatial heat map from the model's final feature block. The purpose is to inspect whether the network attends to plausible regions and to identify suspicious shortcuts. Grad-CAM is not treated as proof that the model reasons like a person.

## Project structure
```text
src/visionsort/
  data.py       dataset download, discovery, class-aware split
  model.py      preprocessing, small CNN, MobileNet transfer model
  train.py      training, validation, held-out test evaluation
  explain.py    Grad-CAM utility
scripts/
  train.py
app.py          simple Streamlit inference demo
tests/          deterministic data-pipeline tests
DATA_CARD.md
MODEL_CARD.md
.github/workflows/ci.yml
```

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[train,dev]"
python scripts/train.py --architecture cnn --epochs 8
python scripts/train.py --architecture mobilenet --epochs 8
```

After verified training:
```bash
streamlit run app.py
```

Generated images, downloaded datasets, and model weights are ignored by Git.

## Experimental discipline
The validation set is used for checkpoint selection. The held-out test set is evaluated only after the best validation checkpoint is selected. Results for the baseline CNN and transfer model should be reported together so the repository demonstrates comparison rather than cherry-picking.

## Responsible interpretation
This is an educational image-classification prototype, **not a municipal recycling decision system**. TrashNet is small and visually constrained. Real-world scenes may contain clutter, multiple objects, unseen materials, local disposal rules, or ambiguous categories. A production system would need broader local data, calibration, monitoring, and physical-system validation.

## Next milestone
Run both architectures, verify CI, add the genuine comparison table, inspect the confusion matrix, generate several correct and incorrect Grad-CAM examples, and write a short error-analysis section based on those real outputs.
