# Model Card — VisionSort AI

## Model families
The training pipeline supports two deliberately different architectures:

1. **Small CNN baseline** — a compact convolutional network trained from scratch.
2. **MobileNetV3-Small transfer model** — an ImageNet-pretrained backbone whose classifier is adapted to six TrashNet classes.

The comparison is intended to show whether transfer learning improves performance on a small image dataset.

## Metrics
The experiment records:
- overall accuracy,
- macro-F1,
- per-class precision/recall/F1,
- confusion matrix,
- training and validation loss/F1 by epoch.

Macro-F1 is emphasized because class counts are uneven.

## Explainability
`visionsort.explain.GradCAM` can generate a class-specific heat map from the final convolutional feature block of the transfer model. The heat map is evidence about spatial influence inside the network; it is not a proof of causal reasoning.

## Model selection
Validation macro-F1 is used to keep the best checkpoint during training. The held-out test set is evaluated once after training finishes.

## Current status
No numerical performance claims are made in this card until an actual training run has been completed and reviewed.

## Limitations
The model may fail on cluttered scenes, multiple objects, unusual camera angles, damaged materials, unseen categories, and material types not represented in TrashNet. Confidence scores can be overconfident and should not be interpreted as guaranteed correctness.
