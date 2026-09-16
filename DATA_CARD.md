# Data Card — TrashNet

## Source
VisionSort AI uses the public **TrashNet** image dataset created by Gary Thung and Mindy Yang. The project downloads `dataset-resized.zip` from the `garythung/trashnet` dataset repository on Hugging Face and does not redistribute the images.

Classes used:
- cardboard
- glass
- metal
- paper
- plastic
- trash

The original TrashNet project documents 2,527 images collected against simple backgrounds. The Hugging Face mirror should be inspected at experiment time because mirrors/conversions can change independently of this repository.

## Split policy
Images are split independently inside each class using a deterministic seed. The default split is 70% train, 15% validation, 15% test. The held-out test set is used only after model selection.

## Intended use
Educational research and portfolio demonstration of image classification, transfer learning, evaluation, and explainability.

## Known limitations
- TrashNet is small relative to modern computer-vision datasets.
- Many images were captured against relatively clean backgrounds, so real-world clutter may reduce performance.
- Class balance is uneven, especially the `trash` category.
- The six labels do not cover all municipal recycling rules or material types.
- Dataset class labels should not be interpreted as local disposal instructions.

## Responsible use
This system should not make high-stakes environmental or industrial sorting decisions. A real deployment would require locally collected data, broader materials, domain-specific labeling, calibration, monitoring, and physical-sorting validation.
