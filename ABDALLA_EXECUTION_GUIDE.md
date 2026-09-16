# Abdalla Execution Guide — VisionSort AI

This file is a **student execution guide**, not evidence that training has already been completed. The branch was prepared with AI-assisted development support. Abdalla should personally run the training/evaluation, inspect the errors, and complete `PROJECT_JOURNAL.md` in his own words.

## Goal
Train and compare a small CNN with MobileNetV3-Small transfer learning on the same deterministic TrashNet split, then inspect class-level errors and model explanations.

## 1. Work on the execution branch
```bash
git clone https://github.com/AbdallaMJS/visionsort-ai.git
cd visionsort-ai
git checkout abdalla-execution
```

## 2. Create a clean environment and install
macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Then:
```bash
python -m pip install --upgrade pip
pip install -e ".[train,dev]"
```

## 3. Quality checks
```bash
ruff check src tests scripts app.py
pytest -q
```
Save the terminal output.

## 4. Train the CNN baseline
Use a separate artifacts directory so its results are not overwritten:
```bash
python scripts/train.py --architecture cnn --epochs 8 --artifacts artifacts/cnn
```
Record:
- device used;
- split sizes;
- test accuracy;
- test macro-F1;
- per-class precision/recall/F1;
- confusion matrix.

## 5. Train the MobileNet transfer model
```bash
python scripts/train.py --architecture mobilenet --epochs 8 --artifacts artifacts/mobilenet
```
Use the same split and compare against the CNN. Do not report only the better model.

## 6. Compare results
Create a compact comparison showing:
- CNN test accuracy and macro-F1;
- MobileNet test accuracy and macro-F1;
- strongest/weakest classes for each;
- largest confusion pairs;
- train-vs-validation behavior from each `history.json`.

If transfer learning does not improve the result, report that honestly.

## 7. Error analysis
Inspect representative wrong predictions. For each example, record:
- true class;
- predicted class;
- whether the object is cropped/cluttered/ambiguous;
- whether material appearance overlaps another class;
- any limitation that can be supported by the image.

Avoid inventing a cause when the evidence is unclear.

## 8. Grad-CAM review
Use the existing `visionsort.explain.GradCAM` utility on a small set of correct and incorrect predictions. Save representative heat maps and note whether the model appears to focus on the object or on irrelevant background regions.

Grad-CAM is a diagnostic visualization, not proof of human-like reasoning.

## 9. Run the reviewer demo
After training a model locally:
```bash
streamlit run app.py
```
Test several images and keep a screenshot if useful.

## 10. Final student-owned evidence
After the real runs, Abdalla should personally add or approve:
- completed `PROJECT_JOURNAL.md`;
- a genuine CNN vs MobileNet comparison table;
- confusion-matrix visualization(s);
- representative Grad-CAM examples;
- an error-analysis section based only on observed cases;
- README/model-card updates with verified metrics.

Suggested final student commit:
```text
Document verified VisionSort model comparison and error analysis
```

## Interview check
Abdalla should be able to answer without notes:
1. Why compare a CNN trained from scratch with transfer learning?
2. Why is macro-F1 useful for this dataset?
3. Why use the validation set for checkpoint selection?
4. What does the confusion matrix reveal that accuracy does not?
5. What can Grad-CAM show, and what can it not prove?
6. Why is TrashNet insufficient for a real municipal recycling system?
