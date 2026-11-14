# AURELIS

**AI-Powered Pneumonia Screening Research Prototype**  
*See Earlier. Understand Better.*

AURELIS is an educational computer-vision project for studying binary chest X-ray image classification (NORMAL vs PNEUMONIA) using transfer learning with ResNet-18. It includes a training script, a held-out test evaluation script, a simple Streamlit demonstration, and a comparison against an always-predict-PNEUMONIA baseline.

> **This project is a prototype and is not medical advice.**

## Intended research user

A hypothetical healthcare-imaging researcher or student exploring how image-classification models behave on a public benchmark dataset. The interface is for controlled demonstration only—not for patient care, diagnosis, screening, or clinical prioritization.

## Repository structure

```text
AURELIS/
├── app.py
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
├── src/
│   ├── config.py
│   ├── data.py
│   ├── inference.py
│   ├── metrics.py
│   └── model.py
├── tests/
├── docs/
├── notebooks/
├── models/       # trained weights are not committed
└── outputs/      # generated evaluation reports are not committed
```

## Dataset setup

This repository does **not** include the dataset. Obtain the public Kaggle Chest X-Ray Images (Pneumonia) dataset yourself and review its current terms and provenance. Extract it so the folders look like this:

```text
data/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

The code creates a stratified validation split from `train/`. The supplied `test/` split is reserved for final evaluation and must not be used for model selection. Confirm class labels, duplicate images, patient-level separation, and dataset licensing before interpreting results.

## Install

Python 3.10 or 3.11 is recommended. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your machine from the official PyTorch installation selector if the default installation does not match your CPU/CUDA environment.

## Run

```bash
# Train (GPU recommended; downloads pretrained ResNet-18 weights on first run)
python train.py --epochs 8

# Evaluate once on the held-out test split
python evaluate.py

# Run one image through the research model
python predict.py path/to/xray.jpeg

# Launch the demo UI
streamlit run app.py
```

Training writes `models/best_model.pt` and `models/metadata.json`. Evaluation writes `outputs/test_metrics.json`, including confusion-matrix counts, false positives, false negatives, sensitivity/recall, specificity, and the baseline comparison. These files are generated locally and are intentionally excluded from Git.

## Reproducibility and reporting

Record the dataset version, exact split, environment, random seed, model checkpoint, and all test metrics. Do not tune thresholds or model settings on the held-out test set. Report false positives and false negatives separately. A single benchmark score is not evidence of clinical performance.

## Current implementation status

- Implemented: ResNet-18 training scaffold, stratified train/validation split, held-out test evaluation, baseline comparison, command-line prediction, and Streamlit demo.
- Not implemented: Grad-CAM, external-dataset evaluation, calibrated uncertainty estimation, robust image-quality validation, prospective testing, and clinical validation.
- No real training run or real test metrics are included in this repository at this stage.

## Safety and limitations

AURELIS is not a medical device, diagnostic system, or substitute for a qualified clinician. Never use its output to make patient-care or triage decisions. Model confidence is not a calibrated probability. Public benchmark datasets may contain label noise, duplicates, source-specific patterns, or data leakage, and performance may not generalize to other hospitals, scanners, age groups, or populations. Any future clinical research requires independent validation, subgroup analysis, calibration, ethics review, and applicable regulatory oversight.

Do not commit patient images, personal data, Kaggle credentials, API tokens, secrets, or trained model weights to this public repository.

## License

MIT License. See [LICENSE](LICENSE).
