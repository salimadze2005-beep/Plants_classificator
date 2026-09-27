# 🌱 Plants Verification Size (PvZ)

**Computer Vision / Applied ML project for automated plant analysis.**

PvZ recognizes the plant type (**wheat / arugula**), segments plant parts and converts segmentation masks into practical biometric measurements such as area and estimated size.

> Portfolio focus: end-to-end CV pipeline, model training and validation, segmentation, image post-processing, metric calculation and integration into a FastAPI application.

## What the project does

Manual plant measurements are slow and difficult to scale. PvZ automates the main workflow:

1. Upload a plant image through the web interface.
2. Classify the image as wheat or arugula.
3. Select the corresponding segmentation model.
4. Segment plant parts such as root, stem and leaf.
5. Calculate area and approximate linear dimensions from masks.
6. Save, visualize and export the analysis result.

## My role — ML / Computer Vision Engineer

The project was developed by a team of three. I was responsible for the **ML/CV part**:

- built the **classification → model selection → segmentation** pipeline;
- trained and evaluated **YOLO** and **U-Net** segmentation models;
- implemented inference and post-processing logic;
- calculated biometric metrics from segmentation masks;
- worked with pixel-to-millimeter conversion and calibration logic;
- validated models using **Precision, Recall, mAP and IoU/mIoU**;
- integrated the CV pipeline with the backend.

This repository demonstrates the complete applied CV workflow: **model preparation → training → validation → error analysis → inference → integration into an application**.

## Model results

### YOLO segmentation

| Metric | Wheat | Arugula |
|---|---:|---:|
| Box Precision | 0.873 | 0.591 |
| Box Recall | 0.875 | 0.557 |
| Box mAP@50 | 0.897 | 0.558 |
| Mask Precision | 0.836 | 0.457 |
| Mask Recall | 0.839 | 0.420 |
| Mask mAP@50 | **0.826** | **0.409** |
| Mask mAP@50–95 | 0.479 | 0.206 |

### U-Net segmentation

| Metric / class | Wheat | Arugula |
|---|---:|---:|
| Best mIoU | **0.8124** | **0.6083** |
| Background IoU | 0.994 | 0.992 |
| Root IoU | 0.742 | 0.276 |
| Stem IoU | 0.881 | 0.614 |
| Leaf IoU | 0.635 | 0.417 |

The metrics show an important practical detail: segmentation quality differs significantly by plant type and class, so the system keeps model-specific processing and explicit validation metrics instead of treating inference as a black box.

## Architecture

```text
Image
  │
  ▼
Plant classifier
  │
  ├── Wheat ─────► Wheat segmentation model
  │
  └── Arugula ───► Arugula segmentation model
                       │
                       ▼
              Mask post-processing
                       │
                       ▼
             Biometric calculations
                       │
                       ▼
          Visualization / DB / export
```

The backend supports YOLO-based segmentation, U-Net segmentation and a combined processing mode.

## Tech stack

**ML / Computer Vision**
- Python
- PyTorch
- Ultralytics YOLO
- U-Net via `segmentation-models-pytorch`
- OpenCV
- NumPy
- Albumentations

**Backend / Application**
- FastAPI
- Uvicorn
- Jinja2
- SQLite
- HTML / CSS / JavaScript

**Engineering**
- Git / Git LFS
- modular inference pipeline
- image post-processing
- CSV export
- analysis history in SQLite

## Repository structure

```text
.
├── app/
│   ├── main.py              # FastAPI routes and application flow
│   ├── database.py          # SQLite persistence
│   ├── templates/           # Web UI templates
│   └── static/              # CSS and JavaScript
│
├── core/
│   ├── models.py            # Model loading and classification
│   ├── ml.py                # Segmentation / inference pipeline
│   ├── metrics.py           # Area and size calculations
│   ├── calibrate.py         # Camera / scale calibration experiments
│   ├── constants.py         # Classes, mappings and scale parameters
│   └── utils.py             # Image processing and visualization helpers
│
├── models/                  # Model references / Git LFS pointers
├── requirements.txt
└── README.md
```

## Local run

### 1. Clone

```bash
git clone https://github.com/salimadze2005-beep/Plants_classificator.git
cd Plants_classificator
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model weights

The source project contains **Git LFS pointer files** for the trained `.pt` / `.pth` models rather than the binary weight files themselves.

For inference, the corresponding trained weights must be available at the expected paths:

```text
models/classificator.pt
models/arugula.pt
models/wheat.pt
models/U-Net/rugola_v3_best.pth
models/U-Net/пшеница_4класса.pth
```

### 5. Start the application

```bash
uvicorn app.main:app --port 8000 --reload
```

Open:

```text
http://127.0.0.1:8000
```

## Why this project is relevant to CV/ML roles

The repository shows more than model training. It covers the path from **classification and segmentation to application-level inference**: routing between plant classes, multiple segmentation architectures, mask post-processing, real-world metric calculation, persistence and a web API/UI.

For a technical review, start with:
- [`core/ml.py`](core/ml.py) — inference pipeline;
- [`core/models.py`](core/models.py) — model loading and classification;
- [`core/metrics.py`](core/metrics.py) — biometric calculations.

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
