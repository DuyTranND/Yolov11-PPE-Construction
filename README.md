# 🦺 PPE Detection System

A production-ready Construction Site Personal Protective Equipment (PPE) detection system with a FastAPI backend and a Gradio frontend. Supports image upload, live webcam, and video file processing.

<p align="center">
  <img src="docs/pipeline_.png" alt="Pipeline" width="85%"><br>
  <em>Camera/Video -> frames -> YOLOv11 PPE detector -> annotated images + JSON via FastAPI -> Gradio UI</em>
</p>

---

## Demo

[▶ Watch the demo on YouTube](https://youtu.be/DGafnpOU2g8)

---

## Detected PPE Classes

- `helmet` / `no-helmet`
- `vest` / `no-vest`
- `gloves` 
- `boots`
- `person`

## Project Structure

```
YoloV11-PPE-Construction/
├── docs/
│ └── pipeline_.png                       # Pipeline diagram
├── models/
│ └── yolo11m_ppe_best.pt                 # YOLOv11 weights
├── src/
│ ├── init.py
│ ├── main.py                             # FastAPI backend
│ ├── processing.py                       # Core detection + drawing utilities
│ ├── video_processing.py                 # realtime/video helpers
│ └── ui.py                               # Gradio frontend
├── train/
│ └── yolov11_ppe_construction.ipynb      # training scripts
├── .gitignore
├── .python-version
├── pyproject.toml                        # Project metadata and dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.9 or higher
- `uv` package manager (recommended) or `pip`

### Install with uv (Recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install project dependencies
uv pip install -e .

# Or install from pyproject.toml directly
uv pip sync
```

### Install with pip

```bash
pip install -e .
```

## Usage

### 1. Start the FastAPI Backend

Open a terminal and run:

```bash
uvicorn src.main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`

**API Documentation**: Visit `http://127.0.0.1:8000/docs` for interactive Swagger UI

### 2. Start the Gradio Frontend

Open another terminal and run:

```bash
python src/ui.py
```

The Gradio interface will launch in your browser (typically at `http://127.0.0.1:7860`)

## API Endpoints

### POST /detect/

Detects PPE in an uploaded image.

## License

MIT License