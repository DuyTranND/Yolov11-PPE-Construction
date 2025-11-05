# 🦺 PPE Detection System

A production-ready Construction Site Personal Protective Equipment (PPE) detection system built with FastAPI backend and Gradio frontend.

<p align="center">

  <img src="pipeline_.png" alt="Pipeline" width="85%"><br>

  <em>Camera/Video -> frames -> YOLOv11 PPE detector -> annotated images + JSON via FastAPI -> Gradio UI</em>

</p>

## Features

- **FastAPI Backend**: RESTful API for PPE detection with image processing
- **Gradio Frontend**: User-friendly web interface for image upload and visualization
- **Modular Architecture**: Clean separation between API, processing logic, and UI
- **Mock AI Model**: Simulated YOLO-style detection for development/testing
- **Base64 Image Encoding**: Efficient image transfer between services

## Detected PPE Classes

- `helmet` / `no-helmet`
- `vest` / `no-vest`
- `gloves`
- `boots`
- `person`

## Project Structure

```
ppe-detection-system/
├── pyproject.toml          # Project dependencies and metadata
├── README.md               # This file
└── src/
    ├── main.py            # FastAPI application
    ├── processing.py      # Core detection and image processing logic
    └── app_gradio.py      # Gradio frontend application
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
python src/app_gradio.py
```

The Gradio interface will launch in your browser (typically at `http://127.0.0.1:7860`)

## API Endpoints

### POST /detect/

Detects PPE in an uploaded image.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `image` (file)

**Response:**
```json
{
  "detections": [
    {
      "label": "helmet",
      "confidence": 0.95,
      "bbox": [10, 20, 50, 60]
    },
    {
      "label": "vest",
      "confidence": 0.88,
      "bbox": [70, 100, 150, 200]
    }
  ],
  "processed_image": "base64_encoded_image_string..."
}
```

## Example Usage with cURL

```bash
curl -X POST "http://127.0.0.1:8000/detect/" \
  -F "image=@/path/to/your/image.jpg"
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
ruff check src/
```

## Architecture

1. **FastAPI Backend** (`src/main.py`):
   - Receives image uploads via HTTP
   - Calls processing logic
   - Returns JSON with detections and annotated image

2. **Processing Module** (`src/processing.py`):
   - Mock PPE detection function (simulates YOLO model)
   - Image annotation with bounding boxes and labels
   - Pure business logic, framework-agnostic

3. **Gradio Frontend** (`src/app_gradio.py`):
   - User-friendly web interface
   - Communicates with FastAPI backend via HTTP
   - Displays detection results and annotated images

## Mock Detection

This project uses a **mock detection function** that generates realistic but fake detection results. This allows you to:

- Test the entire pipeline without GPU requirements
- Develop UI/UX independently of model training
- Demonstrate the system architecture

To integrate a real YOLO model, simply replace the `detect_ppe` function in `src/processing.py` with actual model inference code.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
