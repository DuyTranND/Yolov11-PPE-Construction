"""
FastAPI backend for PPE Detection System.
Provides REST API endpoint for image-based PPE detection.
"""

import io
import base64
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.processing import detect_ppe


# Pydantic models for request/response validation
class Detection(BaseModel):
    """Single detection result."""
    label: str
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    """Response model for detection endpoint."""
    detections: List[Detection]
    processed_image: str  # Base64 encoded image


# Initialize FastAPI app
app = FastAPI(
    title="PPE Detection API",
    description="Construction Site Personal Protective Equipment Detection System",
    version="1.0.0",
)

# Add CORS middleware to allow requests from Gradio frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PPE Detection API",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect/": "Upload image for PPE detection",
            "GET /docs": "Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ppe-detection-api"}


@app.post("/detect/", response_model=DetectionResponse)
async def detect_ppe_endpoint(image: UploadFile = File(...)):
    """
    Detect PPE in an uploaded image.
    
    Args:
        image: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        DetectionResponse containing:
        - detections: List of detected objects with labels, confidence, and bboxes
        - processed_image: Base64-encoded annotated image
    
    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read image bytes ONCE
        image_bytes = await image.read()
        
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image file. Please upload a valid image."
            )
        
        # Validate that it's actually an image
        try:
            test_image = Image.open(io.BytesIO(image_bytes))
            test_image.load()  # ← Dùng load() thay vì verify()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}. Please upload a valid image (JPEG, PNG, etc.)"
            )
        
        # Perform PPE detection (image_bytes vẫn còn nguyên)
        detections, annotated_image = detect_ppe(image_bytes)
        
        # Convert annotated image to Base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Format detections for response
        detection_objects = [
            Detection(
                label=det["label"],
                confidence=det["confidence"],
                bbox=det["bbox"]
            )
            for det in detections
        ]
        
        return DetectionResponse(
            detections=detection_objects,
            processed_image=img_base64
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.get("/classes")
async def get_classes():
    """
    Get list of detectable PPE classes.
    
    Returns:
        Dictionary containing all supported PPE classes
    """
    return {
        "classes": [
            "person",
            "helmet",
            "no-helmet",
            "vest",
            "no-vest",
            "gloves",
            "boots",
        ],
        "description": "PPE classes that can be detected by the system"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
