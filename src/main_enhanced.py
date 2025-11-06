"""
Enhanced FastAPI backend with video streaming endpoints.
Extends the original API with real-time video processing capabilities.
"""

import io
import base64
import cv2
import numpy as np
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

from src.processing import detect_ppe
from src.video_processing import VideoProcessor, process_video_stream_yolo


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


class VideoFrameRequest(BaseModel):
    """Request model for video frame detection."""
    frame: str  # Base64 encoded frame
    conf_threshold: float = 0.3
    iou_threshold: float = 0.5


class VideoStatsResponse(BaseModel):
    """Response model for video processing statistics."""
    total_frames: int
    total_detections: int
    violations: int
    avg_detections_per_frame: float


# Initialize FastAPI app
app = FastAPI(
    title="PPE Detection API - Enhanced",
    description="Construction Site PPE Detection with Real-time Video Support",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global video processor
video_processor = VideoProcessor()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PPE Detection API - Enhanced",
        "version": "2.0.0",
        "endpoints": {
            "POST /detect/": "Upload image for PPE detection",
            "POST /detect-frame/": "Detect PPE in a single video frame",
            "WS /ws/video-stream": "WebSocket endpoint for real-time video",
            "GET /video-stats": "Get video processing statistics",
            "GET /docs": "Interactive API documentation",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ppe-detection-api-enhanced",
        "features": ["image", "video", "websocket"]
    }


@app.post("/detect/", response_model=DetectionResponse)
async def detect_ppe_endpoint(image: UploadFile = File(...)):
    """
    Detect PPE in an uploaded image.
    
    Args:
        image: Uploaded image file (JPEG, PNG, etc.)
    
    Returns:
        DetectionResponse containing detections and annotated image
    """
    try:
        # Read image bytes
        image_bytes = await image.read()
        
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image file. Please upload a valid image."
            )
        
        # Validate image
        try:
            test_image = Image.open(io.BytesIO(image_bytes))
            test_image.load()
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )
        
        # Perform PPE detection
        detections, annotated_image = detect_ppe(image_bytes)
        
        # Convert annotated image to Base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Format detections
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
        raise
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/detect-frame/", response_model=DetectionResponse)
async def detect_frame(request: VideoFrameRequest):
    """
    Detect PPE in a single video frame (Base64 encoded).
    Optimized for video stream processing.
    
    Args:
        request: VideoFrameRequest with Base64 encoded frame
    
    Returns:
        DetectionResponse with detections and annotated frame
    """
    try:
        # Decode Base64 frame
        frame_bytes = base64.b64decode(request.frame)
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode frame"
            )
        
        # Process frame with YOLO
        annotated_frame, detections = process_video_stream_yolo(
            frame,
            conf_threshold=request.conf_threshold,
            iou_threshold=request.iou_threshold
        )
        
        # Encode annotated frame to Base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Format detections
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
            processed_image=frame_base64
        )
    
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing frame: {str(e)}"
        )


@app.websocket("/ws/video-stream")
async def websocket_video_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    
    Protocol:
    - Client sends: Base64 encoded frames
    - Server responds: JSON with detections and annotated frame
    """
    await websocket.accept()
    print("WebSocket connection established")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_json()
            
            frame_base64 = data.get("frame", "")
            conf_threshold = data.get("conf_threshold", 0.3)
            iou_threshold = data.get("iou_threshold", 0.5)
            
            # Decode frame
            frame_bytes = base64.b64decode(frame_base64)
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Process frame
                annotated_frame, detections = process_video_stream_yolo(
                    frame,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                
                # Encode result
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                result_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Send response
                await websocket.send_json({
                    "detections": detections,
                    "frame": result_base64
                })
    
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await websocket.close()


@app.get("/video-stats", response_model=VideoStatsResponse)
async def get_video_stats():
    """
    Get current video processing statistics.
    
    Returns:
        Statistics including frame count, detections, violations
    """
    stats = video_processor.detection_stats
    
    avg_detections = (
        stats["total_detections"] / stats["total_frames"]
        if stats["total_frames"] > 0
        else 0.0
    )
    
    return VideoStatsResponse(
        total_frames=stats["total_frames"],
        total_detections=stats["total_detections"],
        violations=stats["violations"],
        avg_detections_per_frame=round(avg_detections, 2)
    )


@app.post("/reset-stats")
async def reset_video_stats():
    """Reset video processing statistics."""
    video_processor.reset_stats()
    return {"message": "Statistics reset successfully"}


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