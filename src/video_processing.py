"""
Real-time video processing for PPE Detection System.
Handles webcam streams and video file processing with frame-by-frame detection.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Dict, List
from PIL import Image
import io
from src.processing import detect_ppe, draw_boxes, model, COLOR_MAP


class VideoProcessor:
    """Process video streams for real-time PPE detection."""
    
    def __init__(self, conf_threshold: float = 0.30, iou_threshold: float = 0.50):
        """
        Initialize video processor.
        
        Args:
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        self.detection_stats = {
            "total_frames": 0,
            "total_detections": 0,
            "violations": 0
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Process a single video frame using direct YOLO prediction.
        Optimized: frame → YOLO → annotated frame (no PIL conversion)
        
        Args:
            frame: OpenCV frame (BGR format)
        
        Returns:
            Tuple of (annotated_frame, detections, stats)
        """
        # Use the optimized direct YOLO processing
        annotated_frame, detections = process_video_stream_yolo(
            frame,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold
        )
        
        # Update statistics
        self.frame_count += 1
        self.detection_stats["total_frames"] = self.frame_count
        self.detection_stats["total_detections"] += len(detections)
        
        # Count violations (check for 'no' or 'no_' in label)
        violations = sum(1 for d in detections 
                        if 'no' in d['label'].lower() or 'no_' in d['label'].lower())
        self.detection_stats["violations"] += violations
        
        return annotated_frame, detections, self.detection_stats.copy()
    
    def _add_overlay(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Add minimal information overlay to the frame.
        Keeping this method for compatibility but not adding overlays.
        """
        # Return frame without overlay - keep it clean
        return frame
    
    def process_webcam(self, camera_id: int = 0, 
                       skip_frames: int = 1) -> Generator[Tuple[np.ndarray, List[Dict]], None, None]:
        """
        Process webcam stream in real-time.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            skip_frames: Process every Nth frame to improve performance
        
        Yields:
            Tuple of (annotated_frame, detections)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_counter += 1
                
                # Skip frames for performance
                if frame_counter % skip_frames != 0:
                    continue
                
                # Process frame
                annotated_frame, detections, _ = self.process_frame(frame)
                
                yield annotated_frame, detections
        
        finally:
            cap.release()
    
    def process_video_file(self, video_path: str, 
                          skip_frames: int = 1) -> Generator[Tuple[np.ndarray, List[Dict], float], None, None]:
        """
        Process video file frame by frame.
        
        Args:
            video_path: Path to video file
            skip_frames: Process every Nth frame
        
        Yields:
            Tuple of (annotated_frame, detections, progress)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counter = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_counter += 1
                
                # Skip frames for performance
                if frame_counter % skip_frames != 0:
                    continue
                
                # Process frame
                annotated_frame, detections, _ = self.process_frame(frame)
                
                # Calculate progress
                progress = frame_counter / total_frames if total_frames > 0 else 0
                
                yield annotated_frame, detections, progress
        
        finally:
            cap.release()
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.frame_count = 0
        self.detection_stats = {
            "total_frames": 0,
            "total_detections": 0,
            "violations": 0
        }


def process_video_stream_yolo(frame: np.ndarray, 
                               conf_threshold: float = 0.30,
                               iou_threshold: float = 0.50) -> Tuple[np.ndarray, List[Dict]]:
    """
    Optimized YOLO processing for video streams.
    Directly processes numpy arrays without PIL conversion overhead.
    
    Args:
        frame: OpenCV frame (BGR format)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Tuple of (annotated_frame, detections)
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set model parameters
    model.overrides['conf'] = float(conf_threshold)
    model.overrides['iou'] = float(iou_threshold)
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 300
    
    # Perform inference directly on numpy array
    results = model.predict(frame_rgb, verbose=False)
    result = results[0]
    
    # Parse detections
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        try:
            label = model.names[cls_id]
        except Exception:
            label = str(cls_id)
        
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = map(int, xyxy)
        
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2]
        })
    
    # Draw annotations directly on OpenCV frame
    annotated_frame = frame_rgb.copy()
    
    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        
        # Get color
        color = COLOR_MAP.get(label, (0, 255, 255))
        color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR
        
        # Draw bounding box
        thickness = max(2, int(min(frame.shape[:2]) / 400))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, thickness)
        
        # Draw label
        label_text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Draw label background
        label_y = max(text_height + 10, y1)
        cv2.rectangle(
            annotated_frame,
            (x1, label_y - text_height - 10),
            (x1 + text_width + 10, label_y),
            tuple(int(c * 0.7) for c in color),
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated_frame,
            label_text,
            (x1 + 5, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness
        )
    
    # Convert back to BGR for OpenCV
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    
    return annotated_frame, detections