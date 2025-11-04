"""
Core PPE detection and image processing logic.
This module contains the YOLO-based detection function and image annotation utilities.
"""

import io
import numpy as np
import torch
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import cv2

# Patch torch.load để tránh lỗi weights_only
_original_load = torch.load 
def patched_load(*args, **kwargs): 
    kwargs.setdefault('weights_only', False) 
    return _original_load(*args, **kwargs) 
torch.load = patched_load 

# Load model một lần khi import module
print("🔄 Loading YOLOv11 PPE model...")
MODEL_PATH = 'models/yolo11m_ppe_best.pt'
model = YOLO(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"📋 Classes: {list(model.names.values())}")

# Bảng màu RGB cho PIL (đậm, tương phản cao)
COLOR_MAP = {
    "person": (190, 190, 190),  # xám sáng
    "helmet": (255, 120, 30),   # cam đậm
    "vest": (60, 220, 50),      # xanh lá đậm
    "gloves": (255, 200, 60),   # vàng tươi
    "boots": (230, 80, 255),    # tím tươi
    "no-helmet": (255, 50, 50), # đỏ
    "no-vest": (220, 60, 60),   # đỏ đậm
}


def draw_boxes(image: Image.Image, detections: List[Dict]) -> Image.Image:
    """
    Draw bounding boxes and labels on the image using PIL.
    
    Args:
        image: PIL Image object
        detections: List of detection dictionaries
    
    Returns:
        Annotated PIL Image object
    """
    # Create a copy to avoid modifying the original
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Calculate line thickness based on image size
    img_width, img_height = image.size
    line_thickness = max(2, int(min(img_width, img_height) / 400))
    
    # Try to load a font, fall back to default if not available
    try:
        font_size = max(12, int(min(img_width, img_height) / 50))
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                   font_size)
    except Exception:
        font = ImageFont.load_default()
    
    for detection in detections:
        label = detection["label"]
        confidence = detection["confidence"]
        x1, y1, x2, y2 = detection["bbox"]
        
        # Get color for this class
        color = COLOR_MAP.get(label, (0, 255, 255))
        
        # Draw bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=line_thickness
        )
        
        # Draw semi-transparent filled rectangle for label background
        label_text = f"{label} {confidence:.2f}"
        
        # Get text bounding box
        try:
            bbox = draw.textbbox((x1, y1), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(label_text, font=font)
        
        # Position label above the box if possible, otherwise inside
        label_y = max(0, y1 - text_height - 4)
        
        # Draw label background with some padding
        padding = 4
        draw.rectangle(
            [
                (x1, label_y),
                (x1 + text_width + padding * 2, label_y + text_height + padding)
            ],
            fill=tuple(int(c * 0.7) for c in color)  # Darker version of box color
        )
        
        # Draw label text
        draw.text(
            (x1 + padding, label_y + padding // 2),
            label_text,
            fill=(255, 255, 255),
            font=font
        )
    
    return annotated_image


def detect_ppe(image_bytes: bytes, conf_threshold: float = 0.30, 
               iou_threshold: float = 0.50) -> Tuple[List[Dict], Image.Image]:
    """
    Main PPE detection function using YOLOv11 model.
    
    This function:
    1. Loads the image from bytes
    2. Runs YOLO inference
    3. Parses detection results
    4. Draws bounding boxes on the image
    
    Args:
        image_bytes: Raw image bytes
        conf_threshold: Confidence threshold for detections (0.0-1.0)
        iou_threshold: IoU threshold for NMS (0.0-1.0)
    
    Returns:
        Tuple of (detections_list, annotated_image)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary (handle RGBA, grayscale, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Set model parameters
    model.overrides['conf'] = float(conf_threshold)
    model.overrides['iou'] = float(iou_threshold)
    model.overrides['agnostic_nms'] = False
    model.overrides['max_det'] = 300
    
    # Perform inference
    results = model.predict(image, verbose=False)
    result = results[0]
    
    # Parse detections
    detections = []
    print(f"\n🎯 Detected {len(result.boxes)} objects:")
    
    for box in result.boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Get class name
        try:
            label = model.names[cls_id]
        except Exception:
            label = str(cls_id)
        
        # Get bounding box coordinates
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = map(int, xyxy)
        
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "bbox": [x1, y1, x2, y2]
        })
        
        print(f"  • {label}: {confidence:.2f} at [{x1}, {y1}, {x2}, {y2}]")
    
    # Draw bounding boxes on the image
    annotated_image = draw_boxes(image, detections)
    
    return detections, annotated_image