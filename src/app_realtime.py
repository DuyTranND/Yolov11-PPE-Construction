"""
Enhanced Gradio frontend with real-time video detection support.
Supports webcam streams, video file uploads, and image detection.
"""

import io
import base64
import requests
import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from PIL import Image
import gradio as gr
import tempfile
import os

from src.video_processing import VideoProcessor, process_video_stream_yolo


# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/detect/"


def call_detection_api(image: Image.Image) -> Tuple[Optional[Image.Image], str]:
    """
    Call the FastAPI detection endpoint with an image.
    
    Args:
        image: PIL Image from Gradio interface
    
    Returns:
        Tuple of (processed_image, detection_logs_text)
    """
    if image is None:
        return None, "❌ No image provided. Please upload an image."
    
    try:
        # Convert PIL Image to bytes
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        
        # Prepare the file for multipart upload
        files = {"image": ("image.png", image_bytes, "image/png")}
        
        # Make POST request to FastAPI backend
        response = requests.post(API_URL, files=files, timeout=30)
        
        # Check if request was successful
        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error")
            return None, f"❌ API Error ({response.status_code}): {error_detail}"
        
        # Parse JSON response
        result = response.json()
        detections = result.get("detections", [])
        processed_image_b64 = result.get("processed_image", "")
        
        # Decode Base64 image
        try:
            image_data = base64.b64decode(processed_image_b64)
            processed_image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            return None, f"❌ Error decoding processed image: {str(e)}"
        
        # Format detection logs
        detection_text = format_detection_logs(detections)
        
        return processed_image, detection_text
    
    except requests.exceptions.ConnectionError:
        return None, (
            "❌ **Connection Error**: Cannot connect to the API server.\n\n"
            "Please ensure the FastAPI backend is running:\n"
            "```\nuvicorn src.main:app --reload --port 8000\n```"
        )
    except requests.exceptions.Timeout:
        return None, "❌ **Timeout Error**: The API request took too long. Please try again."
    except Exception as e:
        return None, f"❌ **Unexpected Error**: {str(e)}"


def format_detection_logs(detections: list) -> str:
    """Format detection results into readable text."""
    if not detections:
        return "✅ No PPE violations detected."
    
    detection_text = f"🎯 **Detected {len(detections)} object(s):**\n\n"
    
    # Group detections by label
    label_counts = {}
    for det in detections:
        label = det["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Summary
    detection_text += "**Summary:**\n"
    for label, count in sorted(label_counts.items()):
        emoji = get_emoji_for_label(label)
        detection_text += f"- {emoji} {label}: {count}\n"
    
    detection_text += "\n**Detailed Detections:**\n"
    
    # Sort by confidence (highest first)
    sorted_detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
    
    for i, det in enumerate(sorted_detections, 1):
        label = det["label"]
        confidence = det["confidence"]
        bbox = det["bbox"]
        emoji = get_emoji_for_label(label)
        
        detection_text += (
            f"{i}. {emoji} **{label}** "
            f"(confidence: {confidence:.2f})\n"
            f"   📍 BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n\n"
        )
    
    return detection_text


def process_webcam_frame(frame, conf_threshold: float, iou_threshold: float):
    """
    Process a single webcam frame (Gradio streaming style).
    Similar to traffic sign detection code.
    
    Args:
        frame: Frame from webcam (numpy array)
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
    
    Returns:
        Annotated frame
    """
    if frame is None:
        return None
    
    # Make a writable copy
    frame = frame.copy()
    
    try:
        # Convert RGB (from Gradio) to BGR (for OpenCV/YOLO)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Run detection using the optimized function
        annotated_bgr, detections = process_video_stream_yolo(
            frame_bgr,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Convert back to RGB for Gradio display
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        
        # Don't add any overlay text - keep it clean
        return annotated_rgb
        
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        # Return original frame with error message
        cv2.putText(frame, f"Error: {str(e)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame


def process_video_file(video_file, conf_threshold: float, iou_threshold: float, progress=gr.Progress()):
    """
    Process uploaded video file with smooth bounding boxes.
    
    Args:
        video_file: Uploaded video file path
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold
        progress: Gradio progress tracker
    
    Returns:
        Path to processed video file and statistics
    """
    if video_file is None:
        return None, "❌ No video file provided."
    
    processor = VideoProcessor(conf_threshold, iou_threshold)
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Open input video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return None, f"❌ Could not open video file: {video_file}\nPlease ensure the file is a valid video format."
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0 or width == 0 or height == 0:
        cap.release()
        return None, "❌ Invalid video properties. The video file may be corrupted."
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        return None, "❌ Could not create output video file."
    
    frame_count = 0
    processed_count = 0
    last_detections = []  # Cache last detection results
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection every frame for smooth tracking
            # (You can change to % 2 == 0 if you want faster processing)
            try:
                annotated_frame, detections, stats = processor.process_frame(frame)
                last_detections = detections  # Cache detections
                processed_count += 1
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                # Use last known detections if processing fails
                if last_detections:
                    annotated_frame = draw_boxes_on_frame(frame, last_detections)
                else:
                    annotated_frame = frame
            
            out.write(annotated_frame)
            
            # Update progress - fixed format
            if total_frames > 0 and frame_count % 10 == 0:  # Update every 10 frames
                progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")
    
    except Exception as e:
        cap.release()
        out.release()
        return None, f"❌ Error during video processing: {str(e)}"
    
    finally:
        cap.release()
        out.release()
    
    # Generate statistics
    stats = processor.detection_stats
    stats_text = f"""
### 📊 Video Processing Complete

- **Total Frames Processed:** {stats['total_frames']}
- **Total Detections:** {stats['total_detections']}
- **Total Violations:** {stats['violations']}
- **Average Detections per Frame:** {stats['total_detections'] / max(stats['total_frames'], 1):.2f}
- **Processing Rate:** {processed_count}/{frame_count} frames analyzed
"""
    
    return output_path, stats_text


def get_emoji_for_label(label: str) -> str:
    """Get appropriate emoji for each PPE label."""
    emoji_map = {
        "person": "👤",
        "helmet": "⛑️",
        "no-helmet": "🚫⛑️",
        "vest": "🦺",
        "no-vest": "🚫🦺",
        "gloves": "🧤",
        "boots": "🥾",
        "Person": "👤",
        "no_helmet": "🚫⛑️",
        "no_goggle": "🚫🥽",
        "no_gloves": "🚫🧤",
        "no_boots": "🚫🥾",
        "goggles": "🥽",
        "none": "❌",
    }
    return emoji_map.get(label, "🔍")


def draw_boxes_on_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes on a frame using OpenCV (for video processing).
    
    Args:
        frame: OpenCV frame (BGR format)
        detections: List of detection dictionaries with 'label', 'confidence', 'bbox'
    
    Returns:
        Annotated frame
    """
    # Import color map from processing module
    from src.processing import COLOR_MAP
    
    annotated_frame = frame.copy()
    
    for det in detections:
        label = det["label"]
        confidence = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        
        # Get color (RGB format from COLOR_MAP)
        color_rgb = COLOR_MAP.get(label, (0, 255, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert to BGR
        
        # Draw bounding box
        thickness = max(2, int(min(frame.shape[:2]) / 400))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, thickness)
        
        # Prepare label text
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
            tuple(int(c * 0.7) for c in color_rgb),
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
    
    return annotated_frame


def check_api_status() -> str:
    """Check if the FastAPI backend is reachable."""
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if response.status_code == 200:
            return "✅ API Status: **Connected**"
        else:
            return f"⚠️ API Status: **Unhealthy** (Status code: {response.status_code})"
    except requests.exceptions.ConnectionError:
        return "❌ API Status: **Disconnected** (Make sure to run: `uvicorn src.main:app --reload --port 8000`)"
    except Exception as e:
        return f"❌ API Status: **Error** - {str(e)}"


# Create Gradio interface with tabs
with gr.Blocks(
    title="🦺 PPE Detection System - Real-time",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1400px !important;
            margin: auto !important;
        }
        #api-status {
            text-align: center;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    """
) as demo:
    
    gr.Markdown(
        """
        # 🦺 Construction Site PPE Detection System
        
        **Real-time Detection** | Upload images, process videos, or use live webcam feed
        
        **Detectable Items:** Helmets, Vests, Gloves, Boots, Persons, and violations (no-helmet, no-vest)
        """
    )
    
    # API Status indicator
    api_status = gr.Markdown(check_api_status(), elem_id="api-status")
    refresh_status_btn = gr.Button("🔄 Refresh API Status", size="sm")
    
    # Create tabs for different input modes
    with gr.Tabs():
        
        # Tab 1: Image Detection
        with gr.Tab("📸 Image Detection"):
            gr.Markdown("### Upload an image for PPE detection")
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="pil",
                        label="Input Image",
                        height=400
                    )
                    image_detect_btn = gr.Button("🔍 Detect PPE", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    image_output = gr.Image(
                        type="pil",
                        label="Annotated Image",
                        height=400
                    )
            
            image_logs = gr.Markdown(
                label="Detection Results",
                value="Upload an image and click 'Detect PPE' to see results."
            )
        
        # Tab 2: Webcam Real-time
        with gr.Tab("🎥 Live Webcam"):
            gr.Markdown(
                """
                ### Real-time webcam detection
                **Note:** Webcam stream starts automatically. Allow camera access in your browser.
                """
            )
            
            with gr.Row():
                conf_slider_webcam = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Minimum confidence for detections"
                )
                iou_slider_webcam = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="IoU Threshold",
                    info="Overlap threshold for NMS"
                )
            
            # Simple streaming interface like traffic sign detection
            webcam_interface = gr.Interface(
                fn=process_webcam_frame,
                inputs=[
                    gr.Image(sources=["webcam"], streaming=True),
                    conf_slider_webcam,
                    iou_slider_webcam
                ],
                outputs=gr.Image(streaming=True),
                live=True,
                flagging_mode="never"
            )
            
            gr.Markdown(
                """
                **Tips:**
                - Allow camera access when prompted by your browser
                - The detection runs automatically on the live feed
                - Adjust thresholds above to tune sensitivity
                - **WSL2 Users**: This may not work - run on Windows or use Video File tab
                """
            )
        
        # Tab 3: Video File Processing
        with gr.Tab("🎬 Video File"):
            gr.Markdown("### Upload a video file for batch processing")
            
            with gr.Row():
                conf_slider_video = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.3,
                    step=0.05,
                    label="Confidence Threshold"
                )
                iou_slider_video = gr.Slider(
                    minimum=0.1,
                    maximum=0.9,
                    value=0.5,
                    step=0.05,
                    label="IoU Threshold"
                )
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="Input Video",
                        height=400
                    )
                    video_process_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    video_output = gr.Video(
                        label="Processed Video",
                        height=400
                    )
            
            video_stats = gr.Markdown(
                label="Processing Statistics",
                value="Upload a video and click 'Process Video' to see results."
            )
    
    
    # Event handlers
    image_detect_btn.click(
        fn=call_detection_api,
        inputs=[image_input],
        outputs=[image_output, image_logs]
    )
    
    # Webcam uses live=True interface, no button handler needed
    
    video_process_btn.click(
        fn=process_video_file,
        inputs=[video_input, conf_slider_video, iou_slider_video],
        outputs=[video_output, video_stats]
    )
    
    refresh_status_btn.click(
        fn=check_api_status,
        outputs=api_status
    )


# Launch the app
if __name__ == "__main__":
    print("🚀 Launching Enhanced PPE Detection System...")
    print("📡 Features: Image Detection | Live Webcam | Video Processing")
    print("=" * 60)
    
    # Check API status at startup
    status = check_api_status()
    print(status)
    print("=" * 60)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )