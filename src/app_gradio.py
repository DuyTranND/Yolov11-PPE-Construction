"""
Gradio frontend for PPE Detection System.
This app provides a user-friendly interface and communicates with the FastAPI backend.
"""

import io
import base64
import requests
from typing import Tuple, Optional
from PIL import Image
import gradio as gr


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
        if not detections:
            detection_text = "✅ No PPE violations detected."
        else:
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
    }
    return emoji_map.get(label, "🔍")


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


# Create Gradio interface
with gr.Blocks(
    title="🦺 PPE Detection System",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 1200px !important;
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
        
        Upload an image to detect Personal Protective Equipment (PPE) and safety violations.
        
        **Detectable Items:** Helmets, Vests, Gloves, Boots, Persons, and violations (no-helmet, no-vest)
        """
    )
    
    # API Status indicator
    api_status = gr.Markdown(check_api_status(), elem_id="api-status")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                height=400
            )
            
            with gr.Row():
                detect_btn = gr.Button("🔍 Detect PPE", variant="primary", size="lg")
                refresh_status_btn = gr.Button("🔄 Refresh Status", size="sm")
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Detection Results")
            output_image = gr.Image(
                type="pil",
                label="Annotated Image",
                height=400
            )
    
    with gr.Row():
        detection_logs = gr.Markdown(
            label="Detection Logs",
            value="Upload an image and click 'Detect PPE' to see results."
        )
    
    # Example images section
    gr.Markdown("### 📸 Try Example Images")
    gr.Examples(
        examples=[
            # Note: These paths would need actual example images in production
            # For now, they serve as placeholders
        ],
        inputs=input_image,
        label="Example Construction Site Images"
    )
    
    # Event handlers
    detect_btn.click(
        fn=call_detection_api,
        inputs=[input_image],
        outputs=[output_image, detection_logs]
    )
    
    refresh_status_btn.click(
        fn=check_api_status,
        outputs=api_status
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        
        ### 📝 How to Use:
        1. **Start the API**: `uvicorn src.main:app --reload --port 8000`
        2. **Upload an image** of a construction site or worker
        3. **Click 'Detect PPE'** to analyze the image
        4. **Review results** in the annotated image and detection logs
        
        ### 🔧 Technical Details:
        - Backend: FastAPI (http://127.0.0.1:8000)
        - Frontend: Gradio
        - Detection: Mock YOLO-style model (replace with real model in production)
        
        ### 📚 API Documentation:
        Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.
        """
    )


# Launch the app
if __name__ == "__main__":
    print("🚀 Launching PPE Detection Gradio Interface...")
    print("📡 Connecting to FastAPI backend at:", API_URL)
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
