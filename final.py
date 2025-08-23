import os
import cv2
import json
import concurrent.futures
import PIL.Image
from ultralytics import YOLO
# Make sure these local modules are in your project directory
from pothole_agent import get_confidence_and_severity as pothole_infer
from trash_agent1 import analyze_waste

import google.generativeai as genai
from dotenv import load_dotenv

# Load .env file (for GOOGLE_API_KEY)
load_dotenv()

def process_image(image_path, pothole_weights="models/pothole.pt", trash_weights="models/trash_new.pt"):
    results = {}

    # Check image exists
    if not os.path.exists(image_path):
        return {"error": f"Image not found: {image_path}"}

    # Run pothole & trash in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        pothole_future = executor.submit(
            lambda: pothole_infer(YOLO(pothole_weights)(image_path), image_path)
        )
        trash_future = executor.submit(
            lambda: analyze_waste(trash_weights, image_path)
        )

        pothole_conf_tensor, pothole_sev_tensor = pothole_future.result()
        trash_conf_tensor, trash_sev_tensor = trash_future.result()

    # --- THIS IS THE FIX ---
    # Convert potential Tensor objects to standard Python floats using .item()
    # This prevents the "not JSON serializable" error.
    pothole_conf = pothole_conf_tensor.item() if hasattr(pothole_conf_tensor, 'item') else pothole_conf_tensor
    pothole_sev = pothole_sev_tensor.item() if hasattr(pothole_sev_tensor, 'item') else pothole_sev_tensor
    trash_conf = trash_conf_tensor.item() if hasattr(trash_conf_tensor, 'item') else trash_conf_tensor
    trash_sev = trash_sev_tensor.item() if hasattr(trash_sev_tensor, 'item') else trash_sev_tensor


    # --- Calculate overall metrics ---
    overall_severity = (0.6 * pothole_sev + 0.4 * trash_sev)
    overall_confidence = (pothole_conf + trash_conf) / 2

    # --- Gemini for description ---
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No GOOGLE_API_KEY found. Please add it in your .env file.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    prompt = f"""
    You are an urban street monitoring assistant.
    Based on these model results:

    - Pothole: Severity={pothole_sev:.2f}, Confidence={pothole_conf:.2f}
    - Trash: Severity={trash_sev:.2f}, Confidence={trash_conf:.2f}
    - Overall: Severity={overall_severity:.2f}, Confidence={overall_confidence:.2f}

    Return a string with the brief description
    """
    
    try:
        img = PIL.Image.open(image_path)
    except FileNotFoundError:
        return {"error": f"Image file not found at path: {image_path}"}


    response = model.generate_content([prompt, img])

    results["pothole_confidence"] = pothole_conf
    results["pothole_severity"] = pothole_sev
    results["trash_confidence"] = trash_conf
    results["trash_severity"] = trash_sev
    results["overall_severity"] = overall_severity
    results["overall_confidence"] = overall_confidence
    results["gemini_output"] = response.text.strip()
    
    return results

# --- Example Run ---
if __name__ == "__main__":
    img = "C:/Users/Lenovo/OneDrive/Documents/7th semester/smart city/t.jpeg"
    output = process_image(img)
    print(json.dumps(output, indent=2))