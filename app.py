import torch

# Force torch to allow legacy full unpickling (override weights_only defaults)
# These two assignments change the internal flags so torch.load will allow full load:
torch.serialization._open_zipfile_reader_weights_only = False
torch.serialization._open_file_reader_weights_only = False

# Now whitelist the Ultralytics DetectionModel so unpickling of that class is allowed
from torch.serialization import add_safe_globals

# import the DetectionModel class from ultralytics *after* toggling torch flags? 
# Note: we still must import the class but do it AFTER setting the flags
# (importing ultralytics may import torch internally; order matters)
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])
# --- END: PyTorch 2.6+ compatibility block ---

# Now safe to import and load the model


from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import os
# Initialize Flask app
app = Flask(__name__)
# Load YOLO model (replace with your model path)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)
print(f"✅ Ultralytics model loaded from {MODEL_PATH}")
# Endpoint to test API status
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Tomato Ripeness Detection API is running 🚀",
        "usage": "POST an image to /predict"
    })
# Endpoint for image prediction
@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was sent
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    # Run inference with YOLO
    results = model.predict(image, imgsz=640, conf=0.25)
    # Parse results
    predictions = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        label = results[0].names[cls_id]
        confidence = float(box.conf)
        predictions.append({
            "label": label,
            "confidence": round(confidence, 3)
        })
    # Return response
    return jsonify({
        "detections": predictions,
        "num_detections": len(predictions)
    })
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Allow cloud platforms to assign a port
    app.run(host="0.0.0.0", port=port)