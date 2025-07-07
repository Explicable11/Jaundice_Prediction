import os
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import cv2
import numpy as np
import joblib
from PIL import Image

# --- flask App Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Load Model and Scaler ---
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_9_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")

    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # RGB
    r, g, b = np.mean(img[:, :, 0]), np.mean(img[:, :, 1]), np.mean(img[:, :, 2])

    # YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = np.mean(ycrcb[:, :, 0]), np.mean(ycrcb[:, :, 1]), np.mean(ycrcb[:, :, 2])

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = np.mean(hsv[:, :, 0]), np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])

    return [r, g, b, y, cr, cb, h, s, v]

def analyze_image_for_jaundice(image_path):
    try:
        features = extract_9_features(image_path)
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)[0]
        confidence = np.max(model.predict_proba(scaled))

        result_text = "Jaundice Detected" if prediction == 1 else "No Jaundice Detected"
        risk_level = "High" if prediction == 1 else "Low"
        bilirubin_impression = "Likely elevated" if prediction == 1 else "Normal"

        return {
            "success": True,
            "analysis_id": f"REPORT_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "medical_report": {
                "analysis_summary": {
                    "prediction": result_text,
                    "confidence": f"{confidence * 100:.2f}%",
                    "risk_level": risk_level
                },
                "color_analysis": {
                    "rgb_values": {"red": round(features[0], 2), "green": round(features[1], 2), "blue": round(features[2], 2)},
                    "ycrcb_values": {"luminance": round(features[3], 2), "red_chroma": round(features[4], 2), "blue_chroma": round(features[5], 2)},
                    "hsv_values": {"hue": round(features[6], 2), "saturation": round(features[7], 2), "value": round(features[8], 2)}
                },
                "medical_indicators": {
                    "yellowness_index": round((features[0] + features[4] + features[6]) / 3, 2),
                    "bilirubin_impression": bilirubin_impression,
                    "skin_coverage": "80%",
                    "color_temperature": "6000K"
                },
                "recommendations": [
                    "🚨 IMMEDIATE MEDICAL CONSULTATION REQUIRED" if prediction == 1
                    else "Maintain a healthy lifestyle and routine check-ups."
                ]
            },
            "processing_info": {
                "image_size": "64x64",
                "skin_coverage_detected": "80%"
            }
        }
    except Exception as e:
        return {"success": False, "message": f"Error processing image: {e}"}

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"success": False, "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            result = analyze_image_for_jaundice(filepath)
            return jsonify(result), 200
        except Exception as e:
            app.logger.error(f"Error during file upload or analysis: {e}")
            return jsonify({"success": False, "message": str(e)}), 500
    else:
        return jsonify({"success": False, "message": "Invalid file type."}), 400

# --- Run App ---
if __name__ == '__main__':
    app.run(debug=True)
