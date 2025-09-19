from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from ai_core import process_dataset

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".csv", ".json"]:
        return jsonify({"error": "Unsupported file type"}), 400
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_FOLDER, f"{file_id}{ext}")
    output_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_output.json")
    file.save(input_path)
    try:
        process_dataset(input_path, output_path, glossary_path="CUAD.json")
        with open(output_path, "r", encoding="utf-8") as f:
            data = f.read()
        return app.response_class(data, mimetype="application/json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/download", methods=["POST"])
def download():
    data = request.get_json()
    if not data or "output" not in data:
        return jsonify({"error": "No output data provided"}), 400
    file_id = str(uuid.uuid4())
    output_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(data["output"])
    return send_file(output_path, as_attachment=True, download_name="output.json")

if __name__ == "__main__":
    app.run(port=5000, debug=True)