from flask import Flask, request, jsonify
import os
from pathlib import Path
from typing import List
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import openai
import yaml
import uuid
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Configuration
UPLOAD_FOLDER = Path("data/raw")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
# Load OpenAI API key from config/secrets.yaml
secrets_path = Path(__file__).parent / "config" / "secrets.yaml"
if not secrets_path.exists():
    raise FileNotFoundError(f"Secrets file not found: {secrets_path}")
with open(secrets_path, "r") as f:
    secrets = yaml.safe_load(f)
api_key = secrets.get("OPENAI_API_KEY")
if not api_key:
    raise KeyError("OPENAI_API_KEY not found in secrets.yaml")
client = openai.OpenAI(api_key=api_key)

def extract_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        doc = fitz.open(file_path)
        all_text = ""
        for page in doc:
            text = page.get_text()
            if text.strip():
                all_text += text + "\n"
        return all_text.strip()
    else:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

def ask_chatgpt(prompt: str, model="gpt-4", temperature=0.7) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI: {e}"

@app.route("/invocations", methods=["POST"])
def generate_summary():
    print("⚡️ /invocations called")
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    summaries = []
    combined_summary = ""

    # Save and process each file
    for f in files:
        unique_name = f"{uuid.uuid4()}{Path(f.filename).suffix}"
        file_path = UPLOAD_FOLDER / unique_name
        f.save(file_path)

        text = extract_text_from_file(file_path)
        if not text.strip():
            summaries.append("Document unreadable or empty.")
            continue

        summary_prompt = (
            "You are a medical assistant. Below is a set of notes and prescriptions from a doctor.\n"
            "Please extract and rewrite the information in a clear, concise, and patient-friendly format, "
            "including any medication names, dosage instructions, diagnoses, and relevant notes.\n\n"
            "Generate content in points like summary."
            f"{text}"
        )
        summary = ask_chatgpt(summary_prompt)
        summaries.append(summary)
        combined_summary += summary + "\n"

        file_path.unlink(missing_ok=True)  # Clean up

    # Generate health suggestions
    health_prompt = (
        f"Based on the following patient medical summaries:\n\n{combined_summary}\n\n"
        "Generate 4 brief health suggestions in one line each."
    )
    health_suggestions = ask_chatgpt(health_prompt)

    # Generate food suggestions
    food_prompt = (
        f"Based on the patient summaries and health suggestions:\n\n{combined_summary}\n\n{health_suggestions}\n\n"
        "Generate 4 brief food/fruit/vegetable suggestions in one line each for the patient."
    )
    food_suggestions = ask_chatgpt(food_prompt)

    return jsonify({
        "summaries": summaries,
        "healthSuggestions": health_suggestions,
        "foodSuggestions": food_suggestions
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
