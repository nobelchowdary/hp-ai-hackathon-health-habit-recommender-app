from flask import Flask, request, jsonify
import os
from pathlib import Path
from typing import List
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import openai
import requests
import yaml
import uuid
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


# Configuration
UPLOAD_FOLDER = Path("data/raw")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
pytesseract.pytesseract.tesseract_cmd = os.environ.get("TESSERACT_CMD", "tesseract")
# Load secrets and configure model provider
secrets_path = Path(__file__).parent / "config" / "secrets.yaml"
if not secrets_path.exists():
    raise FileNotFoundError(f"Secrets file not found: {secrets_path}")
with open(secrets_path, "r") as f:
    secrets = yaml.safe_load(f)
# Select model provider: 'openai' (default) or 'local' for LLaMA2-7B via llama.cpp
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "openai").lower()
if MODEL_PROVIDER == "openai":
    api_key = secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise KeyError("OPENAI_API_KEY not found in secrets.yaml")
    client = openai.OpenAI(api_key=api_key)
elif MODEL_PROVIDER == "local":
    from langchain_community.llms import LlamaCpp
    LOCAL_MODEL_PATH = os.environ.get(
        "LOCAL_MODEL_PATH",
        "/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf"
    )
    llm = LlamaCpp(
        model_path=LOCAL_MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=512,
        n_ctx=4096,
        max_tokens=1024,
        f16_kv=True,
        verbose=False,
        stop=[],
        temperature=0.7,
    )
elif MODEL_PROVIDER == "hp_studio":
    # HP AI Studio MLflow model serving endpoint
    MLFLOW_SERVER_URL = os.environ.get("MLFLOW_SERVER_URL", "http://localhost:1234")
    STUDIO_MODEL_NAME = os.environ.get("STUDIO_MODEL_NAME")
    if not STUDIO_MODEL_NAME:
        raise KeyError("STUDIO_MODEL_NAME environment variable is required for hp_studio provider")
else:
    raise ValueError(f"Unsupported MODEL_PROVIDER: {MODEL_PROVIDER}")

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

def ask_llm(prompt: str, model="gpt-4", temperature=0.7) -> str:
    if MODEL_PROVIDER == "local":
        try:
            result = llm(prompt)
            if isinstance(result, dict):
                return result.get("text", str(result))
            return str(result)
        except Exception as e:
            return f"Error calling local LLaMA2 model: {e}"
    elif MODEL_PROVIDER == "hp_studio":
        # Invoke the MLflow model serving API
        url = f"{MLFLOW_SERVER_URL}/invocations"
        payload = {
            "columns": ["prompt"],
            "data": [[prompt]]
        }
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            # Expecting 'predictions' key or raw result
            preds = result.get("predictions") if isinstance(result, dict) else None
            if preds:
                # Return first prediction
                return preds[0]
            # Fallback to returning JSON as string
            return str(result)
        except Exception as e:
            return f"Error calling HP AI Studio model: {e}"
    else:
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
        summary = ask_llm(summary_prompt)
        summaries.append(summary)
        combined_summary += summary + "\n"

        file_path.unlink(missing_ok=True)  # Clean up

    # Generate health suggestions
    health_prompt = (
        f"Based on the following patient medical summaries:\n\n{combined_summary}\n\n"
        "Generate 4 brief health suggestions in one line each."
    )
    health_suggestions = ask_llm(health_prompt)

    # Generate food suggestions
    food_prompt = (
        f"Based on the patient summaries and health suggestions:\n\n{combined_summary}\n\n{health_suggestions}\n\n"
        "Generate 4 brief food/fruit/vegetable suggestions in one line each for the patient."
    )
    food_suggestions = ask_llm(food_prompt)

    return jsonify({
        "summaries": summaries,
        "healthSuggestions": health_suggestions,
        "foodSuggestions": food_suggestions
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
