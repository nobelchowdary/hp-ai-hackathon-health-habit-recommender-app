# 🏥 Medical Report Analyzer & Healthy Habits Recommender

## Content
- Project Description
- Overview
- Models and Methods
- Project Structure
- Setup
- MLflow & Deployment
- Usage
- Judging and Testing
- Open-source Repository
- Contact and Support
 
## Project Description
**Technical Workflow:** This solution uses HP AI Studio to orchestrate OCR-based text extraction and large language model calls to generate medical summaries, health suggestions, and food recommendations.

**Challenges & Solutions:** Handling multi-format medical documents (PDFs, scans), ensuring accurate text extraction and concise AI-generated outputs, and packaging the end-to-end pipeline for reproducible deployment in AI Studio.

**HP AI Studio Features Leveraged:** Custom Web App deployments, built-in MLflow model registry and versioning, automatic Swagger UI generation, GPU/CPU compute profiles, and environment management.

**Lessons Learned & Best Practices:** Use environment variables for sensitive keys, leverage MLflow pyfunc for consistent model packaging, maintain clear separation of frontend and backend, and document workflows thoroughly for reproducibility.

## Overview
This application provides a simple web interface to generate summaries, health suggestions, and food recommendations from medical documents (PDFs or images) using a Flask backend and the OpenAI API.

## Models and Methods
- **OCR Pipeline:** Text extraction via Tesseract (`pytesseract`) and PyMuPDF (`fitz`).
- **Language Model:** Summarization and suggestions generated using OpenAI GPT-4 via the `openai` Python SDK.
- **Model Packaging:** Packaged as an MLflow pyfunc model for versioning and reproducibility.
- **API Hosting:** Deployed as a custom AI Studio Web App with built-in Swagger UI for API exploration.

## Project Structure
```
├── app.py              # Flask backend application
├── register_llama_model.py
├── config
│   └── secrets.yaml    # API keys (gitignored; see config/secrets.yaml.example)
├── data
│   └── raw             # Upload and temporary storage of documents
├── demo
│   └── index.html      # Static frontend for user interaction
├── docs
│   └── architecture.md # Application architecture details
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and usage instructions
```

## Setup

### Step 1: Create an AI Studio Project
1. Create a **New Project** in AI Studio.

### Step 2: Create a Workspace
1. Select a workspace image in AI Studio:
   - **Local GenAI**: Includes `llama.cpp` and is suited for GGUF llama models via llama.cpp.
   - **NeMo Framework**: Pre-configured with NVIDIA NeMo for LLMs, including `llama2-7b-chat.nemo` support.

### Step 3: Use the LLaMA2-7B-Chat Model in NeMo Framework
1. If you chose the **NeMo Framework** image, the `llama2-7b-chat.nemo` model should already be available in your workspace (e.g., under `/home/jovyan/datafabric/llama2-7b-chat/`).
2. Otherwise, you can download model from `models` section in `asset` in AI Studio to `datafabric/llama2-7b-chat/` in your workspace.

### Step 4: Verify Project Files
1. Clone or upload this repository to your workspace.
2. Ensure the following structure:
   ```
   ├── app.py
   ├── register_llama_model.py
   ├── config
   │   └── secrets.yaml
   ├── data
   │   └── raw
   ├── demo
   │   └── index.html
   ├── docs
   │   └── architecture.md
   ├── requirements.txt
   └── README.md
   ```

3. Edit `config/secrets.yaml` and set your OpenAI API key (if using OpenAI):

```yaml
OPENAI_API_KEY: <your_openai_api_key>
```

## MLflow & Deployment

### Register & Deploy the LLaMA2-7B Model

1. In your AI Studio workspace terminal, register the LLaMA2-7B model to MLflow:
   ```bash
   python register_llama_model.py \
     --model-path /home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf \
     --model-name llama2-7b \
     --register
   ```
   This logs and registers the model under the name `llama2-7b` in the MLflow registry.

2. Deploy the registered model as a Model Service in AI Studio:
   1. Navigate to **Deployments > New Service**.
   2. Choose **Model Service**, select **llama2-7b** from the MLflow registry, and configure compute (GPU/CPU).
   3. Start the deployment. AI Studio will provide a Swagger UI for the `/invocations` endpoint.
   4. Copy the **Service URL** for later.

## Usage

### Step 1: Deploy the Service
1. Navigate to **Deployments > New Service** in AI Studio.
2. Choose **Custom Web App**, provide a name (e.g., `MedicalDocSummarization`), and set `app.py` as the entry point.
3. Configure environment variables for model provider:

- **OpenAI API (default):**
  ```bash
  export MODEL_PROVIDER=openai
  export OPENAI_API_KEY=<your_openai_api_key>
  ```
- **Local LLaMA2-7B model:**
  ```bash
  export MODEL_PROVIDER=local
  export LOCAL_MODEL_PATH=/home/jovyan/datafabric/llama2-7b/ggml-model-f16-Q5_K_M.gguf
  ```
- **HP AI Studio MLflow Model Service:**
  ```bash
  export MODEL_PROVIDER=hp_studio
  ```
-  # URL for MLflow model service (from the deployed service's Swagger UI)
-  export MLFLOW_SERVER_URL=<service_url>
-  # Registered model name in MLflow registry
  ```bash
  export STUDIO_MODEL_NAME=llama2-7b
  ```
+ **NeMo LLM (NeMo Framework image):**
  ```bash
  export MODEL_PROVIDER=nemo
  # Path to local NeMo .nemo file
  export NEMO_MODEL_PATH=/home/jovyan/datafabric/llama2-7b-chat/llama2-7b-chat.nemo
  ```
+ **HP AI Studio MLflow Model Service:**
  ```bash
  export MODEL_PROVIDER=hp_studio
  ```
+  # URL for MLflow model service (from the deployed service's Swagger UI)
+  export MLFLOW_SERVER_URL=<service_url>
+  # Registered model name in MLflow registry
  ```bash
  export STUDIO_MODEL_NAME=llama2-7b
  ```
4. Start the deployment.
5. Once deployed, click the **Service URL** to access the demo UI.
6. Upload your medical documents and click **Generate Summary**.

## Judging and Testing

To evaluate the service:
1. Upload sample medical documents via the demo UI or send a POST request to `/invocations`:

```bash
curl -X POST "$SERVICE_URL/invocations" \
  -F "files=@sample_report.pdf"
```
2. Verify the JSON response contains `summaries`, `healthSuggestions`, and `foodSuggestions`.

## Open-source Repository

This project is open-source under the MIT License. View or contribute at:
https://github.com/nobelchowdary/medical-report-analyzer-healthy-habits-recommender

## Contact and Support
For issues or questions, please open an issue in the repository or contact the maintainer.