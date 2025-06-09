# Model Details

## How the Medical Summarization Works
- Extracts text from uploaded documents (PDFs or images) using OCR and PDF parsing.
- Invokes the OpenAI Chat API to generate concise, patient-friendly summaries, health suggestions, and food suggestions.
- Packaged as an MLflow pyfunc model for versioning and deployment.

# MLflow & Swagger Integration

## Deployment & Swagger UI
- After registration, deploy the model as a **Model Service** in AI Studio.
- AI Studio automatically generates a Swagger UI for the `/invocations` endpoint, enabling easy API exploration.

---

# API Endpoints

## How the Frontend Interacts with the Backend
- The system provides an `/invocations` API endpoint for processing user-uploaded documents.
- The frontend sends a POST request containing the files to this endpoint.
- The backend processes the request and returns a structured JSON response.

## Project Structure

```
├── app.py              # Flask backend application
├── data
│   └── raw             # Upload and temporary storage of documents
├── demo
│   └── index.html      # Static frontend for user interaction
├── docs
│   └── architecture.md # This document
├── requirements.txt    # Python dependencies
└── README.md           # Project overview and usage instructions
```