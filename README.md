# Email-Intent-Classifer

A beginner-friendly machine learning project that classifies incoming emails by intent. It ships with a toy dataset, training pipeline, REST API, and CLI so you can run predictions immediately.

## Features
- **ML pipeline**: TF-IDF vectorization + Logistic Regression.
- **Toy dataset**: included directly in the training script for quick starts.
- **Saved model**: serialized with `joblib` to `data/processed/model.joblib`.
- **Confidence scores**: predictions include both label and probability.
- **FastAPI service**: `POST /predict` returns `{ label, confidence }`.
- **CLI**: classify emails from your terminal with one command.

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
```

### Train the model
```bash
python -m src.train
```
This will create `data/processed/model.joblib` using the bundled toy dataset.

### Run the CLI
```bash
python -m src.cli "Your payment is due tomorrow."
```
Example output:
```
Prediction: billing (confidence: 0.91)
```

### Start the API
```bash
uvicorn src.api:app --reload
```
Then send a request:
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "Can we move the project sync to 3pm?"}'
```
Sample response:
```json
{"label": "meeting", "confidence": 0.89}
```

## Project Structure
```
src/
 ├─ train.py   # builds & saves the ML model
 ├─ model.py   # loads the model + runs predictions
 ├─ api.py     # FastAPI deployment
 └─ cli.py     # command-line classifier

data/processed/model.joblib  # trained model artifact
```

## Roadmap Ideas
- Add more intent labels and training data.
- Secure the API with authentication and rate limiting.
- Deploy to platforms like Render, Railway, or Vercel.
- Add a UI that consumes the API.
- Track predictions in a database for analytics.
