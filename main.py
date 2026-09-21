from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "amazon_helpfulness_tfidf_sgd.joblib"


# --------------------------------------------------
# Load trained ML pipeline
# --------------------------------------------------

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Amazon Review Helpfulness Classifier",
    description=(
        "Predicts whether an Amazon product review is "
        "Helpful or Not Helpful."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# Frontend setup
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)

templates = Jinja2Templates(
    directory=BASE_DIR / "templates"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class ReviewRequest(BaseModel):
    summary: str = ""
    text: str = Field(..., min_length=1)


# --------------------------------------------------
# User-facing homepage
# --------------------------------------------------

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.post("/predict")
def predict_review(review: ReviewRequest):

    # Recreate the same text structure used for training
    review_text = (
        review.summary.strip()
        + " "
        + review.text.strip()
    ).strip()

    prediction = int(
        model.predict([review_text])[0]
    )

    probabilities = model.predict_proba(
        [review_text]
    )[0]

    # Get class order from the trained SGD model
    class_labels = list(
        model.named_steps["sgd"].classes_
    )

    not_helpful_index = class_labels.index(0)
    helpful_index = class_labels.index(1)

    not_helpful_probability = float(
        probabilities[not_helpful_index]
    )

    helpful_probability = float(
        probabilities[helpful_index]
    )

    return {
        "prediction": prediction,
        "label": (
            "Helpful"
            if prediction == 1
            else "Not Helpful"
        ),
        "probability_not_helpful": round(
            not_helpful_probability,
            4
        ),
        "probability_helpful": round(
            helpful_probability,
            4
        )
    }