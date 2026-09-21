# Amazon Review Helpfulness Classifier

A machine learning application that predicts whether an Amazon product review is likely to be considered Helpful or Not Helpful.

## Machine Learning Model

The deployed model uses:

- TF-IDF text representation
- 10,000 features
- SGDClassifier
- Log-loss objective
- L1 regularization
- Class weighting for imbalanced classification

The final model achieved:

- Held-out Accuracy: 0.804
- Held-out Macro F1: 0.623
- 5-Fold CV Macro F1: 0.627 ± 0.005

## Application

The trained scikit-learn pipeline is exposed through a FastAPI REST API with a user-friendly HTML/CSS frontend.

### Endpoints

- `/` - Web application
- `/predict` - Prediction API
- `/docs` - FastAPI Swagger documentation
- `/health` - Health check

## Technology Stack

- Python
- scikit-learn
- TF-IDF
- SGDClassifier
- FastAPI
- HTML/CSS/JavaScript
- Joblib
- Docker

## Architecture

User Interface → FastAPI → TF-IDF + SGDClassifier → Prediction

The application is containerized with Docker for reproducible deployment across environments.