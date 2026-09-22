# Amazon Review Helpfulness Classifier

An end-to-end NLP application that predicts whether an Amazon Fine Food Review is likely to be considered **Helpful** or **Not Helpful** based on its written content.

**Live Demo:** https://amazon-review-helpfulness-api.onrender.com/  
**API Docs:** https://amazon-review-helpfulness-api.onrender.com/docs  
**Notebook:** [reviews_Project_Final.ipynb](./reviews_Project_Final.ipynb)

## Project Overview

A review is labeled **Helpful (1)** when at least 50% of its recorded helpfulness votes are positive and **Not Helpful (0)** otherwise. Reviews with no helpfulness votes are excluded because their helpfulness ratio is undefined.

The project compares **TF-IDF** and **200-dimensional Word2Vec** text representations across Logistic Regression, Random Forest, Linear SVM, and SGDClassifier models. Because the cleaned dataset is class-imbalanced, model selection emphasizes **Macro F1** rather than accuracy alone.

## Dataset

**Amazon Fine Food Reviews — SNAP / Kaggle**  
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data

The original dataset contains **568,454 reviews**. After removing reviews without helpfulness votes and auditing exact duplicate review text, the final modeling dataset contains **209,960 unique, label-consistent reviews**.

The raw dataset is not stored in this repository. To reproduce the notebook, download `Reviews.csv` from the Kaggle link above and place it in the notebook's working directory.

## Data Quality and Leakage Control

- Identified **88,325 exact duplicate review-text rows** before cleanup.
- Identified **117 review-text groups with conflicting target labels**.
- Removed conflicting text groups and retained one instance of each remaining exact review text.
- Final class distribution:
  - Helpful: **176,467**
  - Not Helpful: **33,493**

This cleanup reduces exact duplicate-text leakage across the random train/test split.

## Final Deployed Model

The deployed model is a scikit-learn Pipeline containing:

- **TF-IDF:** 10,000 features
- **SGDClassifier**
  - `loss="log_loss"`
  - `alpha=1e-5`
  - `penalty="l1"`
  - `class_weight={0: 2.5, 1: 1}`

### Performance

| Metric | Result |
|---|---:|
| Held-out Accuracy | 0.804 |
| Held-out Macro F1 | 0.623 |
| Held-out Class 0 F1 | 0.363 |
| 5-Fold CV Accuracy | 0.810 ± 0.003 |
| 5-Fold CV Macro F1 | 0.627 ± 0.005 |
| 5-Fold CV Class 0 F1 | 0.365 ± 0.011 |

A majority-class baseline achieves approximately **84.05% accuracy** while completely missing the Not Helpful class, which is why Macro F1 is the primary comparison metric.

## Application Architecture

```text
User
  ↓
HTML / CSS / JavaScript
  ↓
FastAPI REST API
  ↓
Saved TF-IDF + SGDClassifier Pipeline
  ↓
Helpful / Not Helpful + Model-Estimated Probabilities
  ↓
Docker Container
  ↓
Render
```

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `/` | User-facing prediction interface |
| `/predict` | REST prediction endpoint |
| `/docs` | Interactive Swagger API documentation |
| `/health` | Service health check |

### Example API Request

```json
{
  "summary": "Great product",
  "text": "The product arrived quickly and worked exactly as expected."
}
```

## Technology Stack

Python, pandas, NumPy, scikit-learn, Gensim, TF-IDF, Word2Vec, Joblib, FastAPI, Jinja2, HTML, CSS, JavaScript, Docker, Render

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Run with Docker

Build:

```bash
docker build -t amazon-helpfulness-api .
```

Run:

```bash
docker run --rm -p 8000:8000 amazon-helpfulness-api
```

Then open:

```text
http://127.0.0.1:8000
```

## Notes and Limitations

- The task predicts **review helpfulness**, not product sentiment. A negative product review can still be helpful to shoppers.
- The deployed TF-IDF representation is based on individual word features and therefore has limited ability to model complex context, compositional meaning, and negation.
- Exact duplicate-text leakage was addressed, but other similarities such as paraphrases, repeated users, or product-level relationships may remain.
