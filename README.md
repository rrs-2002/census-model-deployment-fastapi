# Census Income Prediction — ML Model Deployment with FastAPI

**GitHub Repository:** https://github.com/rrs-2002/census-model-deployment-fastapi

A machine learning classification model trained on [UCI Census Income Data](https://archive.ics.uci.edu/ml/datasets/census+income) to predict whether an individual's annual income exceeds $50K. The model is deployed as a REST API using **FastAPI** with full CI/CD via **GitHub Actions** and **Render**.

---

## Project Structure

```
.
├── .flake8                        # Flake8 linting configuration
├── .github/
│   └── workflows/
│       └── main.yml               # GitHub Actions CI pipeline (pytest + flake8)
├── .gitignore
├── data/
│   └── census.csv                 # Cleaned UCI Census Income dataset
├── model/
│   ├── trained_model.pkl          # Trained RandomForestClassifier
│   ├── encoder.pkl                # Fitted OneHotEncoder
│   └── lb.pkl                     # Fitted LabelBinarizer
├── screenshots/
│   ├── continuous_integration.png # GitHub Actions CI passing
│   ├── continuous_deployment.png  # Render auto-deploy setting
│   ├── example.png                # FastAPI docs with example payload
│   ├── live_get.png               # Browser hitting live root endpoint
│   └── live_post.png              # Terminal output from live_post.py
├── starter/
│   ├── __init__.py
│   ├── train_model.py             # Training pipeline script
│   └── ml/
│       ├── __init__.py
│       ├── data.py                # Data processing (process_data function)
│       ├── model.py               # ML model functions (train, inference, metrics)
│       ├── slice_performance.py   # Compute metrics on data slices
│       └── test_model.py          # Unit tests for ML functions (3 tests)
├── main.py                        # FastAPI application (GET / and POST /predict)
├── test_api.py                    # API tests (1 GET + 2 POST tests)
├── live_post.py                   # Script to query the live deployed API
├── slice_output.txt               # Model performance on categorical feature slices
├── model_card_template.md         # Model Card documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── sanitycheck.py                 # Provided script to validate API test cases
├── dvc_on_heroku_instructions.md  # Reference: DVC on Heroku (not used)
└── README.md                      # This file
```

---

## Environment Setup

- **Python:** 3.12
- **Environment Manager:** Conda (`ml-core` environment)

```bash
conda activate ml-core
pip install -r requirements.txt
```

---

## Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Hyperparameters:** `n_estimators=100`, `random_state=42`
- **Task:** Binary classification — predict `<=50K` or `>50K` salary
- **Data:** UCI Census Income Dataset (~32,561 rows, 14 features)
- **Train/Test Split:** 80/20

### Training

```bash
python -m starter.train_model
```

### Slice Performance

```bash
python -m starter.ml.slice_performance
```

Output is saved to `slice_output.txt`.

---

## API

Built with **FastAPI**. Provides two endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/`      | Returns a welcome message |
| POST   | `/predict` | Accepts census data as JSON, returns income prediction |

### Run Locally

```bash
uvicorn main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

### Live API

Deployed on **Render**: https://census-income-api-xang.onrender.com

```bash
python live_post.py
```

---

## Testing

6 total tests (3 ML + 3 API):

```bash
# Run all tests
pytest -v

# Run only ML tests
pytest starter/ml/test_model.py -v

# Run only API tests
pytest test_api.py -v

# Lint check
flake8 .

# Sanity check for API tests
python sanitycheck.py
```

---

## CI/CD

- **CI:** GitHub Actions runs `pytest` and `flake8` on every push to `main`.
- **CD:** Render auto-deploys after CI checks pass.

---

## Screenshots

| Screenshot | Description |
|---|---|
| `screenshots/continuous_integration.png` | GitHub Actions CI passing |
| `screenshots/continuous_deployment.png` | Render auto-deploy enabled |
| `screenshots/example.png` | FastAPI docs with example payload |
| `screenshots/live_get.png` | Browser GET on live root endpoint |
| `screenshots/live_post.png` | Terminal output from live POST |
