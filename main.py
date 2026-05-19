"""
FastAPI application for Census income prediction.
"""
import os
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load model artifacts at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

with open(os.path.join(MODEL_DIR, "trained_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)
with open(os.path.join(MODEL_DIR, "lb.pkl"), "rb") as f:
    lb = pickle.load(f)

# Import the processing functions
from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI(
    title="Census Income Prediction API",
    description="Predicts whether income exceeds $50K/yr based on census data.",
    version="1.0.0",
)

# Define categorical features (same list used in training)
CAT_FEATURES = [
    "workclass", "education", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]


class CensusData(BaseModel):
    """Pydantic model for census data input with example values."""
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    model_config = {
        "populate_by_name": True,   # allows using Python names OR aliases
        "json_schema_extra": {
            "examples": [
                {
                    "age": 39,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education-num": 13,
                    "marital-status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital-gain": 2174,
                    "capital-loss": 0,
                    "hours-per-week": 40,
                    "native-country": "United-States",
                }
            ]
        },
    }


@app.get("/")
async def root():
    """GET root endpoint — returns a welcome message."""
    return {"message": "Welcome to the Census Income Prediction API!"}


@app.post("/predict")
async def predict(data: CensusData):
    """POST predict endpoint — runs model inference on input data."""
    # Convert Pydantic model to a dict using aliases (hyphenated names)
    data_dict = data.model_dump(by_alias=True)

    # Create a single-row DataFrame matching the training data format
    df = pd.DataFrame([data_dict])

    # Process the data (no label since we're doing inference)
    X, _, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run inference
    preds = inference(model, X)

    # Convert prediction back to human-readable label
    prediction_label = lb.inverse_transform(preds)[0]

    return {"prediction": prediction_label}
