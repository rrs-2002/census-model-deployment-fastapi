# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model type:** Random Forest Classifier (scikit-learn)
- **Python version:** 3.12
- **Hyperparameters:** n_estimators=100, random_state=42, all others default
- **Developer:** [Ritu Raj Singh]
- **Date:** [19th May, 2026]

## Intended Use
This model predicts whether an individual's annual income exceeds $50,000
based on census data. It is intended for educational purposes as part of
the Udacity ML DevOps Engineer Nanodegree.

## Training Data
- **Source:** UCI Census Income Dataset (census.csv)
- **Size:** 80% of ~32,561 rows used for training
- **Features:** 14 attributes including age, workclass, education,
  marital-status, occupation, relationship, race, sex, capital-gain,
  capital-loss, hours-per-week, and native-country
- **Label:** salary (<=50K or >50K)

## Evaluation Data
- **Source:** Same UCI Census Income Dataset
- **Size:** 20% of ~32,561 rows held out for testing

## Metrics
_Actual numbers I got from running train_model.py:_
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F-beta (F1):** 0.6863

## Ethical Considerations
- The dataset contains sensitive attributes like race and sex which could
  introduce bias in predictions.
- Model performance varies across demographic slices (see slice_output.txt).
- This model should NOT be used for real-world financial decisions.

## Caveats and Recommendations
- The data is from the 1994 Census and may not reflect current income
  distributions.
- Performance on underrepresented groups (e.g., certain native-country
  values with very few samples) may be unreliable.
- For production use, regular retraining and bias auditing would be needed.
