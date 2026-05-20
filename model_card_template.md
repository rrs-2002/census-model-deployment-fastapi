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
The following metrics were computed on the 20% held-out test set:
- **Precision:** 0.7419
- **Recall:** 0.6384
- **F-beta (F1):** 0.6863

### Slice Performance
Model performance was also evaluated on slices of each categorical feature,
using only the test set (to avoid data leakage from the training data).
Full results are in `slice_output.txt`. Selected examples:

| Slice | Count | Precision | Recall | F1 |
|---|---|---|---|---|
| education = Bachelors | 1,053 | 0.7523 | 0.7289 | 0.7404 |
| education = HS-grad | 2,085 | 0.6594 | 0.4377 | 0.5261 |
| education = Doctorate | 77 | 0.8644 | 0.8947 | 0.8793 |
| sex = Female | 2,126 | 0.7229 | 0.5150 | 0.6015 |
| sex = Male | 4,387 | 0.7445 | 0.6599 | 0.6997 |
| race = White | 5,595 | 0.7404 | 0.6373 | 0.6850 |
| race = Black | 599 | 0.7273 | 0.6154 | 0.6667 |

## Ethical Considerations
- The dataset contains sensitive attributes like race and sex which could
  introduce bias in predictions.
- Model performance varies across demographic slices (see `slice_output.txt`).
  For example, recall is notably lower for females (0.52) than males (0.66),
  suggesting the model may underpredict high income for women.
- This model should NOT be used for real-world financial decisions.

## Caveats and Recommendations
- The data is from the 1994 Census and may not reflect current income
  distributions.
- Performance on underrepresented groups (e.g., certain native-country
  values with very few samples) may be unreliable.
- For production use, regular retraining and bias auditing would be needed.

