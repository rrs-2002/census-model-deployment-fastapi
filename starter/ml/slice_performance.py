"""
Compute model performance on slices of categorical features.
"""
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import inference, compute_model_metrics


def compute_slice_metrics(data, cat_features, model, encoder, lb):
    """
    Compute performance metrics for each unique value of each
    categorical feature.

    Parameters
    ----------
    data : pd.DataFrame
        The full (cleaned) dataset with labels.
    cat_features : list[str]
        List of categorical feature column names.
    model : trained sklearn model
    encoder : fitted OneHotEncoder
    lb : fitted LabelBinarizer

    Returns
    -------
    results : list[str]
        Lines of text, each describing the metrics for one slice.
    """
    results = []
    for feature in cat_features:
        unique_values = data[feature].unique()
        for value in unique_values:
            # Filter data for this slice
            slice_data = data[data[feature] == value]

            if len(slice_data) == 0:
                continue

            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=cat_features,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            line = (
                f"{feature} = {value} | "
                f"Count: {len(slice_data)} | "
                f"Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | "
                f"F-beta: {fbeta:.4f}"
            )
            results.append(line)
    return results


if __name__ == "__main__":
    # Load data
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    data = pd.read_csv(os.path.join(base_dir, 'data', 'census.csv'))

    # Split to get the same test split as train_model.py
    _, test_data = train_test_split(data, test_size=0.20, random_state=42)

    # Load saved model artifacts
    model_dir = os.path.join(base_dir, 'model')
    with open(os.path.join(model_dir, 'trained_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'encoder.pkl'), 'rb') as f:
        encoder = pickle.load(f)
    with open(os.path.join(model_dir, 'lb.pkl'), 'rb') as f:
        lb = pickle.load(f)

    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]

    results = compute_slice_metrics(test_data, cat_features, model, encoder, lb)

    # Write to file
    output_path = os.path.join(base_dir, 'slice_output.txt')
    with open(output_path, 'w') as f:
        for line in results:
            print(line)
            f.write(line + '\n')

    print(f"\nSlice metrics written to {output_path}")

