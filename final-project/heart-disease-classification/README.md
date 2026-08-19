# Heart Disease Classification

This project builds a classical machine learning model to classify whether a patient record is associated with heart disease or no heart disease using clinical and cardiac examination features.

The target is `HeartDisease`:

- `0`: no heart disease
- `1`: heart disease

This is a classification project, not a clinically validated diagnostic system and not a future disease prediction study.

## Project Structure

```text
heart-disease-classification/
├── README.md
├── requirements.txt
├── data/
│   ├── heart.csv
│   └── data_dictionary.csv
├── notebooks/
│   └── Heart_Disease_Classification_Final.ipynb
└── outputs/
    ├── figures/
    └── metrics/
```

## Dataset

The dataset contains 918 rows and 12 columns:

- 11 input features
- 1 binary target column: `HeartDisease`

The features include age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG result, maximum heart rate, exercise-induced angina, oldpeak, and ST slope.

## Main Workflow

The notebook follows the project from top to bottom:

1. Load the dataset and inspect the schema.
2. Check duplicates, missing values, target balance, and suspicious values.
3. Clean invalid medical values:
   - `RestingBP = 0` is treated as invalid.
   - `Cholesterol = 0` is treated as missing/unknown, not as real cholesterol.
4. Run EDA and descriptive statistics.
5. Split the data into train and test sets before model tuning.
6. Build preprocessing and modeling pipelines.
7. Compare Logistic Regression, Decision Tree, and Random Forest using cross-validation on the training set only.
8. Tune the best candidates using `GridSearchCV`.
9. Evaluate the final selected model once on the held-out test set.
10. Save metrics and figures under `outputs/`.

## How To Run

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook notebooks/Heart_Disease_Classification_Final.ipynb
```

The notebook creates the output files automatically.

## Models Used

Only classical machine learning models are used:

- Logistic Regression
- Decision Tree
- Random Forest

The project does not include PCA, clustering, or unsupervised learning in the current version because that part is outside the current requirements. The structure still leaves room to add it later after preprocessing.

## Final Result

After cross-validation and tuning on the training set, the selected final model is Random Forest.

Held-out test metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.870 |
| Precision | 0.868 |
| Recall | 0.902 |
| F1-score | 0.885 |
| ROC-AUC | 0.917 |

The saved metric files are available under `outputs/metrics/`, and the plots are available under `outputs/figures/`.

## Limitations

- The dataset is a merged heart disease dataset, and the source of each row is not included as a column.
- `HeartDisease` represents presence/absence of heart disease in the dataset, not future risk.
- Some zero values are not medically valid and are handled during preprocessing.
- The model should be treated as an educational ML classifier, not as medical advice or a diagnostic tool.
