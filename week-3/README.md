# Week 3 — Supervised Learning

This week introduces supervised machine learning with Scikit-learn through two core problem types: regression and classification. The work progresses from understanding how a model learns from labeled data to building, evaluating, and comparing multiple classifiers on a real COVID-19 testing dataset.

## Week Objectives

By the end of this week, the following skills were practiced:

- Distinguishing regression from classification problems
- Separating features from the target variable
- Creating reproducible train/test splits
- Training baseline supervised learning models
- Evaluating regression models with MSE, RMSE, and R²
- Preparing categorical and numeric features for classification
- Training Logistic Regression, Decision Tree, Random Forest, SVM, and k-NN models
- Evaluating classification models with accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices
- Comparing models according to the problem objective rather than relying on one metric
- Documenting modeling decisions, results, and limitations in Jupyter notebooks

## Weekly Progress

| Day | Topic | Main Work |
|---|---|---|
| [Day 1](./d1/) | Linear Regression | Built a regression workflow using synthetic data, separated signal from noise, trained a linear model, compared learned and true coefficients, and evaluated residual errors. |
| [Day 2](./d2/) | Classification Foundations | Prepared the COVID-19 dataset for classification, inspected the target values, encoded the binary target, reviewed class balance, and prepared the input features. |
| [Day 3](./d3/) | Logistic Regression | Built a binary classification workflow with Logistic Regression and evaluated predictions beyond accuracy. |
| [Day 4](./d4/) | Classifier Comparison | Trained and compared Logistic Regression, Decision Tree, Random Forest, SVM, and k-NN on the same classification task. |
| [Day 5](./d5/) | Supervised Learning Mini-Project | Combined data inspection, preprocessing, baseline modeling, classifier comparison, detailed evaluation, and result interpretation in a complete mini-project. |

## Main Concepts

### Supervised Learning

Supervised learning uses labeled examples to learn a relationship between input features and a known target.

- **Regression** predicts a continuous numeric value.
- **Classification** predicts one of a set of categories.

### Regression Workflow

The regression work followed this process:

```text
Generate or load data
        ↓
Separate features and target
        ↓
Create training and testing sets
        ↓
Train a baseline regression model
        ↓
Generate predictions
        ↓
Evaluate errors and residuals
```

The regression model was evaluated with:

| Metric | Interpretation |
|---|---|
| MSE | Average squared prediction error |
| RMSE | Prediction error expressed in the target's original unit |
| R² | Proportion of target variation explained by the model |

### Classification Workflow

The classification work followed this process:

```text
Load individual testing records
        ↓
Inspect and clean the target
        ↓
Prepare numeric and categorical features
        ↓
Create a stratified train/test split
        ↓
Fit preprocessing and models on training data
        ↓
Evaluate predictions on unseen test data
        ↓
Compare models and interpret errors
```

## Dataset

The classification exercises use individual COVID-19 testing records located in:

```text
corona dataset/
└── corona_tested_individuals_ver_006.english.csv
```

Each row represents one tested individual. The available information includes:

- Cough
- Fever
- Sore throat
- Shortness of breath
- Headache
- Age group
- Gender
- Test indication
- COVID-19 test result

The original dataset contains **278,848 records**. After excluding the `other` result category, **274,956 records** remain for binary classification:

| Target Class | Records | Percentage |
|---|---:|---:|
| Negative | 260,227 | 94.64% |
| Positive | 14,729 | 5.36% |

The target is strongly imbalanced. A classifier that always predicts the negative class can therefore achieve high accuracy while failing to identify every positive case.

## Models Used

### Regression

- Linear Regression

### Classification

- Dummy Classifier
- Logistic Regression
- Decision Tree
- Random Forest
- Linear Support Vector Machine
- k-Nearest Neighbors

The Dummy Classifier is used as a baseline. It provides a minimum reference point that trained models should improve upon.

## Classification Metrics

| Metric | What It Measures |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision | How many predicted positive cases were actually positive |
| Recall | How many actual positive cases were detected |
| F1-score | Balance between precision and recall |
| ROC-AUC | Ability to rank positive cases above negative cases across thresholds |
| Confusion Matrix | Counts of correct and incorrect predictions by class |

Accuracy is not used alone because the negative class represents 94.64% of the binary dataset.

## Day 5 Mini-Project Results

A reproducible stratified sample of 40,000 records was used for the final classifier comparison. The sample was divided into 32,000 training records and 8,000 testing records.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| k-Nearest Neighbors | **0.9694** | **0.7788** | 0.5991 | **0.6772** | 0.8654 |
| Decision Tree | 0.9299 | 0.4165 | 0.7669 | 0.5398 | 0.9034 |
| Linear SVM | 0.9282 | 0.4081 | 0.7506 | 0.5287 | 0.9044 |
| Random Forest | 0.9165 | 0.3716 | **0.8065** | 0.5088 | **0.9099** |
| Logistic Regression | 0.9166 | 0.3672 | 0.7669 | 0.4966 | 0.9059 |
| Dummy Baseline | 0.9464 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

### Selected Model

k-Nearest Neighbors achieved the highest test-set F1-score and was selected according to the evaluation rule defined for the mini-project.

Its confusion matrix contained:

| Outcome | Count |
|---|---:|
| True Negatives | 7,498 |
| False Positives | 73 |
| False Negatives | 172 |
| True Positives | 257 |

The result illustrates an important engineering trade-off:

- k-NN produced the strongest precision–recall balance.
- Random Forest detected a larger share of positive cases and achieved the highest ROC-AUC.
- The preferred model would change if the main objective changed from balanced performance to maximizing positive-case detection.

## Repository Structure

```text
week-3/
├── README.md
├── corona dataset/
│   └── corona_tested_individuals_ver_006.english.csv
├── d1/
├── d2/
├── d3/
├── d4/
└── d5/
    ├── 05_supervised_learning_classification_mini_project_final.ipynb
    └── README.md
```

Each daily directory contains the notebook and documentation for that stage of the week.

## Requirements

- Python 3.10+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

Install the required packages inside the project virtual environment:

```powershell
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Running the Notebooks

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
jupyter notebook
```

Open the `week-3` directory and run the selected notebook from top to bottom.

The Day 5 notebook expects this relative dataset path:

```python
DATA_PATH = "../corona dataset/corona_tested_individuals_ver_006.english.csv"
```

The folder structure must remain unchanged for the path to work.

## Reproducibility

The experiments use a fixed random seed:

```python
RANDOM_STATE = 42
```

Stratified sampling and splitting are used in the classification project to preserve the original target-class proportions.

## Key Engineering Takeaways

- A model should always be compared with a simple baseline.
- Accuracy can be misleading when the target is imbalanced.
- Preprocessing must be learned from training data rather than the complete dataset.
- Scaling is especially important for distance-based and margin-based models.
- Model selection depends on the cost of different errors.
- A model with the highest F1-score is not automatically the best model for every operational objective.
- Reported metrics must be accompanied by an interpretation of false positives and false negatives.

## Limitations

- The classification features are limited to five symptoms and a small number of demographic fields.
- Age and gender contain missing values.
- The final comparison uses a stratified sample rather than all available records.
- The reported results are based on one train/test split.
- Cross-validation and hyperparameter tuning are reserved for the next stage of the training.
- The COVID-19 classification work is an educational machine learning exercise and is not a medical diagnostic system.

## Next Step

Week 4 extends this work with model evaluation, cross-validation, feature engineering, hyperparameter tuning, and more structured Scikit-learn pipelines.
