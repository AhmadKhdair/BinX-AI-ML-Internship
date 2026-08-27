# Week 6 - Day 4: Training a Neural Network with Keras

This folder contains the Day 4 notebook for Week 6 of the BinX AI/ML Internship.

The work continues from the Heart Disease Classification project and keeps the Day 1 held-out test split frozen until the final comparison.

## Notebook

`04_Keras_Neural_Network.ipynb`

## Covered Work

- same `heart.csv` dataset and Day 1 cleaning rules;
- same stratified 80/20 held-out test split;
- validation split taken only from the original training portion;
- training-only imputation, scaling, one-hot encoding, and the same engineered features;
- Keras `Sequential` binary classifier with `Dense` layers;
- sigmoid output and binary cross-entropy;
- Adam optimizer;
- 50 epochs with `batch_size=32`;
- training and validation loss/accuracy curves;
- second run with `BatchNormalization` and `Dropout(0.3)`;
- validation-based comparison of the two runs;
- final evaluation on the frozen test set;
- comparison with the Day 1 tuned Random Forest.

## Results

The batch-normalized/dropout model had the better validation run, but both networks showed overfitting later in training.

| Metric | Day 1 Tuned Random Forest | Neural Network |
|---|---:|---:|
| Accuracy | 0.8696 | 0.8587 |
| Precision | 0.8679 | 0.8585 |
| Recall | 0.9020 | 0.8922 |
| F1 | 0.8846 | 0.8750 |
| ROC-AUC | 0.9173 | 0.9014 |

The neural network is close to the classical baseline but does not beat it on the held-out test set.

## Dataset

`final-project/heart-disease-classification/data/heart.csv`

## Environment

The notebook was run with Python 3.12 and TensorFlow/Keras.

Required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```
