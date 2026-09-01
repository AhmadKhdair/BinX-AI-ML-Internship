# Week 7 - Day 3: RNNs and LSTMs for ECG Sequences

This folder contains the Day 3 notebook for Week 7. The work uses the MIT-BIH ECG heartbeat files from the provided archive and focuses on sequence modeling with `SimpleRNN` and `LSTM`.

The notebook treats each heartbeat as an ordered sequence of 187 ECG samples. The last column is the class label. The main requirement is the RNN vs LSTM comparison, but I also keep a dense baseline and a GRU run so the result is interpreted honestly.

## Files

| File | Description |
|---|---|
| `03_RNN_LSTM_Sequential_Data.ipynb` | Day 3 notebook for ECG sequence modeling, model comparison, order ablation, and final test evaluation |

## Dataset

The notebook expects the original ECG archive or the two MIT-BIH CSV files:

```text
mitbih_train.csv
mitbih_test.csv
```

The archive may also contain:

```text
ptbdb_normal.csv
ptbdb_abnormal.csv
```

For this notebook I use only the MIT-BIH files because they already provide a multiclass train/test setup.

Expected MIT-BIH structure:

```text
Rows: heartbeat samples
Columns 0-186: ordered ECG signal values
Column 187: class label
Classes: 0, 1, 2, 3, 4
```

Class mapping used in the notebook:

| Label | Class |
|---:|---|
| 0 | Normal beat |
| 1 | Supraventricular ectopic beat |
| 2 | Ventricular ectopic beat |
| 3 | Fusion beat |
| 4 | Unknown / unclassifiable beat |

## Notebook Scope

The notebook covers:

- dataset audit: shape, labels, missing values, signal range, and class distribution
- visual inspection of ECG heartbeat sequences
- train/validation split from `mitbih_train.csv`
- untouched final evaluation on `mitbih_test.csv`
- class-weight handling for the imbalanced labels
- dense baseline on the 187-value feature vector
- `SimpleRNN` baseline on `(187, 1)` sequences
- `LSTM` on the same split and training setup
- GRU as a small bonus experiment
- training curves for all trained models
- validation comparison using macro F1 as the main metric
- timestep-shuffling ablation to check whether order matters
- final test comparison for the selected recurrent model and the dense baseline
- classification report and confusion matrix for the selected recurrent model

## Evaluation Notes

Model selection between `SimpleRNN` and `LSTM` is based on validation macro F1, not on the test file.

Accuracy is reported, but it is not the main metric because the dataset is strongly imbalanced. Macro F1, balanced accuracy, per-class recall, precision, and the confusion matrix are all needed to understand the result.

The dense baseline is included on purpose. If it beats the recurrent models, the notebook says that directly. The Day 3 objective is to build and compare sequence models, not to pretend that an LSTM is always the best architecture.

The order-ablation section shuffles ECG timesteps inside each validation heartbeat. A large metric drop after shuffling supports the idea that the selected recurrent model uses waveform order. It does not prove that the recurrent model is the strongest possible ECG model.

## How to Run

In Google Colab:

1. Open `03_RNN_LSTM_Sequential_Data.ipynb`.
2. Use a GPU runtime if available.
3. Run the notebook from top to bottom.
4. When Colab asks for the dataset, upload the ECG archive.

Locally, place the archive or both MIT-BIH CSV files beside the notebook. The loader checks common archive names such as:

```text
archive (4).zip
archive.zip
heartbeat.zip
dataset.zip
```

You can also set the archive path with:

```bash
WEEK7_ECG_ARCHIVE=/path/to/archive.zip
```

Required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow notebook
```

## Day 3 Result

After running the notebook, the final saved output should show:

- the validation table for Dense baseline, SimpleRNN, LSTM, and GRU
- the selected required recurrent model
- the order-ablation result
- the test result for the selected recurrent model
- the dense baseline test result beside it
- the recurrent model classification report and confusion matrix

The final Markdown is generated from the actual metrics, so the written conclusion should stay aligned with the run output.
