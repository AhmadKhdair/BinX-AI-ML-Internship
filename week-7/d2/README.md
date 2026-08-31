# Week 7 - Day 2: CNN Training and Transfer Learning

This folder contains the Day 2 notebook for Week 7. The work continues from the Day 1 convolution demo and moves from a single filter example to full image-classification training on the `Benign` / `Malignant` dataset.

The notebook compares a CNN trained from scratch, the same CNN with data augmentation, frozen MobileNetV2 transfer learning, and a small fine-tuning extension. Model selection is based on validation performance only. The test split is used once at the end for the final report.

## Files

| File | Description |
|---|---|
| `02_CNN_Transfer_Learning_FINAL.ipynb` | Final Day 2 notebook with saved outputs and result-based Markdown |

## Dataset

The notebook expects a zipped image dataset with this structure:

```text
train/
  Benign/
  Malignant/
test/
  Benign/
  Malignant/
```

The uploaded archive used in the run contains 13,879 images:

| Split | Benign | Malignant |
|---|---:|---:|
| train | 6,289 | 5,590 |
| test | 1,000 | 1,000 |

The original `test/` directory is kept separate. The notebook creates the validation split only from the original training directory.

```text
Training images: 9,504
Validation images: 2,375
Test images: 2,000
Image size: 160 x 160
Batch size: 32
Validation split: 20%
Positive class: Malignant
```

## Notebook Scope

The notebook covers:

- dataset audit and class-count check
- train / validation / test dataset creation
- data augmentation preview
- CNN from scratch with convolution, pooling, flattening, dense, dropout, and sigmoid output
- CNN with augmentation using the same base architecture
- transfer learning with frozen `MobileNetV2`
- fine-tuning the last part of `MobileNetV2` as an extra experiment
- validation comparison across experiments
- final held-out test evaluation for the selected model
- classification report and confusion matrix

## Validation Results

The comparison below is taken from the saved notebook outputs. The table is sorted by validation accuracy.

| Experiment | Epochs Run | Best Epoch | Val Loss | Train Accuracy | Val Accuracy | Val Recall | Val AUC | Gap | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN from scratch | 12 | 9 | 0.2843 | 90.43% | 87.83% | 0.8786 | 0.9534 | 2.59% | 2.84 min |
| MobileNetV2 fine-tuned | 7 | 4 | 0.3075 | 88.11% | 86.78% | 0.8218 | 0.9444 | 1.33% | 2.07 min |
| CNN + augmentation | 15 | 14 | 0.2855 | 87.57% | 86.65% | 0.8873 | 0.9502 | 0.92% | 3.25 min |
| MobileNetV2 frozen | 12 | 9 | 0.3324 | 85.44% | 85.09% | 0.8017 | 0.9326 | 0.34% | 3.04 min |

The best required Day 2 approach was the CNN from scratch, with `87.83%` validation accuracy and `0.9534` validation AUC.

Augmentation reduced the train-validation gap from `2.59%` to `0.92%`, but it did not improve the best validation accuracy. Frozen MobileNetV2 was stable but weaker. Fine-tuning improved MobileNetV2, but it still did not beat the scratch CNN.

## Final Test Result

The selected model was the CNN from scratch. It was evaluated once on the held-out test split after validation-based model selection.

| Metric | Value |
|---|---:|
| Accuracy | 90.50% |
| ROC-AUC | 0.9685 |
| Malignant precision | 92.72% |
| Malignant recall | 87.90% |
| Malignant F1-score | 90.25% |

Confusion matrix:

| True Label | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| Benign | 931 | 69 |
| Malignant | 121 | 879 |

The model is strong overall, but the `121` false negatives are important because they are malignant images predicted as benign. In a real medical workflow, this would justify threshold tuning or a recall-focused operating point using the validation set.

## How to Run

In Google Colab:

1. Open `02_CNN_Transfer_Learning_FINAL.ipynb`.
2. Use a GPU runtime if available.
3. Run the cells from top to bottom.
4. When the notebook asks for the dataset archive, upload the zip file.

Locally, place the dataset archive beside the notebook using one of these names:

```text
archive.zip
archive (3).zip
dataset.zip
```

Or set the full archive path with `WEEK7_IMAGE_ARCHIVE`.

Required packages:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow notebook
```

## Day 2 Result

The Day 2 requirements are covered: a full CNN was trained, augmentation was compared against the scratch model, frozen MobileNetV2 transfer learning was tested, training time and validation metrics were compared, and the final selected model was evaluated on the untouched test split.

For this run, the scratch CNN is the final selected model because it had the strongest validation accuracy and AUC. Transfer learning was still useful to test, but the frozen MobileNetV2 head underfit this dataset and fine-tuning only partially closed the gap.
