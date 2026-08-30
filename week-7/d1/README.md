# Week 7 - Day 1: Sprint 2 Planning and Convolution Demo

This folder contains the Day 1 notebook for Week 7. The work uses the uploaded image dataset and focuses on the first CNN concepts from the sprint: why dense layers are inefficient for raw images, how convolution applies a filter, what a feature map represents, and why parameter sharing matters.

The notebook does not train a full CNN. That part belongs to the next Day 2 work.

## Files

| File | Description |
|---|---|
| `01_Sprint2_Convolution_Demo.ipynb` | Main Day 1 notebook |

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

The uploaded archive used for validation contains 13,879 RGB `.jpg` images:

| Split | Benign | Malignant |
|---|---:|---:|
| train | 6,289 | 5,590 |
| test | 1,000 | 1,000 |

The sampled images in the audit are `224 x 224 x 3`.

## Notebook Scope

The notebook covers:

- Sprint 2 goal and Day 1 backlog
- dataset structure and image-shape audit
- why flattening images into dense layers creates too many parameters
- manual 2D convolution with a hand-defined vertical edge filter
- feature-map visualization from a real dataset image
- stride and padding output-shape comparison
- architecture decision for the image dataset

## How to Run

In Google Colab:

1. Open `01_Sprint2_Convolution_Demo.ipynb`.
2. Run the cells from top to bottom.
3. When the notebook asks for the dataset archive, upload the zip file.

Locally, place the dataset archive beside the notebook using one of these names:

```text
archive.zip
archive (3).zip
dataset.zip
```

Or set the full archive path with `WEEK7_IMAGE_ARCHIVE`.

Required packages:

```bash
pip install numpy pandas matplotlib pillow notebook
```

## Day 1 Result

The selected architecture direction is a CNN-based image classifier because the input data is RGB image data. Convolution preserves local spatial structure and uses the same filter weights across the image, which is more suitable than flattening the pixels into a dense-only model.
