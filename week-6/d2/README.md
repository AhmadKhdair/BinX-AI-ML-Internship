# Week 6 - Day 2: Activation Functions, Forward Propagation & Loss

This folder contains the Day 2 notebook for Week 6 of the BinX AI/ML Internship.

The notebook continues from the Heart Disease Classification project used on Day 1. The focus is the forward side of a neural network: why activation functions are needed, how values move through the layers, and how the final prediction is compared with the target using a suitable loss function.

## Notebook

`02_Activations_Forward_Pass.ipynb`

## What I Covered

The notebook includes:

- why stacked affine layers still need nonlinear activation functions;
- ReLU, sigmoid, tanh, and softmax;
- plots for ReLU, sigmoid, and tanh over the same input range;
- the practical role of each activation function;
- the dying ReLU limitation and the idea behind Leaky ReLU;
- the difference between binary, multi-class, and multi-label output behavior;
- MSE, binary cross-entropy, and categorical cross-entropy;
- the correct output activation and loss for the Heart Disease project;
- a complete two-layer forward pass calculated manually;
- NumPy verification of the same forward pass, including matrix shapes;
- the final prediction and its binary cross-entropy loss.

## Activation and Loss Choice for the Project

The project target is binary:

- `0` - no heart disease
- `1` - heart disease

The neural-network output should therefore use one neuron with a sigmoid activation:

```python
Dense(1, activation="sigmoid")
```

The matching loss is binary cross-entropy.

This choice keeps the output compatible with the binary target while preserving the prediction as a continuous score during training. A classification threshold is applied separately when converting that score to class `0` or `1`.

## Forward Pass Example

The notebook uses a small synthetic example so every step can be checked directly.

The hidden-layer calculation produces:

```text
Z1 = [3.0, 0.0]
A1 = [3.0, 0.0]
```

The output layer then gives:

```text
Z2 = [1.0]
prediction = 0.7311
```

For a true label of `1`, the binary cross-entropy loss is:

```text
BCE = 0.3133
```

The same result is first derived from the equations and then verified with NumPy.

## Loss Functions Covered

| Task | Output Activation | Loss |
|---|---|---|
| Regression | Linear | Mean Squared Error |
| Binary classification | Sigmoid | Binary Cross-Entropy |
| Single-label multi-class classification | Softmax | Categorical Cross-Entropy |

The notebook also notes that sparse categorical cross-entropy is used when multi-class targets are stored as integer class indices instead of one-hot vectors.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install numpy matplotlib jupyter
```

## Running the Notebook

From the directory containing the notebook, run:

```bash
jupyter notebook 02_Activations_Forward_Pass.ipynb
```

Then run the cells from top to bottom.

The notebook does not require an external dataset. The activation plots and the forward-pass example are generated directly from NumPy values.

## Main Takeaway

Day 2 connects the main pieces of a forward neural-network computation:

```text
input
  -> weighted sum + bias
  -> activation
  -> next layer
  -> prediction
  -> loss
```

For the Heart Disease project, the main design decision from this notebook is:

```text
hidden layers -> ReLU
binary output -> Sigmoid
training loss -> Binary Cross-Entropy
```
