# Week 6 - Day 3: Backpropagation, Gradient Descent & Optimizers

This folder contains the Day 3 training-mechanics notebook for Week 6 of the BinX AI/ML Internship.

The notebook continues from Day 2, where the forward pass ended with a prediction and a loss value. Day 3 focuses on the other half of training: how gradients are computed, how the parameters are updated, and how the learning rate changes the behavior of the training process.

## Notebook

`03_Training_Mechanics.ipynb`

## What I Covered

The notebook includes:

- the four-step training loop: forward pass, loss, backpropagation, and update;
- gradient descent and the parameter-update equation;
- the role of the learning rate;
- backpropagation and why the chain rule is involved;
- the difference between backpropagation and the optimizer;
- SGD and Adam at a conceptual level;
- epochs and batches;
- a tiny neural-network experiment using three learning rates;
- loss curves showing slow, stable, and unstable training behavior.

## Learning-Rate Experiment

The experiment uses a small synthetic regression dataset and a fixed NumPy network:

`1 input -> 8 tanh hidden units -> 1 output`

Only the learning rate changes between the runs:

- `0.0001` - too low;
- `0.3` - suitable for this small setup;
- `0.7` - too high and unstable.

The exact values are specific to the toy NumPy network. They are not intended as defaults for the Heart Disease project or for Adam.

## Why No External Dataset Is Needed

Day 3 asks for a tiny-network learning-rate experiment rather than a new dataset analysis. The synthetic data keeps the experiment focused on training mechanics and makes the comparison reproducible.

The Heart Disease Classification project remains the Sprint 1 project. Its neural-network implementation and Keras training are handled in the next training stage rather than being mixed into this notebook.

## Requirements

- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or JupyterLab

Install the required packages with:

```bash
pip install numpy matplotlib jupyter
```

No PyTorch or TensorFlow installation is required for this Day 3 notebook.

## Running the Notebook

From the directory containing the notebook, run:

```bash
jupyter notebook 03_Training_Mechanics.ipynb
```

Then run the cells from top to bottom. The notebook does not require any external files.

## Mid-Sprint Review

Day 3 also includes the mentor code and notebook review. After the notebook is ready, the current Sprint 1 work should be committed on the feature branch and opened as a pull request for mentor review. Any review feedback should be addressed before the sprint continues.
