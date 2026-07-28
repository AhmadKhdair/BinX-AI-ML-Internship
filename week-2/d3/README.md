# Week 2 — Day 3: Linear Algebra for Machine Learning

## Overview

This folder contains the Day 3 work for Week 2 of the BinX Tech AI & Machine Learning Internship.

The notebook introduces the main linear algebra concepts used in machine learning and demonstrates how NumPy can represent data samples, datasets, model weights, and simplified prediction calculations.

## Learning Objectives

By the end of this notebook, the following concepts are covered:

- Representing one data sample as a vector
- Representing multiple samples as a matrix
- Understanding rows as samples and columns as features
- Computing the dot product manually
- Verifying the dot product using `numpy.dot()`
- Using matrix multiplication to process multiple samples
- Connecting features, weights, and bias to model predictions
- Understanding shape compatibility
- Identifying and fixing a shape-mismatch error

## Notebook Structure

The notebook is organised into the following sections:

1. Why linear algebra matters in machine learning
2. Importing NumPy
3. Vectors
4. Matrices
5. The dot product
6. Producing predictions for multiple samples
7. Verifying one prediction
8. Understanding shape compatibility
9. Demonstrating a shape mismatch
10. Fixing the shape mismatch
11. Summary

## Main Concepts

### Vectors

A vector is an ordered collection of numbers.

In machine learning, a vector usually represents one data sample, where each value represents one feature.

```python
student = np.array([6, 3, 4])