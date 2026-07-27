# Week 2 — Day 2: Probability and Distributions

## Overview

This notebook introduces the probability concepts that support later
machine-learning topics. It combines manual calculations with
reproducible NumPy simulations to compare theoretical probabilities with
experimental results.

The work covers the core probability rules, conditional probability,
Bayes' theorem, and three common probability distributions: uniform,
binomial, and normal.

The notebook is written as a documented technical report. Each major
calculation or simulation is introduced with Markdown, implemented in
Python, and followed by an interpretation of the result.

## Learning Objectives

The notebook demonstrates how to:

- Define an experiment, outcome, sample space, and event.
- Calculate probability from favourable and total equally likely outcomes.
- Distinguish theoretical probability from experimental probability.
- Apply the complement, addition, and multiplication rules.
- Distinguish independent events from mutually exclusive events.
- Calculate and interpret conditional probability.
- Explain why \(P(A \mid B)\) generally differs from \(P(B \mid A)\).
- Apply Bayes' theorem using prior, likelihood, evidence, and posterior.
- Recognise uniform, binomial, and normal distributions.
- Use NumPy to simulate random experiments reproducibly.
- Interpret probability results in relation to machine learning.

## Concepts Covered

### Probability Fundamentals

The notebook begins with the basic probability vocabulary:

- Experiment
- Outcome
- Sample space
- Event
- Favourable outcomes
- Theoretical probability
- Experimental probability

A fair-die example is used to calculate the theoretical probability of
rolling an even number.

### Core Probability Rules

The following rules are explained and applied:

- Complement rule
- Addition rule for overlapping events
- Addition rule for mutually exclusive events
- Multiplication rule for independent events

The notebook also compares independent events with mutually exclusive
events to avoid treating the two concepts as equivalent.

### Conditional Probability

A student-workshop scenario is used to calculate:

\[
P(\text{Python} \mid \text{ML workshop})
\]

The probability is calculated manually and then verified with a
100,000-sample NumPy simulation.

The example also compares:

\[
P(A \mid B)
\quad \text{and} \quad
P(B \mid A)
\]

to demonstrate why the order of a conditional probability matters.

### Bayes' Theorem

A simplified spam-detection scenario demonstrates how Bayes' theorem
updates a prior probability after observing evidence.

The example identifies:

- Prior
- Likelihood
- Evidence
- Posterior

It calculates the probability that an email is spam given that it
contains the word `free`.

### Probability Distributions

The notebook covers three common distributions:

- **Uniform distribution:** all possible outcomes have equal probability.
- **Binomial distribution:** counts successes across a fixed number of
  independent binary trials.
- **Normal distribution:** continuous values cluster symmetrically around
  a mean.

## Notebook Workflow

The notebook follows this sequence:

1. Import NumPy and Matplotlib.
2. Create a reproducible random-number generator.
3. Introduce probability terminology.
4. Compare theoretical and experimental probability.
5. Simulate fair-coin flips.
6. Apply the core probability rules.
7. Calculate conditional probability manually.
8. Verify conditional probability through simulation.
9. Apply Bayes' theorem.
10. Simulate uniform, binomial, and normal distributions.
11. Visualise and interpret the generated distributions.
12. Summarise the results and connect them to machine learning.

## Main Simulations and Results

### Fair-Coin Simulation

A fair coin was flipped 10,000 times using NumPy.

| Measure | Result |
|---|---:|
| Total flips | 10,000 |
| Heads | 4,980 |
| Tails | 5,020 |
| Observed heads proportion | 0.4980 |
| Theoretical heads probability | 0.5000 |
| Absolute difference | 0.0020 |

The observed proportion was close to the theoretical probability. The
small difference was expected because the experiment used a finite
random sample.

The notebook also compared 10, 100, 1,000, and 10,000 flips. The larger
samples produced more stable estimates near 0.5.

### Multiplication-Rule Simulation

Fifty thousand pairs of independent coin flips were simulated.

| Measure | Result |
|---|---:|
| Observed probability of two heads | 0.2495 |
| Theoretical probability | 0.2500 |
| Absolute difference | 0.0005 |

The result was consistent with the multiplication rule for independent
events.

### Conditional-Probability Simulation

The student scenario produced:

| Measure | Result |
|---|---:|
| Manual \(P(\text{Python} \mid \text{ML workshop})\) | 0.6667 |
| Simulated conditional probability | 0.6672 |
| Absolute difference | 0.0006 |
| Reverse \(P(\text{ML workshop} \mid \text{Python})\) | 0.5000 |

The result demonstrated that changing the condition changes the relevant
denominator.

### Bayes' Theorem Example

The spam-detection example used:

| Component | Value |
|---|---:|
| Prior \(P(\text{spam})\) | 0.2000 |
| Likelihood \(P(\text{free} \mid \text{spam})\) | 0.7000 |
| Evidence \(P(\text{free})\) | 0.2200 |
| Posterior \(P(\text{spam} \mid \text{free})\) | 0.6364 |

Observing the word `free` increased the estimated probability of spam
from 20% to approximately 63.64%.

### Uniform Distribution

A fair die was rolled 12,000 times.

The observed face proportions ranged from 0.1638 to 0.1692, close to the
theoretical value of approximately 0.1667 for every face.

A labelled bar chart compared the observed proportions with the
theoretical probability.

### Binomial Distribution

Twenty thousand experiments were simulated, with 10 fair-coin flips in
each experiment.

| Measure | Result |
|---|---:|
| Observed mean number of heads | 4.9517 |
| Theoretical mean | 5.0000 |

The distribution was concentrated near five heads, while extreme results
were less frequent.

### Normal Distribution

A sample of 10,000 values was generated from a normal distribution with
a theoretical mean of 100 and a standard deviation of 15.

| Measure | Theoretical | Sample |
|---|---:|---:|
| Mean | 100.0000 | 100.0090 |
| Standard deviation | 15.0000 | 14.9896 |

The histogram formed an approximately symmetric bell shape centred near
the chosen mean.

## Interpretation

The simulations were consistent with their theoretical probability
models.

The results also demonstrated that finite random experiments do not
normally match theoretical probabilities exactly. Larger sample sizes
generally produced more stable estimates, while small samples showed more
variation.

The distribution examples represented different types of random
behaviour and should not be treated as interchangeable.

## Connection to Machine Learning

Probability is relevant to machine learning because models often make
predictions under uncertainty.

Conditional probability can represent the probability of a target class
given observed features. Bayes' theorem explains how class probabilities
can be updated after observing evidence and provides the mathematical
foundation of the Naive Bayes classifier.

Probability distributions are also used to describe repeated binary
outcomes, measurement variation, random sampling, and noise.

No machine-learning model was trained in this notebook. The focus was on
the probability foundations required for later modelling work.

## Tools and Libraries

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

A fixed NumPy random seed was used to keep the simulations reproducible.

## Project Structure

```text
BinX-AI-ML-Internship/
│
├── week-1/
│   └── .venv/
│
└── week-2/
    ├── data/
    │   └── penguins.csv
    │
    ├── d1/
    │   ├── README.md
    │   └── 01_descriptive_statistics.ipynb
    │
    └── d2/
        ├── README.md
        └── 02_probability_and_distributions.ipynb
```

## How to Run the Notebook

Open Windows PowerShell and move to the repository root:

```powershell
cd C:\Users\HP\Desktop\BinX-AI-ML-Internship
```

Activate the existing internship virtual environment:

```powershell
.\week-1\.venv\Scripts\Activate.ps1
```

Move to Week 2:

```powershell
cd week-2
```

Start Jupyter Notebook:

```powershell
jupyter notebook
```

Open:

```text
d2/02_probability_and_distributions.ipynb
```

Then use:

```text
Kernel → Restart Kernel and Run All Cells
```

to run the notebook from top to bottom.

## Notes and Limitations

- Exact simulation results depend on the random seed and sample size.
- The examples assume fair coins, a fair die, and independent trials
  where stated.
- The student and spam examples are simplified educational scenarios.
- A histogram can illustrate a distribution's shape but does not formally
  prove that a real dataset follows that distribution.
- Not every real-world numeric variable follows a normal distribution.

## Files

- `02_probability_and_distributions.ipynb` — documented probability
  calculations, simulations, plots, and interpretations.
- `README.md` — overview, workflow, results, and instructions for running
  the notebook.

## Conclusion

This Day 2 work applied theoretical probability through reproducible
Python simulations.

The notebook demonstrated the core probability rules, conditional
probability, Bayes' theorem, and the uniform, binomial, and normal
distributions. The results showed how experimental values approach
theoretical expectations while retaining normal random variation.

## Next Step

Week 2 Day 3 continues with linear algebra for machine learning,
including vectors, matrices, dot products, and matrix multiplication.