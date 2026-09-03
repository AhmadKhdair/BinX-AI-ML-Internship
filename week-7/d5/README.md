# Week 7 - Day 5 Core Model and Sprint 2 Review

This folder closes Sprint 2 for the Heart Disease Classification capstone.

## Files

 File  Description 
------
 `05_Core_Model_Sprint2_Review.ipynb`  Main Day 5 notebook 
 `outputsday5_experiment_tracking.csv`  Validation experiment log 
 `outputsday5_best_config.csv`  Selected validation-only configuration 
 `outputsday5_selected_threshold_scan.csv`  Threshold scan for the selected model 
 `outputsday5_test_metrics.csv`  Final locked test metrics 
 `outputsday5_model_comparison.csv`  Sprint 2 comparison against Week 6 models 
 `outputsday5_final_audit.csv`  Final readiness audit 
 `outputsday5_final_evaluation_curves.png`  Confusion matrix, ROC curve, and PR curve 

## Scope

Day 5 is not a new CNN, LSTM, or Transformer exercise. It is the Sprint 2 capstone close-out choose the architecture that fits the project data, tune the core model, log experiments, compare against previous baselines, and write the Sprint Review and Retrospective.

The capstone dataset is tabular

```text
final-projectheart-disease-classificationdataheart.csv
```

It contains `918` rows, `11` input features, and the binary target `HeartDisease`.

Because the data is tabular, the notebook does not force CNN, LSTM, or Transformer models into the project. The selected Sprint 2 direction is a tabular classifier, with gradient boosting and tree ensembles tested as the main candidates.

## Protocol

The notebook keeps the Week 6 protocol as closely as possible

- same `heart.csv` dataset;
- same cleaning rules for invalid `RestingBP = 0` and unknown `Cholesterol = 0`;
- same `random_state=42`;
- same stratified `8020` held-out test split;
- validation split created only from the training portion;
- preprocessing fit only on training data;
- model and threshold selection based on validation only;
- final test evaluation done once after selection.

## Historical References

The comparison uses saved Week 6 outputs from the repository

 Model  F1  ROC-AUC 
---------
 Week 6 tuned Random Forest  0.8846  0.9173 
 Sprint 1 Day 4 NN  0.8750  0.9014 
 Week 6 tuned NN  0.8768  0.9304 

The Week 6 tuned Random Forest remains the strongest F1 baseline.

## Day 5 Result

The selected Sprint 2 model was chosen from validation results only

```text
Selected model hgb_lr_0p05_l2_0p0_leaf_15
Model family HistGradientBoosting
Threshold 0.5
```

Final locked test result

 Metric  Value 
------
 Accuracy  0.8641 
 Precision  0.8969 
 Recall  0.8529 
 F1  0.8744 
 ROC-AUC  0.9091 
 PR-AUC  0.9201 

## Final Comparison

 Model  F1  ROC-AUC  Result 
------------
 Week 6 tuned Random Forest  0.8846  0.9173  Best F1 
 Week 6 tuned NN  0.8768  0.9304  Best ROC-AUC 
 Week 7 Day 5 Sprint 2 model  0.8744  0.9091  Did not beat RF 

The Sprint 2 model did not beat the Week 6 tuned Random Forest on the locked F1 reference. The correct conclusion is that the architecture choice and validation protocol are right, but this specific Sprint 2 model is not strong enough to replace the Week 6 Random Forest.

## Sprint Review Summary

Completed

- confirmed that the capstone data is tabular;
- rejected CNNLSTMTransformer for the capstone core model because they do not match the data structure;
- reused the Week 6 cleaning, feature engineering, split, and evaluation protocol;
- tuned tabular boostingtree models;
- logged all validation experiments;
- selected the final model using validation only;
- evaluated once on the locked test split;
- compared against Week 6 tuned RF, Sprint 1 NN, and Week 6 tuned NN.

Moved to Sprint 3

- model replacement, because the Sprint 2 model did not beat the Week 6 RF F1;
- repeated stratified CV for more stable model selection;
- probability calibration and operating-threshold analysis;
- stronger experiment tracking, such as MLflow, if the experiment set grows.

## Final Audit

Implementation status complete.

Model-improvement status not ready as a baseline-beating claim.

```text
Final verdict Not ready for GitHub as a model-improvement claim
Blocker Sprint 2 model F1=0.8744 did not beat Week 6 tuned RF F1=0.8846
```

This is still a valid Day 5 result because it follows the required protocol and reports the outcome honestly instead of forcing a false improvement story.

## How to Run

From the repository root or from `week-7d5`, run the notebook top to bottom.

Required packages

```bash
pip install numpy pandas matplotlib scikit-learn notebook
```

If running in Google Colab, use this setup cell first

```python
from pathlib import Path
import os
import subprocess

repo = Path(contentBinX-AI-ML-Internship)

if Path(content).exists()
    if not repo.exists()
        subprocess.run(
            [git, clone, --depth, 1, httpsgithub.comAhmadKhdairBinX-AI-ML-Internship.git],
            check=True,
        )
    else
        print(Repository already exists in this Colab runtime.)

    os.chdir(repo  week-7  d5)
else
    print(Running locally, keeping current folder.)

print(Current folder, Path.cwd())
```