# Week 7 - Day 4: Attention, Transformers and Pre-trained NLP

This folder contains the Day 4 notebook for Week 7. The work focuses on the limitation of recurrent sequence models, the attention mechanism, the core Transformer idea, and practical inference with a pre-trained Hugging Face Transformer.

The notebook uses a real text-classification dataset instead of manually written examples. I evaluate a pre-trained DistilBERT sentiment classifier on a fixed subset of the official IMDb test split, then inspect tokenization, truncation, metrics, the confusion matrix, and high-confidence errors.

## Files

| File | Description |
|---|---|
| `04_Attention_Transformers_HuggingFace.ipynb` | Day 4 notebook covering Attention, Transformers, Hugging Face inference, IMDb evaluation, and error analysis |

## Dataset

The notebook uses the IMDb Large Movie Review Dataset from Hugging Face:

```text
Dataset: stanfordnlp/imdb
Task: binary sentiment classification
Labels: neg, pos
Train split: 25,000 reviews
Test split: 25,000 reviews
Evaluation subset: 2,000 reviews from the official test split
```

The evaluation subset is stratified:

| Label | Samples |
|---|---:|
| Negative | 1,000 |
| Positive | 1,000 |

I chose IMDb because it is a real, documented NLP dataset with clear labels, enough samples for meaningful evaluation, and a practical use case such as review monitoring or customer-feedback triage.

Dataset source: [Stanford NLP IMDb on Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb)

## Notebook Scope

The notebook covers:

- RNN/LSTM step-by-step processing limitations
- why attention helps with long-range dependencies
- self-attention over tokens in the same sequence
- why Transformers can process positions more parallelly than recurrent models
- the basic Transformer flow: token embeddings, positional information, self-attention blocks, feed-forward blocks, and a classification head
- the role of positional encoding or positional information
- pre-trained Transformer families: BERT, DistilBERT, and GPT-2
- Hugging Face `pipeline` inference on real IMDb reviews
- direct batched PyTorch inference with `model.eval()` and `torch.inference_mode()`
- tokenizer behavior, padding, truncation, and the 512-token input limit
- classification metrics: accuracy, precision, recall, F1, and ROC-AUC
- confusion matrix and high-confidence error inspection
- a final architecture decision explaining when a Transformer is appropriate and when it is not

## Evaluation Protocol

The official IMDb train/test split is preserved. The notebook does not train or tune the model on IMDb. The training split is used only for dataset inspection, and the official test split is used for evaluation.

The model is:

```text
distilbert/distilbert-base-uncased-finetuned-sst-2-english
```

This is a pre-trained DistilBERT checkpoint with a binary sentiment-classification head. The notebook evaluates it on IMDb as cross-dataset inference, so the reported metrics come from the executed notebook outputs, not from the model card.

Model source: [DistilBERT SST-2 model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)

## Saved Run Results

The saved notebook was executed on Google Colab with a Tesla T4 GPU.

```text
Python: 3.13.15
PyTorch: 2.11.0+cu128
Transformers: 5.16.1
Datasets: 5.0.1
CUDA available: True
GPU: Tesla T4
```

Inference speed on the 2,000-review evaluation subset:

```text
Inference seconds: 34.78
Reviews per second: 57.5
```

Final metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.8925 |
| Precision | 0.9198 |
| Recall | 0.8600 |
| F1-score | 0.8889 |
| ROC-AUC | 0.9584 |

Classification report:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.8685 | 0.9250 | 0.8959 | 1,000 |
| Positive | 0.9198 | 0.8600 | 0.8889 | 1,000 |

Confusion matrix:

| True Label | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Negative | 925 | 75 |
| Positive | 140 | 860 |

The model performs strongly overall, but it misses more positive reviews than negative reviews at the default decision rule. That is visible from the `140` false negatives compared with `75` false positives.

## Tokenization and Truncation

IMDb reviews can be longer than the model input limit, so the notebook checks token lengths before evaluation.

| Token Length Statistic | Value |
|---|---:|
| Minimum tokens | 14 |
| Median tokens | 221 |
| 95th percentile tokens | 760 |
| Maximum tokens | 1,392 |
| Reviews over 512 tokens | 276 |
| Truncation rate | 13.8% |

The truncation check is treated as descriptive analysis, not as proof of causality. In the saved run, truncated reviews had a higher error rate than non-truncated reviews, but long reviews can be harder for several reasons besides truncation.

| Group | Error Rate | Samples |
|---|---:|---:|
| All reviews | 0.1075 | 2,000 |
| Truncated reviews | 0.1341 | 276 |
| Non-truncated reviews | 0.1032 | 1,724 |

## How to Run

In Google Colab:

1. Open `04_Attention_Transformers_HuggingFace.ipynb`.
2. Use a GPU runtime if available.
3. Run the notebook from top to bottom.
4. The IMDb dataset and DistilBERT checkpoint are downloaded automatically from Hugging Face.
5. No manual dataset upload is required.

Required packages:

```bash
pip install transformers==5.16.1 datasets==5.0.1 numpy pandas matplotlib scikit-learn torch notebook
```

The notebook finishes successfully when the final cell prints:

```text
Day 4 execution audit passed.
```

## Day 4 Result

The Day 4 requirements are covered. The notebook explains the limitation of RNN/LSTM recurrence, attention and self-attention, long-range context, parallelism, Transformer architecture, positional information, pre-trained Transformers, and practical Hugging Face usage.

For this sentiment-analysis lab, a pre-trained Transformer is the correct choice because the input is text, context matters, and a task-specific sentiment checkpoint is already available. I do not compare this IMDb Transformer metric against the Day 3 ECG LSTM metric because they are different datasets and tasks. A fair LSTM-vs-Transformer comparison would require both models to run on the same text split with the same metric.

For a small tabular capstone, a Transformer is not automatically the core model. The architecture should still match the data type, latency budget, memory cost, and deployment constraints.
