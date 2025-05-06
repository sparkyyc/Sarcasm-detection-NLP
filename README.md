# Sarcasm Detection in Tweets

This repo contains scripts for detecting sarcasm in tweets using both a feature-based ML pipeline and a transformer (BERT)-based approach, plus hyperparameter tuning and error analysis.

Video: https://youtu.be/8O4wm8mfwqQ

## What's included

Report PDF

Slides PDF

SemEval2018-T3-train-taskA.txt            # Raw tweet dataset

sarcasm-detection.py                      # Baseline pipeline: TF–IDF + hand-crafted features & initial BERT fine-tuning

sarc-det-hyperparams.py                   # Full hyperparameter tuning (RandomizedSearchCV + Optuna HPO)

sarc-det-hyperparams-quick.py             # Fast hyperparameter tuning (reduced budget + Optuna pruner)

error-analysis.py                         # Generate ML vs. BERT error analysis tables

requirements.txt                          # Python dependencies

README.md                                 # This file


## Setup & Dependencies

1. Create & activate a virtual environment:
   
    python3 -m venv venv
    source venv/bin/activate

2. Install Python packages:

    pip install -r requirements.txt

3. Download spaCy language model:

    python -m spacy download en_core_web_sm

4. Download NLTK VADER lexicon:

    python -c "import nltk; nltk.download('vader_lexicon')"

## How to Run

### Baseline & initial BERT fine-tuning

    python sarcasm-detection.py

- Trains a logistic regression pipeline (TF-IDF + engineered features).

- Fine-tunes BERT for 3 epochs.

### Full hyperparameter tuning

    python sarc-det-hyperparams.py

- Runs RandomizedSearchCV (5 candidates) for the ML pipeline.

- Runs Optuna HPO (3 trials) for BERT.

- Saves models & writes results_<timestamp>.json.

### Quick hyperparameter tuning

    python sarc-det-hyperparams-quick.py

- Same as above but with reduced budgets and an Optuna pruner for faster iteration.

### Error analysis

    python error-analysis.py

- Loads best_ml_pipeline.joblib and fine-tuned BERT from ./results/.

- Prints false positives/negatives and overlap summaries.

## Outputs

### Models

- best_ml_pipeline.joblib

- Fine-tuned BERT model & tokenizer in ./results/

### Metrics & Reports

- results_<timestamp>.json (includes best hyperparameters, classification reports, BERT eval)

- Error analysis tables printed to console