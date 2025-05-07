# Christa Sparks
# hyper parameter tuning
# runtime ~ 3-5 hours
import re
import json
import numpy as np
import pandas as pd
import torch
import spacy
import contractions
import joblib
import optuna
from optuna.pruners import MedianPruner

from datetime import datetime
from scipy.sparse import csr_matrix
from scipy.stats import loguniform
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase
    text = text.lower()
    # Simple negation tagging
    negation_words = {'not','no','never',"n't"}
    tokens = text.split()
    negated = False
    new_tokens = []
    for tok in tokens:
        if any(neg in tok for neg in negation_words):
            negated = True
            new_tokens.append(tok)
        elif negated:
            if re.search(r'[\.!?]', tok):
                negated = False
                new_tokens.append(tok)
            else:
                new_tokens.append(tok + '_NEG')
        else:
            new_tokens.append(tok)
    text = ' '.join(new_tokens)
    # Remove URLs, user mentions, hashtags markers
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TextFeaturesExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        # emoticons list
        self.emoticons = [":)", ";)", ":D", ":-)", ":-(", ":P", ";-)", ":-P", ";P"]
    
    def count_punctuation(self, text):
        return [text.count('!'), text.count('?')]

    def sentiment_scores(self, text):
        scores = self.sid.polarity_scores(text)
        return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]

    def pos_ratio(self, text):
        doc = nlp(text)
        adj = sum(1 for token in doc if token.pos_ == 'ADJ')
        total = len(doc)
        return [adj / total if total > 0 else 0]

    def emoticon_count(self, text):
        return [sum(text.count(e) for e in self.emoticons)]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        feat_list = []
        for text in X:
            punct = self.count_punctuation(text)
            sent = self.sentiment_scores(text)
            posr = self.pos_ratio(text)
            emo = self.emoticon_count(text)
            feat_list.append(punct + sent + posr + emo)
        return np.array(feat_list)

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return csr_matrix(X)


data_path = 'SemEval2018-T3-train-taskA.txt'
df = pd.read_csv(data_path, sep='\t', skiprows=1,
                 names=['index', 'label', 'tweet'], dtype={'tweet': str})
df['tweet'] = df['tweet'].apply(preprocess_text)
X, y = df['tweet'], df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

combined = FeatureUnion([
    ('word_tfidf', TfidfVectorizer()),
    ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))),
    ('custom_feats', Pipeline([('ext', TextFeaturesExtractor()), ('dens', DenseTransformer())]))
])
pipeline = Pipeline([('features', combined), ('clf', LogisticRegression(max_iter=1000))])
param_dist = {
    'features__word_tfidf__ngram_range': [(1,1), (1,2)],
    'clf__C': loguniform(1e-2, 1e2),
    'clf__class_weight': [None, 'balanced']
}
rand_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist,
    n_iter=5, # 5 candidates
    cv=3, # 3-fold CV
    scoring='f1',
    random_state=42,
    verbose=2
)
rand_search.fit(X_train, y_train)
print("RandomizedSearchCV Best Params:", rand_search.best_params_)
best_ml_pipeline = rand_search.best_estimator_
print("\nTraditional ML Results")
print(classification_report(y_test, best_ml_pipeline.predict(X_test)))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok, max_length=256):
        self.texts, self.labels = texts.reset_index(drop=True), labels.reset_index(drop=True)
        self.tok, self.max_length = tok, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_length,
                       padding='max_length', truncation=True, return_tensors='pt')
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

train_ds = TweetDataset(X_train, y_train, tokenizer)
test_ds  = TweetDataset(X_test, y_test, tokenizer)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': (preds == labels).mean(),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

def model_init():
    return BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=2
    )

def hp_space(trial):
    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
        'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 5),
        'per_device_train_batch_size': 8,
        'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.3)
    }

base_args = TrainingArguments(
    output_dir='./hp_search',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    logging_steps=10,
    seed=42
)

hp_trainer = Trainer(
    model_init=model_init,
    args=base_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

best_run = hp_trainer.hyperparameter_search(
    direction='maximize',
    hp_space=hp_space,
    backend='optuna',
    n_trials=3,
    pruner=MedianPruner(n_warmup_steps=1), # early stopping
)
print("Optuna best run:", best_run)

training_args = TrainingArguments(
    output_dir='./results',
    learning_rate= best_run.hyperparameters['learning_rate'],
    num_train_epochs= best_run.hyperparameters['num_train_epochs'],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay= best_run.hyperparameters['weight_decay'],
    save_steps=200,
    eval_steps=200,
    do_eval=True,
    logging_steps=10,
    metric_for_best_model='f1',
    seed=42
)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Fine-tuning BERT on best params")
trainer.train()
print("Final Evaluation")
bert_eval = trainer.evaluate()

joblib.dump(best_ml_pipeline, 'best_ml_pipeline.joblib')
trainer.save_model('./results')
tokenizer.save_pretrained('./results')

now = datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    'timestamp': now,
    'sklearn': {
        'best_params': rand_search.best_params_,
        'report': classification_report(
            y_test, best_ml_pipeline.predict(X_test), output_dict=True
        )
    },
    'bert_hpo': best_run.hyperparameters,
    'bert_final_eval': bert_eval
}

outfile = f"results_{now}.json"
with open(outfile, 'w') as fp:
    json.dump(results, fp, indent=2)
