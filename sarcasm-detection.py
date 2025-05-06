import re
import numpy as np
import pandas as pd
import nltk
import torch
import spacy
import contractions

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from scipy.sparse import csr_matrix

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback

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
cols = ['index','label','tweet']
df = pd.read_csv(data_path, sep='\t', skiprows=1, names=cols, dtype={'tweet': str})
df['tweet'] = df['tweet'].apply(preprocess_text)

X = df['tweet']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature unions: word ngrams, char ngrams, custom features
combined = FeatureUnion([
    ('word_tfidf', TfidfVectorizer(ngram_range=(1,2))),
    ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5))),
    ('custom_feats', Pipeline([
        ('ext', TextFeaturesExtractor()),
        ('dens', DenseTransformer())
    ]))
])

pipeline = Pipeline([
    ('features', combined),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid = {
    'features__word_tfidf__ngram_range': [(1,1),(1,2)],
    'clf__C': [0.1,1,10],
    'clf__class_weight': ['balanced', None]
}

grid = GridSearchCV(
    pipeline, param_grid,
    cv=5,
    scoring='f1',
    n_jobs=1,           
    verbose=1,
    error_score='raise' 
)

grid.fit(X_train, y_train)
print("Best Params:", grid.best_params_)
print(classification_report(y_test, grid.predict(X_test)))

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

train_ds = TweetDataset(X_train, y_train, tokenizer, max_length=256)
test_ds = TweetDataset(X_test, y_test, tokenizer, max_length=256)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    do_train=True,
    do_eval=True,
    eval_steps=200,
    save_steps=200,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    metric_for_best_model='f1',
    seed=42
)





def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': (preds==labels).mean(),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics
)


print("Fine-tuning BERT ...")
trainer.train()
print("Evaluation ...")
print(trainer.evaluate())

import joblib

joblib.dump(grid, 'ml_grid_search.joblib')

trainer.save_model('./results')               
tokenizer.save_pretrained('./results')
