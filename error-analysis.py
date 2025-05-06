import re
import numpy as np
import torch
import contractions
import pandas as pd
import joblib
from transformers import Trainer, BertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix

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


grid = joblib.load('ml_grid_search.joblib')

model     = BertForSequenceClassification.from_pretrained('./results')
tokenizer = BertTokenizer.from_pretrained('./results')
trainer   = Trainer(model=model)

data_path = 'SemEval2018-T3-train-taskA.txt'
cols = ['index','label','tweet']
df = pd.read_csv(data_path, sep='\t', skiprows=1, names=cols, dtype={'tweet':str})
df['tweet'] = df['tweet'].apply(preprocess_text)

X = df['tweet']
y = df['label']
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    
test_ds = TweetDataset(X_test, y_test, tokenizer, max_length=256)

ml_preds   = grid.predict(X_test)
bert_out   = trainer.predict(test_ds)
bert_preds = np.argmax(bert_out.predictions, axis=1)

df_err = pd.DataFrame({
    'tweet':      X_test.reset_index(drop=True),
    'true':       y_test.reset_index(drop=True),
    'ml_pred':    ml_preds,
    'bert_pred':  bert_preds
})

bert_fp = df_err[(df_err.bert_pred==1) & (df_err.true==0)]
bert_fn = df_err[(df_err.bert_pred==0) & (df_err.true==1)]
ml_fp   = df_err[(df_err.ml_pred  ==1) & (df_err.true==0)]
ml_fn   = df_err[(df_err.ml_pred  ==0) & (df_err.true==1)]

print("\n BERT False Positives")
print(bert_fp[['tweet','true','bert_pred']].head(10))
print("\n BERT False Negatives")
print(bert_fn[['tweet','true','bert_pred']].head(10))

print("\n ML False Positives")
print(ml_fp[['tweet','true','ml_pred']].head(10))
print("\n ML False Negatives")
print(ml_fn[['tweet','true','ml_pred']].head(10))

ml_fp      = df_err[(df_err.ml_pred   == 1) & (df_err.true == 0)]
bert_fp    = df_err[(df_err.bert_pred == 1) & (df_err.true == 0)]
overlap_fp = df_err[(df_err.ml_pred   == 1) &
                    (df_err.bert_pred == 1) &
                    (df_err.true       == 0)]

ml_fn      = df_err[(df_err.ml_pred   == 0) & (df_err.true == 1)]
bert_fn    = df_err[(df_err.bert_pred == 0) & (df_err.true == 1)]
overlap_fn = df_err[(df_err.ml_pred   == 0) &
                    (df_err.bert_pred == 0) &
                    (df_err.true       == 1)]

summary = pd.DataFrame({
    'Error Type':      ['False Positives', 'False Negatives'],
    'ML Errors':       [ml_fp.shape[0],      ml_fn.shape[0]],
    'BERT Errors':     [bert_fp.shape[0],    bert_fn.shape[0]],
    'Overlap Errors':  [overlap_fp.shape[0], overlap_fn.shape[0]]
})

print(summary.to_markdown(index=False))
