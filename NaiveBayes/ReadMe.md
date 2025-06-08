# Twitter Sentiment Classification with Naive Bayes

A simple implementation of a Naive Bayes classifier to perform sentiment analysis on tweets. This project demonstrates data preprocessing, feature extraction, model training, evaluation, and prediction on Twitter data labeled as **positive** or **negative**.

---

## Table of Contents

- [Overview](#overview)  
- [Dataset](#dataset)  
- [Preprocessing](#preprocessing)  
- [Feature Extraction](#feature-extraction)  
- [Model Training](#model-training)  
- [Evaluation](#evaluation)  
- [Usage](#usage)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Overview

Naive Bayes is a probabilistic classifier based on Bayes’ theorem, assuming that features are independent given the class label. In this project, we:

1. Collect and load a labeled Twitter dataset.  
2. Preprocess tweets (lowercase, remove punctuation/stopwords, stemming, tokenization).  
3. Extract bag-of-words features.  
4. Train a Multinomial Naive Bayes classifier.  
5. Evaluate model performance on a held-out test set.  
6. Provide a command-line interface for sentiment prediction on new tweets.  

---

## Dataset

We use a CSV of pre-labeled tweets with the following columns:

| Column     | Description                          |
|------------|--------------------------------------|
| `tweet`    | Raw tweet text                       |
| `sentiment`| Label: `0` = Negative, `1` = Positive |

> **Note:** You can substitute your own dataset as long as it follows this format.

---

## Preprocessing

Tweets must be cleaned and tokenized before feature extraction:

1. **Lowercase** all text.  
2. **Remove** URLs, mentions (`@username`), hashtags, punctuation, and numbers.  
3. **Tokenize** into words/sentences.  
4. **Remove stopwords** (e.g. “the”, “is”, “and”).  
5. **Stemming** using Porter Stemmer.

```python
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|#\S+|\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    stops = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in tokens if w not in stops]