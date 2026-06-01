from collections import defaultdict, Counter
import kagglehub
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import joblib
import os

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

stop_words = set(stopwords.words('english'))
stemmer_english = SnowballStemmer('english')

def download_dataset() -> str:
    path = kagglehub.dataset_download("thedevastator/uncovering-financial-insights-with-the-reuters-2")
    return os.path.join(path, 'ModApte_train.csv')

def process_raw(raw_text: str) -> list:
    clean_text = re.sub(r'[^a-z\s]', '', raw_text)
    tokens = word_tokenize(clean_text.lower())
    stemmed_words = [stemmer_english.stem(token) for token in tokens if token not in stop_words] 
    return stemmed_words

def process_corpus(path: str)-> pd.DataFrame:
    corpus = {'doc_id': [], 'title': [], 'raw': [], 'tokenized': [],'processed': []}
    df_input = pd.read_csv(path)
    for id_doc, row in df_input.iterrows():
        text = str(row['title'])+' '+str(row['text'])
        stemmed_words = process_raw(text)
        corpus['doc_id'].append(id_doc)
        corpus['title'].append(row['title'])
        corpus['raw'].append(text)
        corpus['tokenized'].append(stemmed_words)
        corpus['processed'].append(' '.join(stemmed_words))
    return pd.DataFrame(corpus)

def inverse_index(df_corpus):
    index = defaultdict(dict)
    for _, row in df_corpus.iterrows():
        doc_id = row['doc_id']
        conteo = Counter(row['tokenized'])
        for term, f in conteo.items():
            index[term][doc_id] = f
    return index

def load_or_build(cache_dir='cache'):
    os.makedirs(cache_dir, exist_ok=True)
    corpus_cache = os.path.join(cache_dir, 'corpus.pkl')
    index_cache = os.path.join(cache_dir, 'inv_index.pkl')
    path_cache = os.path.join(cache_dir, 'path.pkl')

    if os.path.exists(corpus_cache) and os.path.exists(index_cache) and os.path.exists(path_cache):
        df_corpus = joblib.load(corpus_cache)
        inv_index = joblib.load(index_cache)
        path = joblib.load(path_cache)
    else:
        path = download_dataset()
        df_corpus = process_corpus(path)
        inv_index = inverse_index(df_corpus)
        joblib.dump(df_corpus, corpus_cache)
        joblib.dump(inv_index, index_cache)
        joblib.dump(path, path_cache)
        print("Caché guardado.")

    return df_corpus, inv_index, path