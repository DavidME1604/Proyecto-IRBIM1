"""
preprocessing.py — Carga y preprocesamiento del corpus Reuters.

Descarga el dataset desde Kaggle, aplica una pipeline estándar de
normalización léxica (minúsculas, tokenización, eliminación de stopwords,
stemming con Snowball) y construye el índice invertido usado por los modelos
de recuperación léxica (Jaccard, TF-IDF, BM25).

El corpus procesado y el índice se cachean en disco con joblib para evitar
recomputación en ejecuciones sucesivas.
"""

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
    """
    Descarga el dataset Reuters desde Kaggle y retorna la ruta al CSV.

    Returns
    -------
    str
        Ruta absoluta al archivo ModApte_train.csv dentro del directorio
        descargado por kagglehub.
    """
    path = kagglehub.dataset_download("thedevastator/uncovering-financial-insights-with-the-reuters-2")
    return os.path.join(path, 'ModApte_train.csv')


def process_raw(raw_text: str) -> list[str]:
    """
    Aplica la pipeline de normalización léxica a un texto crudo.

    La pipeline es: eliminar caracteres no alfabéticos → minúsculas →
    tokenizar → filtrar stopwords → stemming con Snowball (inglés).

    Parameters
    ----------
    raw_text : str
        Texto de entrada sin procesar.

    Returns
    -------
    list[str]
        Lista de tokens normalizados (stems).
    """
    clean_text = re.sub(r'[^a-z\s]', '', raw_text.lower())
    tokens = word_tokenize(clean_text)
    return [stemmer_english.stem(t) for t in tokens if t not in stop_words]


def process_corpus(path: str) -> pd.DataFrame:
    """
    Carga el CSV y aplica process_raw a cada documento del corpus.

    Cada fila del CSV genera un registro con el texto crudo (título + body),
    los tokens normalizados y su versión como string para scikit-learn/TF-IDF.

    Parameters
    ----------
    path : str
        Ruta al archivo ModApte_train.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas:
        - doc_id     : índice original del CSV (int)
        - title      : título original del documento
        - raw        : concatenación "título body" sin procesar
        - tokenized  : list[str] de stems
        - processed  : string de stems separados por espacio
    """
    corpus = {'doc_id': [], 'title': [], 'raw': [], 'tokenized': [], 'processed': []}
    df_input = pd.read_csv(path)
    for id_doc, row in df_input.iterrows():
        text = str(row['title']) + ' ' + str(row['text'])
        stemmed_words = process_raw(text)
        corpus['doc_id'].append(id_doc)
        corpus['title'].append(row['title'])
        corpus['raw'].append(text)
        corpus['tokenized'].append(stemmed_words)
        corpus['processed'].append(' '.join(stemmed_words))
    return pd.DataFrame(corpus)


def inverse_index(df_corpus: pd.DataFrame) -> dict:
    """
    Construye el índice invertido con frecuencias de término por documento.

    El índice tiene la forma {term: {doc_id: tf}}, donde tf es la frecuencia
    bruta del término en el documento. Este índice es compartido por todos
    los modelos léxicos para evitar recalcular postings en tiempo de consulta.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus procesado, debe contener las columnas 'doc_id' y 'tokenized'.

    Returns
    -------
    dict[str, dict[int, int]]
        Índice invertido: término → {doc_id → frecuencia}.
    """
    index = defaultdict(dict)
    for _, row in df_corpus.iterrows():
        doc_id = row['doc_id']
        conteo = Counter(row['tokenized'])
        for term, f in conteo.items():
            index[term][doc_id] = f
    return index


def load_or_build(cache_dir: str = 'cache') -> tuple:
    """
    Retorna el corpus procesado y el índice invertido, usando caché si existe.

    Si los archivos de caché no existen, descarga el dataset, procesa el
    corpus y construye el índice, luego los serializa para uso futuro.

    Parameters
    ----------
    cache_dir : str
        Directorio donde se guardan/leen los archivos .pkl.

    Returns
    -------
    tuple[pd.DataFrame, dict, str]
        - df_corpus : corpus procesado
        - inv_index : índice invertido
        - path      : ruta al CSV original (necesaria para cargar los qrels)
    """
    os.makedirs(cache_dir, exist_ok=True)
    corpus_cache = os.path.join(cache_dir, 'corpus.pkl')
    index_cache  = os.path.join(cache_dir, 'inv_index.pkl')
    path_cache   = os.path.join(cache_dir, 'path.pkl')

    if all(os.path.exists(p) for p in (corpus_cache, index_cache, path_cache)):
        df_corpus = joblib.load(corpus_cache)
        inv_index = joblib.load(index_cache)
        path      = joblib.load(path_cache)
    else:
        path      = download_dataset()
        df_corpus = process_corpus(path)
        inv_index = inverse_index(df_corpus)
        joblib.dump(df_corpus, corpus_cache)
        joblib.dump(inv_index, index_cache)
        joblib.dump(path,      path_cache)

    return df_corpus, inv_index, path
