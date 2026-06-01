"""
bm25.py — Modelo probabilístico BM25 (Best Match 25).

BM25 es la función de ranking estándar de la familia probabilística RSJ.
A diferencia de TF-IDF clásico, satura la contribución de la frecuencia
de término (evitando que documentos largos dominen el ranking) y penaliza
documentos más largos que el promedio del corpus:

    score(Q, D) = Σ IDF(t) · [TF(t,D) · (k1+1)] / [TF(t,D) + k1·(1 - b + b·|D|/avgdl)]

Los parámetros estándar k1=1.5 y b=0.75 (Robertson & Zaragoza, 2009)
funcionan bien en la mayoría de colecciones de texto en inglés.

Referencia:
    Robertson, S. & Zaragoza, H. (2009). The Probabilistic Relevance Framework:
    BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4).
"""

from models.base import RetrievalModel
import numpy as np


class BM25Model(RetrievalModel):
    """
    Modelo de recuperación BM25 con saturación de TF y normalización por longitud.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus procesado con columna 'tokenized' (list[str] de stems).
    inv_index : dict[str, dict[int, int]]
        Índice invertido: término → {doc_id → TF}.
    k1 : float
        Parámetro de saturación de TF. Controla cuánto contribuye la
        repetición de un término. Valor típico: 1.2–2.0.
    b : float
        Parámetro de normalización por longitud de documento.
        b=0 desactiva la normalización; b=1 normalización completa.
    """

    def __init__(self, df_corpus, inv_index, k1: float = 1.5, b: float = 0.75):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.N = len(df_corpus)
        self.k1 = k1
        self.b = b
        self.doc_lengths = df_corpus['tokenized'].apply(len).values
        self.avgdl = self.doc_lengths.mean()

    def search(self, query_processed: str, top_n: int = 5) -> tuple:
        """
        Rankea documentos usando la función de scoring BM25.

        Para cada término de la consulta que aparece en el índice, calcula
        el IDF con la variante Robertson (con +1 para evitar valores negativos
        en términos muy frecuentes) y acumula el score BM25 por documento.

        Parameters
        ----------
        query_processed : str
            Términos de la consulta (stems) separados por espacio.
        top_n : int
            Número de documentos a retornar.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - ranking : top_n índices del corpus en orden descendente de score.
            - scores  : array de BM25 scores para todos los documentos.
        """
        query_terms = query_processed.split()
        scores = np.zeros(self.N)

        for term in query_terms:
            if term not in self.inv_index:
                continue

            doc_freqs = self.inv_index[term]
            n_t = len(doc_freqs)

            idf = np.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

            for doc_id, tf in doc_freqs.items():
                dl = self.doc_lengths[doc_id]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (num / den)

        ranking = scores.argsort()[::-1]
        return ranking[:top_n], scores
