"""
jaccard.py — Modelo de recuperación basado en similitud de Jaccard.

Jaccard mide la superposición entre conjuntos de términos:

    J(Q, D) = |Q ∩ D| / |Q ∪ D|

donde Q es el conjunto de términos de la consulta y D el del documento.
Al trabajar con conjuntos se ignoran las frecuencias de término, lo que
hace al modelo simple pero insensible al peso de los términos.

La intersección se computa eficientemente usando el índice invertido:
por cada término de la consulta se incrementan los contadores de los
documentos que lo contienen, evitando iterar sobre todo el corpus.
"""

from models.base import RetrievalModel
import numpy as np


class JaccardModel(RetrievalModel):
    """
    Recuperación por similitud de conjuntos de Jaccard sobre el índice invertido.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus procesado con columna 'tokenized' (list[str] de stems).
    inv_index : dict[str, dict[int, int]]
        Índice invertido: término → {doc_id → frecuencia}.
    """

    def __init__(self, df_corpus, inv_index):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.doc_unique_terms = df_corpus['tokenized'].apply(lambda t: len(set(t))).values

    def search(self, query_processed: str, top_n: int = 5) -> tuple:
        """
        Rankea documentos por similitud de Jaccard con la consulta.

        La unión se calcula como |D| + |Q| - |D ∩ Q| (usando cardinalidades
        de conjuntos), evitando materializar la unión real de cada par (Q, D).

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
            - scores  : array de Jaccard scores para todos los documentos.
        """
        query_terms = set(query_processed.split())
        N = len(self.df_corpus)
        intersection = np.zeros(N)

        for term in query_terms:
            if term in self.inv_index:
                for doc_id in self.inv_index[term]:
                    intersection[doc_id] += 1

        union = self.doc_unique_terms + len(query_terms) - intersection
        scores = np.divide(intersection, union,
                           out=np.zeros(N, dtype=float), where=union != 0)

        ranking = scores.argsort()[::-1]
        return ranking[:top_n], scores
