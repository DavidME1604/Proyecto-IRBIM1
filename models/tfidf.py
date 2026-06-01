"""
tfidf.py — Modelo de espacio vectorial con ponderación TF-IDF y similitud coseno.

El ranking se basa en la similitud coseno entre los vectores TF-IDF de la
consulta y cada documento. La fórmula de peso usada es la variante logarítmica:

    w(t, d) = (1 + log TF(t,d)) · log(N / DF(t))

donde TF es la frecuencia bruta del término, DF es el número de documentos
que contienen el término y N es el tamaño del corpus.

Las normas de los documentos se precomputan en el constructor para no
recalcularlas en cada consulta — el cuello de botella sin este paso sería
O(|V| · |D|) por búsqueda. Las normas de la consulta se calculan en tiempo
de consulta porque varían según los términos ingresados.
"""

import numpy as np
from models.base import RetrievalModel


class TFIDFModel(RetrievalModel):
    """
    Modelo de espacio vectorial con TF-IDF logarítmico y similitud coseno.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus procesado. Se usa para obtener el tamaño del corpus (N).
    inv_index : dict[str, dict[int, int]]
        Índice invertido: término → {doc_id → TF}.
    """

    def __init__(self, df_corpus, inv_index):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.N = len(df_corpus)
        self.doc_norms = self._precompute_norms()

    def _precompute_norms(self) -> np.ndarray:
        """
        Calcula la norma L2 del vector TF-IDF de cada documento.

        Itera sobre el índice invertido en lugar del corpus para aprovechar
        la estructura dispersa: sólo procesa (término, doc) pares existentes.

        Returns
        -------
        np.ndarray
            Array de normas, indexado por posición en df_corpus.
        """
        doc_norms = np.zeros(self.N)
        for term, doc_freqs in self.inv_index.items():
            df_t = len(doc_freqs)
            idf = np.log(self.N / df_t)
            for doc_id, tf in doc_freqs.items():
                w = (1 + np.log(tf)) * idf
                doc_norms[doc_id] += w ** 2
        return np.sqrt(doc_norms)

    def search(self, query_processed: str, top_n: int = 5) -> tuple:
        """
        Rankea documentos por similitud coseno TF-IDF con la consulta.

        Calcula el producto punto entre el vector de la consulta y el de
        cada documento en el espacio TF-IDF, luego normaliza por las normas
        precomputadas para obtener la similitud coseno.

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
            - scores  : array de similitudes coseno para todos los documentos.
        """
        query_terms = query_processed.split()
        scores = np.zeros(self.N)
        query_norm_sq = 0.0

        for term in query_terms:
            if term not in self.inv_index:
                continue
            doc_freqs = self.inv_index[term]
            df_t = len(doc_freqs)
            idf = np.log(self.N / df_t)

            q_weight = (1 + np.log(query_terms.count(term))) * idf
            query_norm_sq += q_weight ** 2

            for doc_id, tf in doc_freqs.items():
                d_weight = (1 + np.log(tf)) * idf
                scores[doc_id] += d_weight * q_weight

        query_norm = np.sqrt(query_norm_sq)
        norm = self.doc_norms * query_norm
        scores = np.divide(scores, norm, out=np.zeros(self.N, dtype=float), where=norm != 0)

        ranking = scores.argsort()[::-1]
        return ranking[:top_n], scores
