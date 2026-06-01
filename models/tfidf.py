import numpy as np
from models.base import RetrievalModel

class TFIDFModel(RetrievalModel):
    def __init__(self, df_corpus, inv_index):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.N = len(df_corpus)
        self.doc_norms = self._precompute_norms()

    def _precompute_norms(self):
        doc_norms = np.zeros(self.N)
        for term, doc_freqs in self.inv_index.items():
            df_t = len(doc_freqs)
            idf = np.log(self.N / df_t)
            for doc_id, tf in doc_freqs.items():
                w = (1 + np.log(tf)) * idf
                doc_norms[doc_id] += w ** 2
        return np.sqrt(doc_norms)

    def search(self, query_processed, top_n=5):
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