from models.base import RetrievalModel
import numpy as np

class JaccardModel(RetrievalModel):
    
    def __init__(self, df_corpus, inv_index):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.doc_unique_terms = df_corpus['tokenized'].apply(lambda t: len(set(t))).values

    def search(self, query_processed, top_n=5):
        query_terms = set(query_processed.split())
        N = len(self.df_corpus)
        intersection = np.zeros(N)
        for term in query_terms:
            if term in self.inv_index:
                for doc_id in self.inv_index[term]:
                    intersection[doc_id] += 1
        union = self.doc_unique_terms + len(query_terms) - intersection
        scores = np.divide(intersection, union, out=np.zeros(N, dtype=float), where=union != 0)
        ranking = scores.argsort()[::-1]
        return ranking[:top_n], scores