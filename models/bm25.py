from models.base import RetrievalModel
import numpy as np

class BM25Model(RetrievalModel):
    def __init__(self, df_corpus, inv_index, k1=1.5, b=0.75):
        self.df_corpus = df_corpus
        self.inv_index = inv_index
        self.N = len(df_corpus)
        self.k1 = k1
        self.b = b
        self.doc_lengths = df_corpus['tokenized'].apply(len).values
        self.avgdl = self.doc_lengths.mean()

    def search(self, query_processed, top_n=5):
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