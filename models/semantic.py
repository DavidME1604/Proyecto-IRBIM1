import os
import faiss
from sentence_transformers import SentenceTransformer
from models.base import RetrievalModel
import numpy as np

class SemanticModel(RetrievalModel):
    def __init__(self, df_corpus, model_name='all-MiniLM-L6-v2', index_path='faiss_index.bin'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path

        if os.path.exists(index_path):
            print("Cargando índice FAISS desde disco...")
            self.faiss_index = faiss.read_index(index_path)
        else:
            print("Generando embeddings...")
            doc_embeddings = self.model.encode(df_corpus['raw'].tolist(), show_progress_bar=True)
            faiss.normalize_L2(doc_embeddings)

            dimension = doc_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(doc_embeddings.astype('float32'))

            faiss.write_index(self.faiss_index, index_path)
            print(f"Índice guardado en {index_path}")

    def search(self, query_text, top_n=5):
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        sim_scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_n)

        N = self.faiss_index.ntotal
        scores = np.zeros(N)
        ranking = indices.flatten()
        sim_scores = sim_scores.flatten()

        for i, idx in enumerate(ranking):
            scores[idx] = sim_scores[i]

        return ranking, scores