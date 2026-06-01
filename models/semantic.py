"""
semantic.py — Modelo de recuperación semántica con embeddings y FAISS.

A diferencia de los modelos léxicos, este modelo captura similitud semántica
entre consulta y documentos sin depender de coincidencia exacta de términos.
Usa SentenceTransformers para generar embeddings de 384 dimensiones y FAISS
para búsqueda por similitud coseno (inner product sobre vectores normalizados)
en tiempo sub-lineal.

Pipeline:
    1. Codifica todos los documentos con 'all-MiniLM-L6-v2'.
    2. Normaliza los vectores en L2 y los indexa en un IndexFlatIP de FAISS.
    3. En tiempo de consulta, codifica la query y busca los top-N vecinos
       más cercanos por producto interno (equivalente a coseno tras normalizar).

El índice FAISS se persiste en disco para evitar re-codificar el corpus
en cada ejecución — el paso de encoding es el cuello de botella (~minutos
en CPU para corpus grandes).

Referencia del modelo:
    Wang et al. (2020). MINILM: Deep Self-Attention Distillation for Task-Agnostic
    Compression of Pre-Trained Transformers. NeurIPS 2020.
"""

import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from models.base import RetrievalModel


class SemanticModel(RetrievalModel):
    """
    Recuperación semántica densa usando SentenceTransformers + FAISS.

    Parameters
    ----------
    df_corpus : pd.DataFrame
        Corpus con columna 'raw' (texto original sin procesar). Se usa el
        texto crudo —no el preprocesado— para que el modelo de lenguaje
        reciba texto completo con contexto morfológico.
    model_name : str
        Identificador del modelo de SentenceTransformers a cargar.
    index_path : str
        Ruta donde se guarda/carga el índice FAISS serializado.
    """

    def __init__(self, df_corpus,
                 model_name: str = 'all-MiniLM-L6-v2',
                 index_path: str = 'faiss_index.bin'):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path

        if os.path.exists(index_path):
            self.faiss_index = faiss.read_index(index_path)
        else:
            doc_embeddings = self.model.encode(
                df_corpus['raw'].tolist(), show_progress_bar=True
            )
            faiss.normalize_L2(doc_embeddings)

            dimension = doc_embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)
            self.faiss_index.add(doc_embeddings.astype('float32'))
            faiss.write_index(self.faiss_index, index_path)

    def search(self, query_text: str, top_n: int = 5) -> tuple:
        """
        Rankea documentos por similitud semántica coseno con la consulta.

        La consulta se codifica con el mismo modelo usado para los documentos.
        La búsqueda en FAISS retorna sólo los top_n vecinos; los scores del
        resto del corpus se dejan en 0.0, lo que es consistente con la
        interfaz de los demás modelos.

        Parameters
        ----------
        query_text : str
            Texto crudo de la consulta (sin preprocesar). Los modelos de
            lenguaje como MiniLM trabajan sobre texto natural completo.
        top_n : int
            Número de documentos a retornar.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - ranking : top_n índices del corpus en orden descendente de score.
            - scores  : array de similitudes coseno; sólo los top_n tienen
                        valor no nulo (FAISS sólo devuelve los vecinos pedidos).
        """
        query_embedding = self.model.encode([query_text], normalize_embeddings=True)
        sim_scores, indices = self.faiss_index.search(
            query_embedding.astype('float32'), top_n
        )

        N = self.faiss_index.ntotal
        scores = np.zeros(N)
        ranking = indices.flatten()
        sim_scores = sim_scores.flatten()

        for i, idx in enumerate(ranking):
            scores[idx] = sim_scores[i]

        return ranking, scores
