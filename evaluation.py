"""
evaluation.py — Evaluación batch de modelos de recuperación de información.

Ejecuta un modelo sobre el conjunto completo de queries definidas en los
qrels y agrega las métricas resultantes (MAP, P@K promedio, R@K promedio)
junto con el desglose por consulta.

El diseño desacopla la evaluación del modelo concreto: `evaluar_modelo`
recibe un callable genérico `retrieve_fn(query_text) -> ranking`, de
modo que Jaccard, TF-IDF, BM25 y el modelo semántico se evalúan con la
misma lógica, variando únicamente la función pasada como argumento.
"""

from metrics import precision_at_k, recall_at_k, average_precision


def evaluar_modelo(retrieve_fn, qrels: dict, df_corpus, k: int) -> dict:
    """
    Evalúa un modelo de recuperación sobre todas las queries del set de qrels.

    Para cada query en qrels, obtiene el ranking del modelo, mapea los
    índices del corpus a doc_ids y calcula P@K, R@K y AP. Luego agrega
    las métricas individuales en MAP y promedios globales.

    Parameters
    ----------
    retrieve_fn : callable
        Función con firma query_text -> list[int], donde la salida es una
        lista de índices del corpus en orden de relevancia descendente.
        El caller es responsable de incluir el preprocesamiento de la
        query dentro de este callable si el modelo lo requiere.
    qrels : dict[str, list[int]]
        Juicios de relevancia: {topic: [doc_ids relevantes]}.
    df_corpus : pd.DataFrame
        Corpus indexado. Se usa exclusivamente para mapear posiciones del
        ranking a doc_ids reales mediante df_corpus.loc[idx, 'doc_id'].
    k : int
        Corte de ranking para calcular P@K y R@K.

    Returns
    -------
    dict
        Diccionario con las siguientes claves:
        - 'map'        : float — Mean Average Precision sobre todas las queries.
        - 'avg_p_at_k' : float — Promedio de P@K.
        - 'avg_r_at_k' : float — Promedio de R@K.
        - 'k'          : int   — Valor de K utilizado.
        - 'per_query'  : dict[str, dict] — Métricas individuales por query,
                         con claves f'P@{k}', f'R@{k}' y 'AP'.
    """
    per_query = {}
    for query, relevantes in qrels.items():
        ranking = retrieve_fn(query)
        doc_ids = [df_corpus.loc[idx, 'doc_id'] for idx in ranking]
        per_query[query] = {
            f'P@{k}': precision_at_k(doc_ids, relevantes, k),
            f'R@{k}': recall_at_k(doc_ids, relevantes, k),
            'AP':     average_precision(doc_ids, relevantes),
        }

    n = len(per_query)
    return {
        'map':        sum(m['AP']      for m in per_query.values()) / n,
        'avg_p_at_k': sum(m[f'P@{k}'] for m in per_query.values()) / n,
        'avg_r_at_k': sum(m[f'R@{k}'] for m in per_query.values()) / n,
        'k':          k,
        'per_query':  per_query,
    }
