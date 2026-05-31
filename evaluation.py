"""
evaluation.py — Evaluación batch de modelos de recuperación.

Corre un modelo sobre todas las queries del set de qrels y devuelve
sus métricas agregadas (MAP, promedios de P@K y R@K) más el detalle
por consulta.

Diseño: `evaluar_modelo` recibe un callable `retrieve_fn(query_text)`
que devuelve un ranking de índices del corpus. Eso desacopla la
evaluación del modelo concreto — Jaccard, TF-IDF, BM25 y semántico
se evalúan con la misma función, sólo cambia el callable.
"""

from metrics import precision_at_k, recall_at_k, average_precision


def evaluar_modelo(retrieve_fn, qrels: dict, df_corpus, k: int) -> dict:
    """
    Ejecuta un modelo sobre todas las queries de qrels y agrega métricas.

    Parameters
    ----------
    retrieve_fn : callable
        Función query_text -> ranking (lista/array de índices del corpus
        en orden de relevancia). El caller es responsable de envolver el
        preprocesamiento de la query dentro de este callable.
    qrels : dict[str, list[int]]
        {topic: [doc_ids relevantes]}.
    df_corpus : pd.DataFrame
        Corpus indexado. Sólo se usa para mapear índices del ranking
        -> doc_ids vía df_corpus.loc[idx, 'doc_id'].
    k : int
        Top-k para calcular P@K y R@K.

    Returns
    -------
    dict
        {
            'map':        float,
            'avg_p_at_k': float,
            'avg_r_at_k': float,
            'k':          int,
            'per_query':  {topic: {'P@K': ..., 'R@K': ..., 'AP': ...}}
        }
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
        'map':        sum(m['AP']     for m in per_query.values()) / n,
        'avg_p_at_k': sum(m[f'P@{k}'] for m in per_query.values()) / n,
        'avg_r_at_k': sum(m[f'R@{k}'] for m in per_query.values()) / n,
        'k':          k,
        'per_query':  per_query,
    }