"""
metrics.py — Métricas de evaluación de Recuperación de Información.

Implementaciones desde cero (sin sklearn). Trabajan sobre listas de
doc_ids: `retrieved` viene del ranking de tus modelos, `relevant` viene
de los qrels (juicios de relevancia).
"""


def precision_at_k(retrieved: list, relevant, k: int) -> float:
    """P@K = |top-k recuperados ∩ relevantes| / k"""
    if k == 0:
        return 0.0
    relevant = set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list, relevant, k: int) -> float:
    """R@K = |top-k recuperados ∩ relevantes| / |relevantes|"""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list, relevant) -> float:
    """
    AP = (1/|relevantes|) · Σ P@i · rel(i)

    Premia que los documentos relevantes aparezcan en posiciones altas
    del ranking, no sólo que estén en la lista.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    suma = 0.0
    hits = 0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            suma += hits / i  # precisión en la posición i
    return suma / len(relevant)


def mean_average_precision(retrieved_por_query: dict, relevant_por_query: dict) -> float:
    """
    MAP = promedio del AP sobre todas las queries de evaluación.

    Parameters
    ----------
    retrieved_por_query : dict[str, list]
        {query: [doc_ids recuperados en orden de ranking]}
    relevant_por_query : dict[str, set | list]
        {query: doc_ids relevantes}  (los qrels)
    """
    if not retrieved_por_query:
        return 0.0
    aps = [
        average_precision(retrieved, relevant_por_query.get(q, set()))
        for q, retrieved in retrieved_por_query.items()
    ]
    return sum(aps) / len(aps)