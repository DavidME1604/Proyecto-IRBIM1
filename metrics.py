"""
metrics.py — Métricas estándar de evaluación para sistemas de recuperación.

Implementaciones desde cero de P@K, R@K, AP y MAP. Todas las funciones
operan sobre listas de doc_ids: `retrieved` proviene del ranking del
modelo y `relevant` de los juicios de relevancia (qrels).

Referencias:
    Manning, Raghavan & Schütze — Introduction to Information Retrieval,
    Cap. 8: Evaluation in information retrieval.
"""


def precision_at_k(retrieved: list, relevant, k: int) -> float:
    """
    Calcula Precision@K: fracción de documentos relevantes en el top-K.

    P@K = |retrieved[:K] ∩ relevant| / K

    Parameters
    ----------
    retrieved : list
        Lista de doc_ids en orden de ranking (más relevante primero).
    relevant : iterable
        Conjunto de doc_ids considerados relevantes para la consulta.
    k : int
        Corte de ranking a evaluar.

    Returns
    -------
    float
        Valor en [0, 1]. Retorna 0.0 si k == 0.
    """
    if k == 0:
        return 0.0
    relevant = set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list, relevant, k: int) -> float:
    """
    Calcula Recall@K: fracción del total de relevantes recuperados en top-K.

    R@K = |retrieved[:K] ∩ relevant| / |relevant|

    Parameters
    ----------
    retrieved : list
        Lista de doc_ids en orden de ranking.
    relevant : iterable
        Conjunto de doc_ids relevantes para la consulta.
    k : int
        Corte de ranking a evaluar.

    Returns
    -------
    float
        Valor en [0, 1]. Retorna 0.0 si el conjunto de relevantes está vacío.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list, relevant) -> float:
    """
    Calcula Average Precision (AP) para una sola consulta.

    AP = (1/|relevant|) · Σ P@i · rel(i)

    A diferencia de P@K, AP penaliza cuando los documentos relevantes
    aparecen en posiciones bajas del ranking, no sólo si están presentes.

    Parameters
    ----------
    retrieved : list
        Lista de doc_ids en orden de ranking.
    relevant : iterable
        Conjunto de doc_ids relevantes para la consulta.

    Returns
    -------
    float
        Valor en [0, 1]. Retorna 0.0 si el conjunto de relevantes está vacío.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    suma = 0.0
    hits = 0
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            hits += 1
            suma += hits / i
    return suma / len(relevant)


def mean_average_precision(retrieved_por_query: dict, relevant_por_query: dict) -> float:
    """
    Calcula Mean Average Precision (MAP) sobre un conjunto de consultas.

    MAP = (1/|Q|) · Σ AP(q)  para q en Q

    Es la métrica de ranking más usada en evaluación de IR porque captura
    tanto precisión como ordenamiento a lo largo de todas las consultas.

    Parameters
    ----------
    retrieved_por_query : dict[str, list]
        {query: [doc_ids recuperados en orden de ranking]}.
    relevant_por_query : dict[str, set | list]
        {query: doc_ids relevantes} — los qrels.

    Returns
    -------
    float
        Valor en [0, 1]. Retorna 0.0 si el diccionario de resultados está vacío.
    """
    if not retrieved_por_query:
        return 0.0
    aps = [
        average_precision(retrieved, relevant_por_query.get(q, set()))
        for q, retrieved in retrieved_por_query.items()
    ]
    return sum(aps) / len(aps)
