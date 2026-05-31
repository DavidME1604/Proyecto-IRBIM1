"""
qrels.py — Carga de juicios de relevancia (qrels) desde el dataset Reuters.

Cada documento del corpus viene etiquetado con sus topics (columna 'topics').
Usamos los topics como queries de evaluación: si el usuario consulta "earn",
los documentos relevantes son todos los que tienen "earn" entre sus topics.

Uso:
    from qrels import cargar_qrels
    qrels = cargar_qrels("ruta/a/ModApte_train.csv")
    relevantes = qrels.get("earn", [])
"""

import re
import pandas as pd
from collections import defaultdict


def parse_topics(raw: str) -> list:
    """
    Convierte el string "['grain' 'wheat']" en ['grain', 'wheat'].

    El dataset guarda los topics con un formato no estándar (lista sin comas),
    así que ast.literal_eval falla. Se extraen con regex los fragmentos
    entre comillas simples.
    """
    return re.findall(r"'([^']+)'", str(raw))


def cargar_qrels(csv_path: str, min_docs: int = 10) -> dict:
    """
    Construye el diccionario de qrels a partir del CSV.

    Parameters
    ----------
    csv_path : str
        Ruta al CSV (debe tener columna 'topics').
    min_docs : int
        Mínimo de documentos relevantes para que un topic se incluya como
        query de evaluación. Filtra topics demasiado raros.

    Returns
    -------
    dict[str, list[int]]
        Diccionario {topic: [doc_ids relevantes]}.
    """
    df = pd.read_csv(csv_path)
    qrels = defaultdict(list)
    for doc_id, row in df.iterrows():
        for topic in parse_topics(row['topics']):
            qrels[topic].append(doc_id)
    return {t: docs for t, docs in qrels.items() if len(docs) >= min_docs}