"""
qrels.py — Construcción de juicios de relevancia (qrels) desde el corpus Reuters.

Cada documento del dataset viene etiquetado con uno o más topics (p.ej.
'earn', 'grain', 'acq'). Este módulo los invierte para obtener, por cada
topic, la lista de documentos que lo contienen — que es exactamente la
definición de relevancia que usamos: si el usuario consulta "earn",
son relevantes todos los documentos etiquetados con ese topic.

El formato de los topics en el CSV es no estándar ("['grain' 'wheat']",
sin comas), por lo que se parsean con regex en lugar de ast.literal_eval.
"""

import re
import pandas as pd
from collections import defaultdict


def parse_topics(raw: str) -> list[str]:
    """
    Extrae los topics individuales del string de lista que guarda el CSV.

    El dataset serializa los topics como "['grain' 'wheat']" — una
    representación de lista de Python pero sin comas, lo que hace fallar
    ast.literal_eval. Se usa regex para extraer los fragmentos entre
    comillas simples.

    Parameters
    ----------
    raw : str
        String crudo de la columna 'topics', p.ej. "['earn' 'trade']".

    Returns
    -------
    list[str]
        Lista de topics, p.ej. ['earn', 'trade'].
    """
    return re.findall(r"'([^']+)'", str(raw))


def cargar_qrels(csv_path: str, min_docs: int = 10) -> dict:
    """
    Construye el diccionario de qrels a partir del CSV del corpus Reuters.

    Itera sobre cada documento, parsea sus topics y construye la relación
    inversa topic → documentos. Topics con muy pocos documentos relevantes
    se descartan porque no son informativos para la evaluación.

    Parameters
    ----------
    csv_path : str
        Ruta al archivo ModApte_train.csv. Debe contener la columna 'topics'.
    min_docs : int
        Umbral mínimo de documentos relevantes para incluir un topic como
        query de evaluación. El valor por defecto de 10 filtra topics
        extremadamente raros que generarían métricas poco representativas.

    Returns
    -------
    dict[str, list[int]]
        {topic: [doc_ids relevantes]} — los qrels listos para evaluar.
    """
    df = pd.read_csv(csv_path)
    qrels = defaultdict(list)
    for doc_id, row in df.iterrows():
        for topic in parse_topics(row['topics']):
            qrels[topic].append(doc_id)
    return {t: docs for t, docs in qrels.items() if len(docs) >= min_docs}
