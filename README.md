# Sistema de Recuperación de Información

Sistema que compara 4 modelos de recuperación (Jaccard, TF-IDF, BM25 y semántico) sobre el corpus Reuters.

## Requisitos

- Python 3.10+
- Cuenta en [Kaggle](https://www.kaggle.com) con el archivo de credenciales configurado en `~/.kaggle/kaggle.json` (se obtiene desde *Settings → API → Create New Token*)

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

La primera ejecución:
1. Descarga el dataset Reuters desde Kaggle (`ModApte_train.csv`, ~9000 docs)
2. Procesa el corpus y construye el índice invertido (se cachea en `cache/`)
3. Genera embeddings con `all-MiniLM-L6-v2` y construye el índice FAISS (se guarda en `faiss_index.bin`)

Las ejecuciones posteriores cargan todo desde caché en segundos.

## Uso

La CLI permite:
- Escribir consultas de texto libre
- Seleccionar un modelo (1-4) o ejecutar todos (5)
- Ver el ranking de documentos con scores
- Si la query coincide con un topic de Reuters, muestra métricas (P@K, R@K, AP) automáticamente
- Escribir `evaluar` para ejecutar la evaluación batch (61 queries, 4 modelos)
- Escribir `salir` para terminar

## Estructura

```
├── main.py              # CLI interactiva (Rich)
├── preprocessing.py     # Pipeline de texto, índice invertido, caché
├── models/
│   ├── base.py          # Clase abstracta RetrievalModel
│   ├── jaccard.py       # Similitud Jaccard (vectores binarios)
│   ├── tfidf.py         # TF-IDF con similitud coseno
│   ├── bm25.py          # BM25 (k1=1.5, b=0.75)
│   └── semantic.py      # Embeddings (all-MiniLM-L6-v2) + FAISS
├── metrics.py           # P@K, R@K, AP, MAP
├── evaluation.py        # Evaluación batch sobre qrels
├── qrels.py             # Extracción de qrels desde topics de Reuters
├── requirements.txt
└── Informe_RI.ipynb 
```

## Notas

- Si se modifica el preprocesamiento, borrar la carpeta `cache/` para regenerar.
- Para regenerar los embeddings, borrar `faiss_index.bin`.
