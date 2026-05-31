"""
ui.py — Interfaz CLI del Sistema de Recuperación de Información.

Este módulo se encarga únicamente del renderizado y la interacción con
el usuario. La lógica de recuperación (Jaccard, TF-IDF, BM25, semántico)
vive en proyecto_recuperacion.py y se enchufa dentro de loop_principal().

Para conectar un modelo:
    1. Importar la función de recuperación correspondiente.
    2. Construir la lista `resultados` con el formato:
           [(rank, doc_id, score, titulo), ...]
    3. Pasarla a mostrar_resultados().

Buscar "TODO: CONECTAR" para localizar los puntos de integración.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.align import Align
from rich import box
from metrics import precision_at_k, recall_at_k, average_precision
from proyecto_recuperacion import (
    process_query, jaccard_similarity, tf_idf_similarity,
)

# ─── Configuración ───────────────────────────────────────────────────

K = 10  # top-k de documentos a recuperar por modelo

VIOLETA = "#a78bfa"
AZUL    = "#60a5fa"
ROSA    = "#f472b6"
GRIS    = "#9ca3af"

console = Console()

MODELOS = {
    '1': 'Jaccard',
    '2': 'TF-IDF Coseno',
    '3': 'BM25',
    '4': 'Semántico',
    '5': 'Todos',
}


# ─── Componentes visuales ────────────────────────────────────────────

def mostrar_header():
    # Banner principal. Se imprime una sola vez al iniciar la sesión.
    titulo = Text("Sistema de Recuperación de Información", style=f"bold {VIOLETA}")
    subtitulo = Text("Proyecto 1er Bimestre — Recuperación de Información", style=f"italic {GRIS}")
    contenido = Align.center(Text.assemble(titulo, "\n", subtitulo))
    console.print(Panel(contenido, border_style=VIOLETA, box=box.ROUNDED, padding=(1, 2)))


def mostrar_menu_modelos():
    # Lista numerada de modelos. Las claves coinciden con las de MODELOS.
    tabla = Table(show_header=False, box=None, padding=(0, 2))
    tabla.add_column(style=f"bold {AZUL}", width=3)
    tabla.add_column(style="white")
    for key, nombre in MODELOS.items():
        tabla.add_row(f"{key}.", nombre)
    panel = Panel(tabla, title=f"[bold {VIOLETA}]Modelos disponibles[/]",
                  border_style=GRIS, box=box.ROUNDED, title_align="left")
    console.print(panel)


def mostrar_resultados(model_name: str, query: str, resultados):
    """
    Imprime un panel con la tabla de top-K resultados de un modelo.

    Parameters
    ----------
    model_name : str
        Nombre del modelo (aparece en el título del panel).
    query : str
        Texto original de la consulta (sólo para mostrar en el título).
    resultados : list[tuple] | None
        Lista de tuplas (rank, doc_id, score, titulo) en orden de
        relevancia. Si es None, se muestra un placeholder indicando
        que el modelo aún no está conectado.
    """
    tabla = Table(box=box.SIMPLE_HEAD, header_style=f"bold {VIOLETA}", expand=True)
    tabla.add_column("Rank",   style=f"bold {AZUL}",  width=6,  justify="right")
    tabla.add_column("Doc ID", style=GRIS,            width=10)
    tabla.add_column("Score",  style=f"bold {ROSA}",  width=10, justify="right")
    tabla.add_column("Título", style="white", overflow="ellipsis", no_wrap=True)

    if resultados:
        for rank, doc_id, score, titulo in resultados:
            tabla.add_row(str(rank), str(doc_id), f"{score:.4f}", str(titulo))
    else:
        tabla.add_row("—", "—", "—", "[dim italic]Modelo no conectado todavía[/]")

    titulo_panel = f"[bold {VIOLETA}]{model_name}[/] [dim]· top-{K} · query: \"{query}\"[/]"
    console.print(Panel(tabla, title=titulo_panel, border_style=VIOLETA,
                        box=box.ROUNDED, title_align="left"))


def mostrar_metricas(model_name: str, metricas: dict, info_qrels: str = ""):
    """
    Imprime un panel compacto con métricas de evaluación para una consulta.

    Parameters
    ----------
    model_name : str
        Nombre del modelo (aparece en el título).
    metricas : dict[str, float]
        Diccionario {nombre: valor}, ej: {'P@10': 0.6, 'R@10': 0.3, 'AP': 0.42}.
    info_qrels : str
        Texto opcional para el título, ej: "qrels: 'earn' · 2877 docs".
    """
    tabla = Table(box=box.SIMPLE, header_style=f"bold {VIOLETA}", expand=True)
    for nombre in metricas:
        tabla.add_column(nombre, justify="center", style=f"bold {ROSA}")
    tabla.add_row(*[f"{v:.4f}" for v in metricas.values()])

    titulo = f"[bold {VIOLETA}]Métricas — {model_name}[/]"
    if info_qrels:
        titulo += f" [dim]· {info_qrels}[/]"
    console.print(Panel(tabla, title=titulo, border_style=ROSA,
                        box=box.ROUNDED, title_align="left"))

def mostrar_evaluacion(resultados_por_modelo: dict, n_queries: int):
    """
    Imprime una tabla comparativa de evaluación batch entre modelos.

    Parameters
    ----------
    resultados_por_modelo : dict[str, dict]
        {nombre_modelo: dict devuelto por evaluation.evaluar_modelo()}
    n_queries : int
        Número de queries usadas en la evaluación (aparece en el título).
    """
    if not resultados_por_modelo:
        console.print(f"[{ROSA}]No hay resultados para mostrar.[/]")
        return

    k = next(iter(resultados_por_modelo.values()))['k']

    tabla = Table(box=box.SIMPLE_HEAD, header_style=f"bold {VIOLETA}", expand=True)
    tabla.add_column("Modelo",      style="white",         width=20)
    tabla.add_column("MAP",         style=f"bold {ROSA}",  justify="right")
    tabla.add_column(f"avg P@{k}",  style=f"bold {AZUL}",  justify="right")
    tabla.add_column(f"avg R@{k}",  style=GRIS,            justify="right")

    for modelo, r in resultados_por_modelo.items():
        tabla.add_row(
            modelo,
            f"{r['map']:.4f}",
            f"{r['avg_p_at_k']:.4f}",
            f"{r['avg_r_at_k']:.4f}",
        )

    titulo = f"[bold {VIOLETA}]Evaluación batch[/] [dim]· {n_queries} queries · k={k}[/]"
    console.print(Panel(tabla, title=titulo, border_style=ROSA,
                        box=box.ROUNDED, title_align="left"))

# ─── Entrada del usuario ─────────────────────────────────────────────

def pedir_query() -> str:
    return Prompt.ask(f"\n[bold {VIOLETA}]Consulta[/] [dim](o 'salir')[/]").strip()


def pedir_modelo() -> str:
    return Prompt.ask(f"[bold {VIOLETA}]Seleccione modelo[/]",
                      choices=list(MODELOS.keys()), show_choices=False).strip()

# ─── Ejecución de modelos ─────────────────────────────────────────────
def ejecutar_modelo(nombre: str, retrieve_fn, query: str,
                    df_corpus, qrels: dict | None):
    """
    Corre un modelo, muestra resultados, y muestra métricas si la query
    coincide con un topic con qrels.

    Parameters
    ----------
    nombre : str
        Nombre del modelo (aparece en los títulos de los paneles).
    retrieve_fn : callable
        query_procesada -> (ranking, scores). Envuelve la firma específica
        del modelo (matrices, vectorizadores, etc.) en un closure/lambda.
    query : str
        Texto crudo ingresado por el usuario.
    """
    query_proc = process_query(query)['processed']
    ranking, scores = retrieve_fn(query_proc)

    resultados = [
        (i + 1, df_corpus.loc[idx, 'doc_id'], scores[idx], df_corpus.loc[idx, 'title'])
        for i, idx in enumerate(ranking[:K])
    ]
    mostrar_resultados(nombre, query, resultados)

    if qrels and query.lower() in qrels:
        relevantes = qrels[query.lower()]
        doc_ids_ranking = [df_corpus.loc[idx, 'doc_id'] for idx in ranking]
        metricas = {
            f'P@{K}': precision_at_k(doc_ids_ranking, relevantes, K),
            f'R@{K}': recall_at_k(doc_ids_ranking, relevantes, K),
            'AP':     average_precision(doc_ids_ranking, relevantes),
        }
        mostrar_metricas(nombre, metricas,
                         f"qrels: '{query.lower()}' · {len(relevantes)} docs")
        
# ─── Loop principal ──────────────────────────────────────────────────

def loop_principal(qrels: dict | None, df_corpus, recursos: dict):
    """
    Loop interactivo principal de la CLI.

    Parameters
    ----------
    qrels : dict[str, list] | None
        {topic: [doc_ids relevantes]} para evaluación. Si es None,
        las métricas no se muestran.
    df_corpus : pd.DataFrame
        Corpus indexado con columnas 'doc_id', 'title', 'processed'.
    recursos : dict
        Matrices y vectorizadores precomputados. Claves esperadas:
            'binary_matrix', 'binary_vec' (Jaccard)
            'tfidf_matrix',  'tfidf_vec'  (TF-IDF)
    """
    console.clear()
    mostrar_header()
    if qrels:
        console.print(f"[{GRIS}]Qrels cargados: {len(qrels)} topics disponibles para evaluación.[/]\n")

    while True:
        query = pedir_query()
        if query.lower() == 'salir':
            console.print(f"\n[{VIOLETA}]Hasta luego![/]\n")
            break
        if query.lower() == 'evaluar':

            # TODO: EVALUACION BATCH
            # Requiere los modelos conectados (sección "CONECTAR" de abajo).
            # Patrón:
            #
            #   from evaluation import evaluar_modelo
            #
            #   retrieve_jaccard = lambda q: jaccard_similarity(
            #       process_query(q)['processed'], binary_matrix, binary_vec
            #   )[0]
            #   retrieve_tfidf   = lambda q: tf_idf_similarity(
            #       process_query(q)['processed'], tfidf_matrix, vectorizer_tfidf
            #   )[0]
            #   # ...mismo patrón para BM25 y semántico
            #
            #   with console.status(f"[{VIOLETA}]Ejecutando evaluación batch..."):
            #       resultados = {
            #           'Jaccard':       evaluar_modelo(retrieve_jaccard, qrels, df_corpus, K),
            #           'TF-IDF Coseno': evaluar_modelo(retrieve_tfidf,   qrels, df_corpus, K),
            #           'BM25':          evaluar_modelo(retrieve_bm25,    qrels, df_corpus, K),
            #           'Semántico':     evaluar_modelo(retrieve_sem,     qrels, df_corpus, K),
            #       }
            #   mostrar_evaluacion(resultados, len(qrels))
            console.print(f"[{ROSA}]Evaluación batch no conectada todavía.[/]")
            continue
        if not query:
            console.print(f"[{ROSA}]La consulta no puede estar vacía.[/]\n")
            continue

        mostrar_menu_modelos()
        opcion = pedir_modelo()

        # ─────────────────────────────────────────────────────────────
        # Punto de integración con los modelos de recuperación.
        #
        # Cada bloque debe armar una lista `resultados` con el formato:
        #     [(rank, doc_id, score, titulo), ...]
        # y pasarla a mostrar_resultados(). Mientras `resultados` sea
        # None, el panel muestra un placeholder.
        # ─────────────────────────────────────────────────────────────

        if opcion in ('1', '5'):
            ejecutar_modelo(
                "Jaccard",
                lambda q: jaccard_similarity(q, recursos['binary_matrix'], recursos['binary_vec']),
                query, df_corpus, qrels,
            )
        if opcion in ('2', '5'):
            ejecutar_modelo(
                "TF-IDF Coseno",
                lambda q: tf_idf_similarity(q, recursos['tfidf_matrix'], recursos['tfidf_vec']),
                query, df_corpus, qrels,
            )

        if opcion in ('3', '5'):
            # TODO: CONECTAR BM25
            resultados = None
            mostrar_resultados("BM25", query, resultados)

        if opcion in ('4', '5'):
            # TODO: CONECTAR Semántico (embeddings)
            resultados = None
            mostrar_resultados("Semántico", query, resultados)


if __name__ == "__main__":
    from proyecto_recuperacion import (
        process_corpus, build_binary_matrix, build_tfidf_matrix,
    )
    from qrels import cargar_qrels
    CSV_PATH = r"ModApte_train.csv"
    with console.status(f"[{VIOLETA}]Cargando corpus..."):
        df_corpus = process_corpus(CSV_PATH)
    with console.status(f"[{VIOLETA}]Construyendo índices..."):
        binary_matrix, binary_vec = build_binary_matrix(df_corpus)
        tfidf_matrix,  tfidf_vec  = build_tfidf_matrix(df_corpus)
    with console.status(f"[{VIOLETA}]Cargando qrels..."):
        qrels = cargar_qrels(CSV_PATH)
    recursos = {
        'binary_matrix': binary_matrix, 'binary_vec': binary_vec,
        'tfidf_matrix':  tfidf_matrix,  'tfidf_vec':  tfidf_vec,
    }

    loop_principal(qrels, df_corpus, recursos)