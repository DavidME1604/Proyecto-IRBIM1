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


# ─── Entrada del usuario ─────────────────────────────────────────────

def pedir_query() -> str:
    return Prompt.ask(f"\n[bold {VIOLETA}]Consulta[/] [dim](o 'salir')[/]").strip()


def pedir_modelo() -> str:
    return Prompt.ask(f"[bold {VIOLETA}]Seleccione modelo[/]",
                      choices=list(MODELOS.keys()), show_choices=False).strip()


# ─── Loop principal ──────────────────────────────────────────────────

def loop_principal():
    console.clear()
    mostrar_header()

    while True:
        query = pedir_query()
        if query.lower() == 'salir':
            console.print(f"\n[{VIOLETA}]Hasta luego![/]\n")
            break
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
            # TODO: CONECTAR Jaccard
            # ranking, scores = jaccard_similarity(query_proc, binary_matrix, binary_vec)
            # resultados = [(i+1, df.loc[idx,'doc_id'], scores[idx], df.loc[idx,'title'])
            #               for i, idx in enumerate(ranking[:K])]
            resultados = None
            mostrar_resultados("Jaccard", query, resultados)

        if opcion in ('2', '5'):
            # TODO: CONECTAR TF-IDF
            resultados = None
            mostrar_resultados("TF-IDF Coseno", query, resultados)

        if opcion in ('3', '5'):
            # TODO: CONECTAR BM25
            resultados = None
            mostrar_resultados("BM25", query, resultados)

        if opcion in ('4', '5'):
            # TODO: CONECTAR Semántico (embeddings)
            resultados = None
            mostrar_resultados("Semántico", query, resultados)


if __name__ == "__main__":
    loop_principal()