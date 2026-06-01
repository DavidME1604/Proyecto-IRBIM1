"""
base.py — Clase base abstracta para los modelos de recuperación.

Define la interfaz mínima que debe implementar cualquier modelo del sistema.
Usar ABC fuerza que las subclases implementen `search` antes de poder
instanciarse, lo que previene errores silenciosos si se agrega un nuevo
modelo sin implementar el método principal.
"""

from abc import ABC, abstractmethod


class RetrievalModel(ABC):
    """Interfaz común para todos los modelos de recuperación de información."""

    @abstractmethod
    def search(self, query_processed: str, top_n: int = 5) -> tuple:
        """
        Retorna el ranking de documentos para una consulta preprocesada.

        Parameters
        ----------
        query_processed : str
            Texto de la consulta ya normalizado (stems separados por espacio)
            para modelos léxicos, o texto crudo para el modelo semántico.
        top_n : int
            Número de documentos a retornar en el ranking.

        Returns
        -------
        tuple[array-like, array-like]
            - ranking : índices del corpus ordenados por relevancia descendente.
            - scores  : array de puntuaciones para todos los documentos del
                        corpus (índice = posición en df_corpus).
        """
